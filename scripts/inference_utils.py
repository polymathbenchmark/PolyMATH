import os
import ast
import json
import time
import argparse
from tqdm import tqdm
import pandas as pd

from inference_models import inference_model_factory

"""
This script contains the function for multimodal inference
"""

parser = argparse.ArgumentParser(description="Run inference and utility scripts on multimodal models")
parser.add_argument('-m', '--mode', help='The inference utility mode', required=False)
parser.add_argument("-mn", "--model_name",
                    help="The model name/path endpoint, model-id from huggingface or local directory",
                    required=True)
parser.add_argument('-pp', '--paper_path', help='The name of paper to be evaluated', required=False)
parser.add_argument('-dp', '--datastore_path',
                    help='The datastore path that contains the papers to be evaluated',
                    required=False)
parser.add_argument('-drs', '--datastore_range_start', type=int,
                    help='The subset of datastore files that contains the papers to be evaluated',
                    required=False)
parser.add_argument('-dre', '--datastore_range_end', type=int,
                    help='The subset of datastore files that contains the papers to be evaluated',
                    required=False)
parser.add_argument('-isq', '--is_quantization', type=bool,
                    help='Whether to run in quantized mode.', required=False)
args = vars(parser.parse_args())

if __name__ == "__main__":
    root_path = "../"

    with open("./scripts/prompts.json") as f:
        CONFIG = json.load(f)
        f.close()

    if (args["mode"] == "categorization") and (args["model_name"] not in CONFIG["CATEGORIZATION_ALLOW_LISTED_MODELS"]):
        raise ValueError(f"{args['model_name']} is currently not supported. "
                         f"Try one of {CONFIG['CATEGORIZATION_ALLOW_LISTED_MODELS']}")
    
    if args.get("datastore_path"):
    
        datastore_path = args["datastore_path"]

        paper_paths = [os.path.join(datastore_path, i) for i in os.listdir(datastore_path)
                       if i not in ['metadata.json', '.DS_Store', 'model_run.json', 'final_datasets']]
        paper_paths = [paper_path for paper_path in paper_paths if 'merged_screenshots'
                       in os.listdir(paper_path)]
        
        if (args.get("datastore_range_start")) is not None and (args.get("datastore_range_end") is not None):
            start_idx, end_idx = args["datastore_range_start"], args["datastore_range_end"]
        else:
            start_idx, end_idx = None, None
            print(f"Running inference on all {len(paper_paths)} files in datastore path. "
                  "If you only need to run inference on subset, pass the datastore_range_start and"
                  " datastore_range_end arguments.")
    elif args.get("paper_path"):
        start_idx, end_idx = None, None
        datastore_path = os.path.dirname(args["paper_path"])
        paper_paths = [args["paper_path"]]
    else:
        start_idx, end_idx = None, None
        raise ValueError("One of --paper_path or --datastore_path value is required!")

    if (start_idx is not None) and (end_idx is not None):
        try:
            print("**** Stats ****")
            print("Total datastore size: ", len(paper_paths))
            paper_paths = paper_paths[start_idx:end_idx]
            print("Total datastore size chosen for inference: ", len(paper_paths))
            print(f"Running inference for file with index range: {start_idx} to {end_idx}")
        except IndexError:
            raise IndexError(f"start_idx and end_idx is too restrictive for datastore sized {len(paper_paths)}")

    if args["mode"] == "inference":
        PROMPT_PARTS = CONFIG["inference"]
        PROMPT = PROMPT_PARTS["common_prefix"] + PROMPT_PARTS["zeroshot"]
        MODE = "INFERENCE"
    elif args["mode"] == "categorization":
        PROMPT = CONFIG["categorization"]
        MODE = "CATEGORIZATION"
    else:
        raise ValueError(f"Unknown mode {args['mode']}. Mode should be one of `inference` or `categorization`.")

    model = inference_model_factory(model_id=args["model_name"], is_quantization=args["is_quantization"])
    
    for paper_path_idx in range(len(paper_paths)):
        print(
            f"----------------- {paper_path_idx + 1}/{len(paper_paths)} | "
            f"{round((paper_path_idx + 1) * 100 / len(paper_paths))}% -----------------")
        paper_path = paper_paths[paper_path_idx]
        paper_name = os.path.basename(os.path.normpath(paper_path))
        model_response_dict = dict()
        
        # Load annotations.csv file to check if a model inference is completed for a certain file
        # try:
        #     annotations_df = pd.read_csv(os.path.join(paper_path, "annotations.csv"))
        # except:
        annotation_csv = pd.read_csv(os.path.join(paper_path, "annotations_final.csv"))
        current_columns = set(annotation_csv.columns)
        if args["model_name"]+"-response" in current_columns:
            # Inference run already completed successfully. Ignore the loop
            print(f"\033[92m{MODE} STATUS of {paper_path} with model {args['model_name']}: ALREADY COMPLETED",
                    '\u2713\033[0m')
            continue
                    
        try:
            merged_screenshot_path = os.path.join(paper_path, 'merged_screenshots')
            screenshots = os.listdir(merged_screenshot_path)
            annotation_csv = annotation_csv[~annotation_csv['paper_id-input'].isna()]

            pbar = tqdm(range(len(screenshots)))
            # pbar = tqdm(range(3))
            parse_failed_count = 0

            for idx in pbar:
                file = screenshots[idx]
                q_no = file.split(".")[0]

                pbar.set_description(f"Inference paper: {paper_name} | {q_no}")

                if args["mode"] == 'inference':
                    required_row = annotation_csv.loc[annotation_csv["input_image_location-input"] == q_no]
                    category = required_row["category"].values[0]
                    if pd.isna(category):
                        # Fallback category.
                        # This can be `none` as this is auto extracted from a superior LLM,
                        # But even superior LLMs are occasionally prone to inference errors/parsing errors
                        category = "mathematical_reasoning"
                    category_definition = CONFIG["inference"]["category_definition"][category]
                    answer_range = required_row["final_answer_range-output"].values[0]
                    answer_range = "A-D" if pd.isna(answer_range) else answer_range
                    PROMPT = PROMPT \
                        .replace("{{category}}", category) \
                        .replace("{{category_definition}}", category_definition) \
                        .replace("{{answer_range}}", answer_range)

                response = model.get_response(prompt=PROMPT, image_path=os.path.join(merged_screenshot_path, file))
                
                # print('RESPONSE: ', response)

                try:
                    model_response_dict[q_no] = ast.literal_eval(response
                                                                    .split("[/INST] ")[-1]
                                                                    .replace("```json", "")
                                                                    .replace("\n", "")
                                                                    .replace("```", "")
                                                                    )
                except Exception as e:
                    model_response_dict[q_no] = f"FAILED_{e}_<RESPONSE>{response}</RESPONSE>"
                    parse_failed_count += 1

                if not model.is_open_source:
                    time.sleep(30)

            if args["mode"] == 'inference':
                annotation_csv[f'{args["model_name"]}-response'] = annotation_csv['input_image_location-input'] \
                    .apply(lambda x: model_response_dict[x])

            if args["mode"] == "categorization":
                annotation_csv.loc[:, "category-input"] = annotation_csv['input_image_location-input'] \
                    .apply(lambda x: model_response_dict[x]["category"])
                annotation_csv.loc[:, "input_text_parsed-input"] = annotation_csv['input_image_location-input'] \
                    .apply(lambda x: model_response_dict[x]["question"])
                annotation_csv.loc[:, "section_instruction-input"] = annotation_csv['input_image_location-input'] \
                    .apply(lambda x: model_response_dict[x]["explanation"])

            annotation_csv.to_csv(os.path.join(paper_path, 'annotations_final.csv'), index=False)

            if parse_failed_count > 0:
                print(
                    f"\033[93m{MODE} STATUS of {paper_path} with model {args['model_name']}: "
                    f"OK with {parse_failed_count} failed parses."
                    '\u2713\033[0m')
            else:
                print(f"\033[92m{MODE} STATUS of {paper_path} with model {args['model_name']}: OK", '\u2713\033[0m')

            # INFERENCE_RUN_TRACKER[paper_path][args["model_name"]] = True
            # with open(inference_run_persistence_path, 'w') as file:
            #     json.dump(INFERENCE_RUN_TRACKER, file, indent=4)
                
        except Exception as e:
            print(f"\033[91m{MODE} STATUS of {paper_path} with model {args['model_name']}: Failed due to {e}",
                  '\u2717\033[0m')
