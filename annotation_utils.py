import os
import re
import cv2
import uuid
import json
import argparse
import glob
import shutil
import jsonlines
import numpy as np
import pandas as pd
from collections import Counter

"""
This script contains the set of utility functions
useful for the annotator
"""

parser = argparse.ArgumentParser(description='Prepare uniform directory structure')
parser.add_argument('-m', '--mode', help='The annotations utility mode', required=False)
parser.add_argument('-pp', '--paper_path', help='The name of paper', required=False)
parser.add_argument('-dp', '--datastore_path', help='The path of datastore', required=False)
parser.add_argument('-an', '--annotator_name', help='The name of annotator', required=False)
parser.add_argument('-u', '--update', help='Update an entry', required=False, default=False)
parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                    help='overwrite existing annotations.csv?', required=False, default=False)
args = vars(parser.parse_args())


class SchemaMismatchError(Exception):
    pass


def merge_images(image_list):
    """
    Given a list of image array, merge them into a single image with a padding
    :param image_list: Numpy array of images in a list
    :return: The merged image array
    """
    # Find the maximum width among all images
    max_width = max(image.shape[1] for image in image_list)

    # Pad images to have the same width
    padded_images_list = []
    for image in image_list:
        height, width, _ = image.shape
        pad_width = max_width - width
        padded_image = np.pad(image, ((0, 0), (0, pad_width), (0, 0)), constant_values=255)
        padded_images_list.append(padded_image)
    return np.concatenate(padded_images_list, axis=0)


def merge_split_screenshots(paper_path):
    """
    This method merges screenshots that are in multiple parts using OpenCV.
    Specifically files of the following format:
        q<number>_<blob [0...n]>_c<number>.png --> q<number>_c<number>.png
        c<number>_<blob [0...n]>.png --> c<number>.png
    :param paper_path: Folder path of the paper
    :return: Void
    """
    try:
        screenshots_folder_path = os.path.join(paper_path, "screenshots")

        if not os.path.exists(screenshots_folder_path):
            return None

        # Determine context images that have splits
        contexts_list = [i for i in os.listdir(screenshots_folder_path) if i.startswith('c')]
        only_context_list = [i.split('_')[0] + "_" for i in os.listdir(screenshots_folder_path) if i.startswith('c')]
        contexts_with_splits_list = [context for context, count in Counter(only_context_list).items() if count > 1]
        context_images_to_be_merged = []
        for context_with_split in contexts_with_splits_list:
            contexts_to_be_merged = []
            for context_image in contexts_list:
                if context_with_split in context_image:
                    contexts_to_be_merged.append(os.path.join(screenshots_folder_path, context_image))
            context_images_to_be_merged.append(sorted(contexts_to_be_merged))

        # Determine question images that have splits
        questions_list = [i for i in os.listdir(screenshots_folder_path) if i.startswith('q')]
        questions_without_context_list = [i.split('_')[0] + '_' for i in os.listdir(screenshots_folder_path) if
                                          i.startswith('q')]
        questions_with_splits_list = [question for question, count in Counter(questions_without_context_list).items() if
                                      count > 1]
        question_images_to_be_merged = []
        for question_with_split in questions_with_splits_list:
            questions_to_be_merged = []
            for question_image in questions_list:
                if question_with_split in question_image:
                    questions_to_be_merged.append(os.path.join(screenshots_folder_path, question_image))
            question_images_to_be_merged.append(sorted(questions_to_be_merged))

        images_to_be_merged = context_images_to_be_merged + question_images_to_be_merged

        for image_group_to_be_merged in images_to_be_merged:
            image_list = []
            merged_image_name = os.path.basename(re.sub(r'_\d+', '', image_group_to_be_merged[0]))
            for child_image_path in image_group_to_be_merged:
                img = cv2.imread(child_image_path)
                image_list.append(img)

                # Track images with split with markers
                os.rename(child_image_path,
                          os.path.join(
                              os.path.dirname(child_image_path),
                              "CORRECTED_" + os.path.basename(child_image_path))
                          )

            # Write out merged file
            merged_image = merge_images(image_list)
            cv2.imwrite(os.path.join(screenshots_folder_path, merged_image_name), merged_image)

        # Final cleanup of naming conventions to maintain consistency:
        # ==> Files of format: q<number>_0.png --> q<number>.png
        for img_name in os.listdir(screenshots_folder_path):
            if not img_name.startswith("CORRECTED"):
                if "_" in img_name:
                    os.rename(
                        os.path.join(screenshots_folder_path, img_name),
                        os.path.join(screenshots_folder_path,
                                     re.sub(r'_\d+', '', img_name)
                                     )
                    )

        # Delete merged screenshots folder if exists
        merged_screenshots_directory = os.path.join(paper_path, "merged_screenshots")
        if os.path.exists(merged_screenshots_directory):
            shutil.rmtree(merged_screenshots_directory, ignore_errors=True)

        print(f"\033[92mMERGE STATUS of {paper_path}: OK", '\u2713\033[0m')
        return True
    except:
        print(f"\033[91mMERGE STATUS of {paper_path}: Failed", '\u2717\033[0m')
        return False


def create_annotations_helper(paper_path, is_override=False):
    """
    This method is the helper method to create annotations.csv
    :param paper_path: Path to the paper folder
    :param is_override: This argument is a boolean flag to control if autogenerated fields should be overwritten.
    :return: Returns the basic empty annotation dataframe with pre-filled columns
    """
    try:
        questions_directory = 'screenshots'
        annotations_file = os.path.join(paper_path, "annotations.csv")
        autogenerated_fields = {
            "sample_id-input": [],
            "input_image_location-input": [],
            "context-input": []
        }

        # Determine file metadata
        location, paper_name_with_extension = os.path.split(paper_path)
        paper_name_without_extension = paper_name_with_extension.split(".")[0]
        paper_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, paper_name_without_extension))

        # Extract all screenshots into respective lists
        files = [os.path.basename(file_name) for file_name in
                 glob.glob(os.path.join(paper_path, questions_directory, '*.png'))]
        question_regex = r'(q\d+)(?:_c\d+)?\.png'
        q_pattern = re.compile(question_regex)
        question_number_regex = r'q(\d+)'
        only_qno_pattern = re.compile(question_number_regex)
        questions = [file for file in files if file.startswith("q")]

        # Fields for automatic annotation
        autogenerated_fields["sample_id-input"] = [int(only_qno_pattern.match(question).group(1)) for
                                                   question in questions]
        paper_ids = [paper_id] * len(questions)
        input_text_parsed = []

        # Fields for manual annotation
        instruction = []
        explanation = []
        final_answer = []
        final_answer_range = []
        page_number = []
        category = []

        for idx, question in enumerate(questions):

            # if context exists for the question name
            if 'c' in question:
                context_file = question.split('_')[-1]

                # if context files not found
                if not os.path.exists(os.path.join(paper_path, questions_directory, context_file)):
                    print(f'[ANNOTATION ERROR] context not found for {question}!')
                    autogenerated_fields["context-input"].append("NOT_FOUND_ERROR")
                else:
                    autogenerated_fields["context-input"].append(context_file)
            else:
                autogenerated_fields["context-input"].append("NO_CONTEXT")

            autogenerated_fields["input_image_location-input"].append(q_pattern.match(question).group(1))

            input_text_parsed.append('')
            instruction.append('')
            final_answer.append('')
            final_answer_range.append('')
            page_number.append('')
            explanation.append('')
            category.append('')

        if os.path.exists(annotations_file):
            if is_override:
                existing_annotation_file = pd.read_csv(annotations_file)
                if (existing_annotation_file.shape[0] > 0) and ("q" not in str(existing_annotation_file["sample_id"
                                                                                                        "-input"][0])):
                    print(f"\033[92mCREATE ANNOTATION STATUS of {paper_path}: OK - "
                          f"Already in optimal state", '\u2713\033[0m')
                    return True
                try:
                    # If override is allowed, ONLY the auto generated fields will be updated
                    newly_generated_annotation_file = pd.DataFrame(autogenerated_fields)
                    annotation_file_internal = pd.merge(
                        existing_annotation_file,
                        newly_generated_annotation_file,
                        left_on="sample_id-input",
                        right_on="input_image_location-input"
                    )
                    annotation_file_internal.drop([
                        "input_image_location-input_x",
                        "context-input_x",
                        "sample_id-input_x",
                    ], axis=1, inplace=True)

                    annotation_file_internal.rename(
                        columns={"input_image_location-input_y": "input_image_location-input",
                                 "sample_id-input_y": "sample_id-input",
                                 "context-input_y": "context-input"},
                        inplace=True)
                except Exception:
                    raise SchemaMismatchError("The newly auto generated fields does not match with the schema of the "
                                              "existing annotation file.")
            else:
                # Do nothing if annotation.csv exist and the override flag is false
                annotation_file_internal = None
        else:
            # Create schema for annotation table
            annotation_file_internal = pd.DataFrame(
                data={
                    "paper_id-input": paper_ids,
                    "sample_id-input": autogenerated_fields["sample_id-input"],
                    "page_number-input": page_number,
                    "input_image_location-input": autogenerated_fields["input_image_location-input"],
                    "section_instruction-input": instruction,
                    "context-input": autogenerated_fields["context-input"],
                    "input_text_parsed-input": input_text_parsed,
                    "explanation-output": explanation,
                    "final_answer-output": final_answer,
                    "final_answer_range-output": final_answer_range,
                    "category-input": category
                }
            )

        # Renaming image files of the format q<number>_c<number>.png to q<number>.png after
        # capturing the context fields
        if is_override:
            for question in questions:
                os.rename(
                    os.path.join(paper_path, questions_directory, question),
                    os.path.join(paper_path, questions_directory, f"{q_pattern.match(question).group(1)}.png")
                )

        # sorting by question number
        if annotation_file_internal is not None:
            annotation_file_internal['sort_col'] = annotation_file_internal['sample_id-input'].apply(
                lambda x: int(x))
            annotation_file_internal = annotation_file_internal.sort_values(by=['sort_col']).drop(columns=['sort_col'])

            if annotation_file_internal.shape[0] == 0:
                print(f"\033[93mCREATE ANNOTATION STATUS of {paper_path}: SKIPPED since screenshots not found. "
                      '\u2713\033[0m')
                return None
            else:
                annotation_file_internal.to_csv(os.path.join(paper_path, 'annotations.csv'), index=False)
                print(f"\033[92mCREATE ANNOTATION STATUS of {paper_path}: OK", '\u2713\033[0m')
                return True
        else:
            print(f"\033[93mCREATE ANNOTATION STATUS of {paper_path}: SKIPPED since annotations exists. "
                  f'Re-run command with "--overwrite" flag to overwrite.', '\u2713\033[0m')
            return None
    except (SchemaMismatchError, ValueError, AttributeError) as e:
        print(f"\033[91mCREATE ANNOTATION STATUS of {paper_path}: Failed with '{e}'", '\u2717\033[0m')
        return False


def get_dict_array(annotation_dataframe):
    """
    This method returns a dictionary array from the annotation dataframe
    :param annotation_dataframe: The annotation.csv loaded as a dataframe
    :return: Returns the array of output dictionaries
    """
    ann_df = annotation_dataframe.astype(str)
    output_dict_array = []

    for idx, row in ann_df.iterrows():
        output_dict = {'input': {}, 'output': {}}
        for column in list(ann_df.columns):
            field_name, input_or_output_identifier = column.split('-')
            output_dict[input_or_output_identifier][field_name] = row[column]
        output_dict_array.append(output_dict)

    return output_dict_array


def merge_answer_keys(anskey_path, annotations_path):
    paper_path = os.path.dirname(annotations_path)
    try:
        annotations_df = pd.read_csv(annotations_path)
        # print(annotations_df["final_answer-output"].isna().sum() >= (0.9 * len(annotations_df)))
        if annotations_df["final_answer-output"].isna().sum() <= (0.1 * len(annotations_df)):
            print(f"\033[93mMERGE ANSWER KEY STATUS of {paper_path}: SKIPPED since merging already done. ")
            return None
        else:
            anskey_df = pd.read_csv(anskey_path)
            col_names = anskey_df.columns
            anskey_df = anskey_df.groupby(col_names[0])[col_names[1]].agg(list).reset_index()
            anskey_df[col_names[1]] = anskey_df[col_names[1]].apply(lambda x: x[0] if len(x) == 1 else x)

            annotations_df = pd.merge(annotations_df, anskey_df,
                                      left_on="sample_id-input",
                                      right_on=col_names[0]
                                      )
            annotations_df["final_answer-output"] = annotations_df[col_names[1]]
            annotations_df.drop(columns=[col_names[0], col_names[1]], inplace=True)
            annotations_df.to_csv(annotations_path)
            print(f"\033[92mMERGE ANSWER KEY STATUS of {paper_path}: OK", '\u2713\033[0m')
            return True
    except Exception as e:
        print(f"\033[91mMERGE ANSWER KEY STATUS of {paper_path}: Failed with '{e}'", '\u2717\033[0m')
        return False


if __name__ == "__main__":

    # Enter a root path
    root_path = "scripts/"

    if args["mode"] == "create_metadata":
        """
        Create required files for metadata saving and prepares annotation related directories
        """
        if args["paper_path"] is None:
            raise ValueError("Required --paper_path value not specified!")

        # Determine file metadata
        location, paper_name_with_extension = os.path.split(args["paper_path"])
        paper_name_without_extension = paper_name_with_extension.split(".")[0]

        if args["annotator_name"] is None:
            annotator = input("Enter name of annotator: ")
        else:
            annotator = args["annotator_name"]

        if args["datastore_path"] is None:
            raise ValueError("Required --datastore_path value not specified!")

        if not os.path.exists(os.path.join(args["datastore_path"], "metadata.json")):
            os.makedirs(args["datastore_path"], exist_ok=True)
            file_path = os.path.join(args["datastore_path"], "metadata.json")

            # Create new entry
            paper_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, paper_name_without_extension))
            metadata_entry = {
                'file_name': paper_name_with_extension,
                'num_questions': None,
                'annotator': annotator
            }
            with open(file_path, "w") as file:
                json.dump({paper_id: metadata_entry}, file, indent=4)
        else:
            # Open existing json file
            with open(os.path.join(args["datastore_path"], "metadata.json"), "r") as file:
                existing_metadata_entries = json.load(file)

            # Create new entry
            paper_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, paper_name_without_extension))
            annotator = args["annotator_name"]
            metadata_entry = {
                'file_name': paper_name_with_extension,
                'num_questions': None,
                'annotator': annotator
            }
            existing_metadata_entries[paper_id] = metadata_entry
            with open(os.path.join(args["datastore_path"], "metadata.json"), 'w') as f:
                json.dump(existing_metadata_entries, f, indent=4)

        # Create the subdirectory of the paper being handled
        os.makedirs(os.path.join(args["datastore_path"], paper_name_without_extension), exist_ok=True)

        # Create the screenshots directory for each paper
        os.makedirs(os.path.join(args["datastore_path"], paper_name_without_extension, "screenshots"),
                    exist_ok=True)

        # Copy the paper from the raw_dataset to the subdirectory created
        shutil.copy(args["paper_path"], os.path.join(args["datastore_path"], str(paper_name_without_extension)))
        print(f'Created directory for {os.path.join(root_path, "datastore", paper_name_without_extension)}')

    if args["mode"] == "merge_screenshots":
        """
        Merge screenshots of contexts and questions
        """
        if args["datastore_path"] is not None:
            filtered_directories = [i for i in os.listdir(args["datastore_path"]) if not i.startswith(".")]
            filtered_directories.remove("metadata.json")
            total_files_count = len(filtered_directories)
            success_count, failed_count, skipped_count = 0, 0, 0
            for paper_folders in filtered_directories:
                paper_path = os.path.join(args["datastore_path"], paper_folders)
                is_success = merge_split_screenshots(paper_path)
                if is_success is None:
                    skipped_count += 1
                    print(f"\033[93m MERGE STATUS of {paper_path}: SKIPPED since screenshots not found. \u2713\033[0m")
                elif is_success:
                    success_count += 1
                else:
                    failed_count += 1
            print(f"STATS: Success={success_count}/{total_files_count} | Fail={failed_count}/{total_files_count}"
                  f" | Skipped={skipped_count}/{total_files_count}")
        else:
            if args["paper_path"] is None:
                raise ValueError("One of --paper_path or --datastore_path value is required!")
            is_success = merge_split_screenshots(args["paper_path"])

    if args["mode"] == "create_annotation":
        """
        Create annotations.csv after you take screenshots
        """

        if args["datastore_path"] is not None:
            filtered_directories = [i for i in os.listdir(args["datastore_path"]) if not i.startswith(".")]
            filtered_directories.remove("metadata.json")
            total_files_count = len(filtered_directories)
            success_count, failed_count, skipped_count = 0, 0, 0
            for paper_folder in filtered_directories:
                paper_path = os.path.join(args["datastore_path"], paper_folder)

                # Allow bulk annotation generation for files that was not run previously.
                # Potentially destructive operation.
                # Advise caution before making changes to this line
                is_success = create_annotations_helper(paper_path, is_override=args["overwrite"])
                if is_success is None:
                    skipped_count += 1
                elif is_success:
                    success_count += 1
                else:
                    failed_count += 1
            print(f"STATS: Success={success_count}/{total_files_count} | Fail={failed_count}/{total_files_count}"
                  f" | Skipped={skipped_count}/{total_files_count}")
        else:
            if args["paper_path"] is None:
                raise ValueError("One of --paper_path or --datastore_path value is required!")

            create_annotations_helper(args["paper_path"], is_override=args["overwrite"])

    if args["mode"] == "merge_answer_keys":
        """
        Merges the answer keys file to annotations.csv file
        """
        if args["datastore_path"] is not None:
            filtered_directories = [i for i in os.listdir(args["datastore_path"]) if not i.startswith(".")]
            filtered_directories.remove("metadata.json")
            total_files_count = len(filtered_directories)
            success_count, failed_count, skipped_count = 0, 0, 0
            for paper_folder in filtered_directories:
                paper_path = os.path.join(args["datastore_path"], paper_folder)
                annotations_path = os.path.join(paper_path, "annotations.csv")
                anskey_probable_path = os.path.join(paper_path, "ans*.csv")
                anskey_matching_files = glob.glob(anskey_probable_path)
                if len(anskey_matching_files) > 0:
                    anskey_path = anskey_matching_files[0]
                    is_success = merge_answer_keys(anskey_path=anskey_path,
                                                   annotations_path=annotations_path)
                else:
                    # Since answer key.csv does not exist
                    print(f"{paper_path} does not have answer key yet")
                    is_success = None
                if is_success is None:
                    skipped_count += 1
                elif is_success:
                    success_count += 1
                else:
                    failed_count += 1
            print(f"STATS: Success={success_count}/{total_files_count} | Fail={failed_count}/{total_files_count}"
                  f" | Skipped={skipped_count}/{total_files_count}")
        else:
            if args["paper_path"] is None:
                raise ValueError("One of --paper_path or --datastore_path value is required!")

            annotations_path = os.path.join(args["paper_path"], "annotations.csv")
            anskey_probable_path = os.path.join(args["paper_path"], "ans*.csv")
            anskey_matching_files = glob.glob(anskey_probable_path)
            anskey_path = anskey_matching_files[0]
            is_success = merge_answer_keys(anskey_path=anskey_path, annotations_path=annotations_path)

    if args["mode"] == "freeze_annotation":
        """
        Create/appends to root/annotations.jsonl after human annotations.
        """

        if args["datastore_path"] is None:
            raise ValueError("Required --datastore_path value not specified!")
        else:
            filtered_directories = [i for i in os.listdir(args["datastore_path"]) if not i.startswith(".")]
            for paper_folders in os.listdir(args["datastore_path"]):
                if paper_folders not in ["metadata.json", ".DS_Store"]:
                    paper_path = os.path.join(args["datastore_path"], paper_folders)
                    annotation_dataframe = pd.read_csv(os.path.join(paper_path, 'annotations.csv'))
                    all_annotations = get_dict_array(annotation_dataframe)
                    with jsonlines.open(os.path.join(args["datastore_path"], 'annotation.jsonl'), mode='a') as writer:
                        for annotation in all_annotations:
                            writer.write(annotation)
                    writer.close()
