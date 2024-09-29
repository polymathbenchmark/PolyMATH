import base64
import json
import mimetypes
import os
import requests
import sys
from dotenv import load_dotenv
from tqdm import tqdm
from ast import literal_eval
import pandas as pd
import argparse
import logging
from PIL import Image
from typing import List
import google.generativeai as genai
import time
import re
import yaml
 
load_dotenv("../../.env")
API_KEY = os.environ.get("GEMINI_API_KEY")
do_gemini = True
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
ROOT_DIR = "../../"

post_processing_prompt = """
- You are an expert parsing assistant.
- Convert the output to a valid JSON parseable file.
- Ensure that the key of each json is 'description'.
- The corresponding value for 'description' also should be added.
- In case the corresponding values for 'description' is not found in the text, then
the value should be "NoValueFound".
- THE ONLY OUTPUT SHOULD BE THE JSON PARSEABLE STRING AND NOTHING ELSE!!
 
The JSON string is:
{json_string}
"""

class GeminiInference :
    def __init__(
            self,
            folders,
            prompting_strategy = "zeroshot",
            model = "gemini-1.5-flash",
            datastore_path = None,
            post_processing_json="",
            analysis_model = "",

    ) :
        self.model = model
        self.datastore_path = datastore_path
        self.post_processing_json = post_processing_json
        self.analysis_model = analysis_model
        with open("../prompts.json") as f :
            all_prompts = json.load(f)
        self.inference_prompts = all_prompts["inference"]
        self.image_description_prompts = all_prompts["image_description"]
        self.analysis_prompts = all_prompts["analysis"]
        self.category_definitions = self.inference_prompts["category_definition"]
        self.nshot_prompts = all_prompts["fewshot"]
        self.prompting_strategy = prompting_strategy
        prompt_dict = {
            "zeroshot" : "zeroshot",
            "chain_of_thought" : "cot",
            "step_back" : "step_back",
            "fewshot" : "fewshot",
            "answer_options_list_extraction" : "qa_extraction",
            "post_processing" : "post_processing",
            "text_based_inference" : "text_only",
        }
        self.prompting_init = prompt_dict[prompting_strategy]
 
        self.folders = [os.path.join(self.datastore_path,folder) for folder in folders]
        if "final_datasets" in self.datastore_path  :
            self.ss_folders = [os.path.join(self.datastore_path,'merged_screenshots')]
        else:
            self.ss_folders = [os.path.join(self.datastore_path,folder,'merged_screenshots') for folder in folders]
        self.response_csv_path_folder_map = dict()
        self.context_map = dict((k,self.get_image_list(v)) for k,v in zip(self.folders,self.ss_folders))
        self.exact_matches = 0
        self.total_count = 0
 
    def get_question_images(self) :
        all_qs = []
        for file in os.listdir(self.folder) :
            if file[0] == 'q' :
                all_qs.append(file)
        return all_qs
    
    def get_image_list(self,folder) :
        all_imagelists = []
        if "final_datasets" in self.datastore_path  :
            folder_df = pd.read_csv(os.path.join(self.folders[0],f'annotations_final.csv'))
            
            all_imagelists =[['',os.path.join(folder,id_ + "_" + q_no + ".png")] for id_,q_no in zip(folder_df["paper_id-input"],folder_df["input_image_location-input"])]
        else:
            for file in os.listdir(folder) :
                split_filename = file.split('_')
                q_image = file
                if (
                    len(split_filename) > 1 and # context present
                    file[0] == 'q' # its a question image
                    ): 
                    c_image = split_filename[-1]
                    all_imagelists.append([os.path.join(folder,c_image),os.path.join(folder,q_image)])
                elif (
                    len(split_filename) == 1 # no context present
                    and file[0] == 'q'  
                ):
                    all_imagelists.append(['',os.path.join(folder,q_image)])
        return all_imagelists
 
    def gen_template_nshot(self,category,image,ans_range):
        pre_example_prompt = self.nshot_prompts['pre_example_prompt'].format(answer_range=ans_range)
        image_path = Image.open(image)
        example_image = Image.open(os.path.join("../prompt_examples",self.nshot_prompts["examples"][category][0]["image"]))
        example_exp = self.nshot_prompts["examples"][category][0]["explanation"]
        post_example_prompt = self.nshot_prompts['post_example_prompt'].format(example_output = example_exp)
        
        template = [pre_example_prompt]+ [example_image]+ [post_example_prompt] + [image_path]

        # print(template)
        return template
    
    def gen_template(
        self,
        images,
        category,
        ans_range,
        json_str = "",
        extracted_question="",
        image_description="",
        extracted_answer_list="",
        ):
        cat_def = self.category_definitions[category]
        if self.prompting_strategy == "post_processing" : 
           prompt = post_processing_prompt.format(json_string = json_str)
        elif self.prompting_strategy  == "fewshot" :
            return self.gen_template_nshot(category,images[-1],ans_range)
        elif self.prompting_strategy == "answer_options_list_extraction" :
            prompt = self.analysis_prompts['answer_options_list_extraction']
        elif self.prompting_strategy  == "step_back" :
            self.pre_image_prompt = self.inference_prompts["common_prefix"] + self.inference_prompts[self.prompting_strategy]["meta_prompt"] + self.inference_prompts["common_postfix"]
            prompt = self.pre_image_prompt.format(category=category,
                                            category_definition = cat_def,
                                            answer_range = ans_range,
                                            step_back_category_prompt = self.inference_prompts[self.prompting_strategy]["step_back_category_prompt"][category])
        elif self.prompting_strategy  == "text_based_inference" :
            prompt = self.analysis_prompts[self.prompting_strategy].format(
                category = category,
                category_definition = cat_def,
                extracted_question = extracted_question,
                image_description = image_description,
                extracted_answer_list = extracted_answer_list,
                answer_range = ans_range
            )
        
        else :
            self.pre_image_prompt = self.inference_prompts["common_prefix"] + self.inference_prompts[self.prompting_strategy] + self.inference_prompts["common_postfix"]
            prompt = self.pre_image_prompt.format(category=category,
                                                category_definition = cat_def,
                                                answer_range = ans_range)
        img_list = []
        if self.prompting_strategy  != "text_based_inference" :
            for img_path in images:
                img_list.append(Image.open(img_path))
        
        template = [prompt] + img_list
        # print(template)
        return template
    
    def generate_response(self,
    imagelist,
    category,
    ans_range,
    q_no,
    json_str,
    extracted_question="",
    image_description="",
    extracted_answer_list=""):
        template = self.gen_template(imagelist,
                                    category,
                                    ans_range,
                                    json_str,
                                    extracted_question,
                                    image_description,
                                    extracted_answer_list) 
        response = model.generate_content(template, stream=True)
        response.resolve()
        try:
            response_text = response.candidates[0].content.parts[0].text
        except Exception as e:
            logging.debug(f"GeminiApiException: {e}")
            response_text = response.prompt_feedback
        return str(response_text)
    
    def get_matches(self,response_csv) :
        response_csv['final_answer-output'].fillna('',inplace=True)
        response_csv['final_answer-output'] = response_csv['final_answer-output'].astype(str)
        response_csv['final_answer-output_fmt'] = response_csv['final_answer-output'].apply(lambda val : [x.strip('[').strip(']') for x in val.split(',')])
        response_csv['claude_ans'] = response_csv.apply(lambda x : str(ord(x['claude_ans']) - ord('A') + 1).upper() if x['final_answer_range-output'] in ['1-4','1,2,3,4'] and not(x['claude_ans']>='1' and x['claude_ans']<='4' )else str(x['claude_ans']).upper(),axis=1)
        response_csv['exact_match'] = response_csv.apply(lambda row : 1 if row['claude_ans'] in row['final_answer-output_fmt'] else 0,axis=1)
        self.exact_matches += response_csv.loc[response_csv['exact_match']==1].shape[0]
        self.total_count += response_csv.shape[0]
 
    def get_accuracy(self) :
        return self.exact_matches/self.total_count
    
    def delete_response_csv(self,del_file):
        if os.path.exists(self.response_csv_path_folder_map[del_file]) :
            os.remove(self.response_csv_path_folder_map[del_file]) 
        return
    
    def encode_range(self,input_string):
        has_letter = bool(re.search(r'[a-zA-Z]', input_string))
        has_digit = bool(re.search(r'\d', input_string))
        
        if has_letter:
            return '[A-D]'
        elif has_digit:
            return '[1-4]'
        else:
            return None
 
    def generate_response_retry(self,imagelist,category,ans_range,q_no,json_str="",extracted_question="",image_description="",extracted_answer_list=""):
        if self.prompting_strategy != "post_processing" :
            retry = True
            while retry:
                try:
                    response = self.generate_response(imagelist,category,ans_range,q_no,json_str,extracted_question,image_description,extracted_answer_list)
                    retry = False
                except Exception as e:
                    print(f"\033[1;31mAn exception occurred: {e}. Retrying in 2 seconds...\033[0m")
                    time.sleep(2)
        else:
            is_parsed = False
            num_tries = 1
            try : response = literal_eval(json_str)
            except  :
                try  :  response = json.loads(json_str)
                except: 
                    while not is_parsed or num_tries>5:
                        try:
                            response = self.generate_response(imagelist,category,ans_range,q_no,json_str,extracted_question,image_description,extracted_answer_list)
                            response = json.loads(response)
                            is_parsed = True
                        except:
                            
                            # print("ORIGINAL OUTPUT: ",  f"{response}")
                            # print("="*20)
                            # print("LLM FAILED OUTPUT: ", response)
                            # print()
                            is_parsed=False
                            num_tries+=1
                            response = "None"

        return response
    
    def run_inference(self):
        for folder,imagelist in tqdm(self.context_map.items()) :
            print(f"Folder : {folder}")
            response_csv_path = os.path.join(folder,f'annotations_final_{self.prompting_init}_{self.model}.csv')
            self.response_csv_path_folder_map[folder.split('/')[-1]] = response_csv_path
 
            if os.path.exists(response_csv_path) :
                print('Inferencing already done')
                continue
                
            response_csv = pd.read_csv(os.path.join(folder,f'annotations_final.csv')) #  annotations_final
            

            extracted_question = ""
            image_description = ""
            extracted_answer_list = ""
            if "final_datasets" in self.datastore_path  :
                response_csv["image_mapping"] = response_csv["paper_id-input"] + "_" + response_csv["input_image_location-input"]
                q_list = response_csv['image_mapping'].values
                q_col_name = "image_mapping"
                if self.prompting_init == 'post_processing' : 
                    task = "_".join(self.post_processing_json.split("/")[-1].split(".")[0].split("_")[:-1])
                    response_json = os.path.join(folder,f'{self.prompting_init}_{task}.json')
                
                else:
                    response_json = os.path.join(folder,f'{self.prompting_init}_{self.model}.json')

            else :
                q_list = response_csv['input_image_location-input'].values
                q_col_name = "input_image_location-input"
            
            if self.prompting_init == 'post_processing' : 
                with open(self.post_processing_json, "r") as file:
                    post_proc_dict = json.load(file) 

            for idx in tqdm(range(len(imagelist))):
                image_list = imagelist[idx]
                if "final_datasets" in self.datastore_path  :
                    q_no = image_list[-1].split('/')[-1].split('.')[0]
                else:
                    q_no = image_list[-1].split('/')[-1].split('.')[0].split('_')[0]
                if ("final_datasets" in self.datastore_path  
                    and self.prompting_init == 'text_only') :
                    extracted_question =  response_csv.loc[response_csv[q_col_name]==q_no,'question_extracted'].values[0]
                    image_description = response_csv.loc[response_csv[q_col_name]==q_no,'image_description'].values[0]
                    extracted_answer_list =  response_csv.loc[response_csv[q_col_name]==q_no,'ans_list_extracted'].values[0]

                try:
                    with open(response_json, "r") as file:
                        all_responses = json.load(file) 
                        if q_no in all_responses.keys() :
                            continue
                except FileNotFoundError:
                    all_responses = dict()
                
                if q_no not in q_list :
                    continue

                category = response_csv.loc[response_csv[q_col_name]==q_no,'category'].values[0]
                ans_range = response_csv.loc[response_csv[q_col_name]==q_no,'final_answer_range-output'].values[0]
                if str(ans_range) == 'nan' :
                    ans_key = response_csv.loc[response_csv[q_col_name]==q_no,'final_answer-output'].values[0]
                    ans_range = self.encode_range(str(ans_key))
                    response_csv.loc[response_csv[q_col_name]==q_no,'final_answer_range-output'] = ans_range

                # response_csv.loc[response_csv[q_col_name]==q_no,f'response_{self.prompting_init}_{self.model}'] = str(gemini_response_dict)

                if self.prompting_init == 'step_back' : 
                    gemini_response_dict = self.generate_response_retry([image_list[-1]],category,ans_range,q_no)
                    values_extracted = self.extract_regex_stepback(str(gemini_response_dict))
                    dict_to_add = {
                        q_no : {
                            f"{self.model}_response" : str(gemini_response_dict),
                            f'answer_option_{self.prompting_init}_{self.model}' : values_extracted
                        }
                    }
                    # response_csv.loc[response_csv[q_col_name]==q_no,f'answer_option_{self.prompting_init}_{self.model}'] = self.extract_regex_stepback(str(gemini_response_dict))
                elif self.prompting_init == 'cot' : 
                    gemini_response_dict = self.generate_response_retry([image_list[-1]],category,ans_range,q_no)
                    values_extracted = self.extract_regex_cot(str(gemini_response_dict))
                    dict_to_add = {
                        q_no : {
                            f"{self.model}_response" : str(gemini_response_dict),
                            f'answer_option_{self.prompting_init}_{self.model}' : values_extracted
                        }
                    }
                    # response_csv.loc[response_csv[q_col_name]==q_no,f'answer_option_{self.prompting_init}_{self.model}'] = self.extract_regex_cot(str(gemini_response_dict))
                elif self.prompting_init  in ['zeroshot','fewshot'] :
                    gemini_response_dict = self.generate_response_retry([image_list[-1]],category,ans_range,q_no) 
                    values_extracted = self.response_post_processing(str(gemini_response_dict))
                    dict_to_add = {
                        q_no : {
                            f"{self.model}_response" : str(gemini_response_dict),
                            f'explaination_{self.prompting_init}_{self.model}' : values_extracted[0],
                            f'answer_{self.prompting_init}_{self.model}' : values_extracted[1],
                            f'answer_option_{self.prompting_init}_{self.model}' : values_extracted[2]
                        }
                    }
                    # response_csv.loc[response_csv[q_col_name]==q_no,f'answer_option_{self.prompting_init}_{self.model}'] = self.extract_regex_zeroshot(str(gemini_response_dict))
                elif self.prompting_init == 'qa_extraction' : 
                    gemini_response_dict = self.generate_response_retry([image_list[-1]],category,ans_range,q_no)
                    values_extracted = self.extract_regex_qa(str(gemini_response_dict))

                    dict_to_add = {
                        q_no : {
                            f"{self.model}_response" : str(gemini_response_dict),
                            "extracted_question" : values_extracted[0],
                            "extracted_ans_list" : values_extracted[1]
                        }
                    }
                elif self.prompting_init == 'post_processing' : 
                    item = post_proc_dict[q_no]
                    # print(item)
                    # break
                    json_str = item[f"{self.analysis_model}_response"]
                    
                    if json_str == '' :
                        response = 'None'
                    else:
                        # break
                        response = self.generate_response_retry([],category,ans_range,q_no,json_str)
                    item[f"post_processed_{task}"] = response
                    dict_to_add = {
                        q_no : item
                    }
                elif self.prompting_init == 'text_only' :
                    # print(q_no)
                    gemini_response_dict = self.generate_response_retry([image_list[-1]],category,ans_range,q_no,"",extracted_question,image_description,extracted_answer_list) 
                    values_extracted = self.response_post_processing(str(gemini_response_dict))
                    dict_to_add = {
                        q_no : {
                            f"{self.model}_response" : str(gemini_response_dict),
                            f'explaination_{self.prompting_init}_{self.model}' : values_extracted[0],
                            f'answer_{self.prompting_init}_{self.model}' : values_extracted[1],
                            f'answer_option_{self.prompting_init}_{self.model}' : values_extracted[2]
                        }
                    }
                merged_dict = {**all_responses, **dict_to_add}
                with open(response_json, 'w') as file:
                    json.dump(merged_dict, file, indent=4)

    def extract_regex_qa(self,json_string):
        values = []
        json_string = json_string.replace("\\n","")
        pattern_q1 = r"'question'\s*:\s*'(.*?)'"
        pattern_q1_backup = r'"question"\s*:\s*"(.*?)"'
        match_q1 = re.search(pattern_q1, json_string, re.DOTALL)
        if match_q1 == None : match_q1 = re.search(pattern_q1_backup, json_string, re.DOTALL)
        if match_q1: values.append(match_q1.group(1))
        else : values.append("None")
 
        pattern_q2 = r"answer_list':\s*'(.*?)'"
        pattern_q2_backup = r'answer_list":\s*"(.*?)"'
        match_q2 = re.search(pattern_q2, json_string, re.DOTALL)
        # if match_q2: values.append(match_q2.group(1))
        if match_q2 == None : match_q2 = re.search(pattern_q2_backup, json_string, re.DOTALL)
        if match_q2: values.append(match_q2.group(1))
        else : values.append("None")
        return values
    
    def extract_regex_cot(self,json_string):
        json_string = json_string.strip().replace("}","").replace("{","").replace("\"","'")
        grp_idx = 1
        # pattern_q3 = r'"Q6"\s*:\s*("?)([^"]+)(\1?)' #r"'Q6':\s*'(\d+)'"
        pattern = r"'Q6'\s*:\s*'[^']*\(([^)]+)\)" # without paranthesis
        pattern_backup = r"'Q6'\s*:\s*('?)([^']+)(\1?)" # with paranthesis  -everything - backup
        # match_q3 = re.search(pattern_q3, json_string, re.DOTALL)
        match = re.search(pattern, json_string, re.DOTALL)
        if match == None : # use pattern_backup
            grp_idx = 2
            match = re.search(pattern_backup, json_string, re.DOTALL)
        return match.group(grp_idx).upper() if match else "NONE"
    
    def extract_regex_stepback(self,json_string):
        json_string = json_string.strip().replace("}","").replace("{","").replace("\"","'")
        grp_idx = 1
        # pattern_q3 = r'"Q6"\s*:\s*("?)([^"]+)(\1?)' #r"'Q6':\s*'(\d+)'"
        pattern = r"'Q5'\s*:\s*'[^']*\(([^)]+)\)" # without paranthesis
        pattern_backup = r"'Q5'\s*:\s*('?)([^']+)(\1?)" # with paranthesis  -everything - backup
        # match_q3 = re.search(pattern_q3, json_string, re.DOTALL)
        match = re.search(pattern, json_string, re.DOTALL)
        if match == None : # use pattern_backup
            grp_idx = 2
            match = re.search(pattern_backup, json_string, re.DOTALL)
        return match.group(grp_idx).upper() if match else "NONE"

    def extract_regex_zeroshot(self,json_string):
        values = []
        pattern_q1 = r'"Q1"\s*:\s*"(.*?)"'
        match_q1 = re.search(pattern_q1, json_string, re.DOTALL)
        if match_q1: values.append(match_q1.group(1))
        else : values.append("None")
 
        pattern_q2 = r'"Q2"\s*:\s*("?)(\w+)(\1?)'
        match_q2 = re.search(pattern_q2, json_string, re.DOTALL)
        if match_q2: values.append(match_q2.group(2))
        else : values.append("None")
 
        pattern_q3 = r'"Q3"\s*:\s*("?)([^"]+)(\1?)' #r'"Q3"\s*:\s*("?)(\w+)(\1?)'
        match_q3 = re.search(pattern_q3, json_string, re.DOTALL)
        if match_q3: values.append(match_q3.group(2))
        else : values.append("None")
        return values
 
    def response_post_processing(self,json_string) :
        json_string = json_string.replace("json","").replace("```","").replace("\n","").replace("{{","{").replace("}}","}")
        values = ['None','None','None']
        try :
            dict = literal_eval(json_string)
        except :
            try :
                dict = json.loads(json_string)
            except :
                try : 
                    dict = yaml.safe_load(json_string)
                except :
                    dict = []
        # print(f"dict : {dict}")
        if len(dict) == 0:
            print("passed to regex for post processing\n")
            return self.extract_regex_zeroshot(json_string)
        # coming from json parsing through try-catch
        for idx,key in enumerate(dict.keys()) :
            values[idx] = dict[key]
        return values
    
    def get_exact_match(self,row) :
        gt,response = str(row["final_answer-output"]).upper(),str(row[f"{self.prompting_init}_final_answer_gpt4o"]).upper()
        ans_range = row["final_answer_range-output"]

        response = response.strip().replace("}","").strip()
        pattern =   r'\d+'
        match = re.search(pattern, response, re.DOTALL)
        if match: response  = match.group()

        # make sure both anskey and reponse are 1-4
        if (
            (ans_range in ["1-4","[1-4]"]) 
            and not response.isnumeric()
            ): 
            try : response = ord(response) - ord('A') + 1
            except : response = response
        if (
            (ans_range in ["A-D","[A-D]"]) 
            and response.isnumeric()
            ):
            try : response = chr(ord('@')+int(response))
            except : response = response
        row[f"{self.prompting_init}_final_answer_gpt4o_cleaned"] = response
        
        if gt == response : row[f"{self.prompting_init}_exact_match_gpt4o"] = 1
        else :  row[f"{self.prompting_init}_exact_match_gpt4o"] = 0
        return row
    
    
    
    
if __name__ == "__main__":
    datastore_path = os.path.join(ROOT_DIR,"datastore","final_datasets")
    folder_paths = os.listdir(datastore_path)
    folder_paths = [i for i in folder_paths if i not in ['metadata.json','.DS_Store','model_run.json','final_datasets']]
    folder_paths = ["testmini"]
    post_processing_json = "../../datastore/final_datasets/test/image_description_gpt-4o.json"
    inference_obj = GeminiInference(
                                    folder_paths,
                                    prompting_strategy="fewshot",
                                    datastore_path = datastore_path,
                                    post_processing_json = "",
                                    analysis_model = "",
                                    )
    print('Starting Inference..')
    inference_obj.run_inference()
