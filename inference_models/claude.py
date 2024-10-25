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
import re
import logging
import os
import json
import time
import boto3
import botocore
from typing import Type
# from botocore.client import BaseClient
# from botocore.config import Config

# config = Config(
#     retries = dict(
#         max_attempts = 10
#     )
# )


bedrock_client = boto3.client('bedrock-runtime',region_name='us-east-1') #, config=config
import pandas as pd
from tqdm.notebook import tqdm_notebook
import base64
import yaml

load_dotenv("../../.env")

ROOT_DIR = "../"
PROMPT = '''
You are given a mathematical/logical reasoning question. 
You are provided with two images, the first one is the context information for the second image.
The first image, if provided, is the context and the second image has the question along with 4 answer choices.

NOTE : If you are provided with just one image, there is no extra context needed.
NOTE : The answer is surely one amongst the four choices[A-D or 1-4] provided, it cannot be NONE.

Using on all this information provided, output a dictionary with the following fields - 
{
"question": a transcription of the question being asked based on the Question images,
"choices": a list containing each answer choice transcribed from the Question images,
"answer": the option out of the given choices that correctly answers the question. Just output the answer choice number [A-D or 1-4] and NOT the actual answer associated with that choice number,
"explanation": 100-word explanation of the solution to the question that results in the correct answer.
}
Output only the JSON and nothing else. Answer now.
'''

class ClaudeInference :
    def __init__(
            self,
            folders,
            prompting_strategy = "zeroshot",
            # pre_image_prompt = PROMPT
    ) :
        # self.model = model
        # self.pre_image_prompt = pre_image_prompt
        with open("../prompts.json") as f :
            self.inference_prompts = json.load(f)["inference"]
        self.category_definitions = self.inference_prompts["category_definition"]
        self.prompting_strategy = prompting_strategy

        self.folders = [os.path.join(ROOT_DIR,'datastore',folder) for folder in folders]
        self.ss_folders = [os.path.join(ROOT_DIR,'datastore',folder,'merged_screenshots') for folder in folders]
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
    
    def generate_message(self,bedrock_runtime, model_id, messages, max_tokens=512,top_p=1,temp=0.5,system=''):
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": temp,
                "top_p": top_p,
                "system": system
            }  
        )  
        
        response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
        response_body = json.loads(response.get('body').read())

        return response_body

    def query_claude(self,payload):
        """Sends a request to the OpenAI API and prints the response."""
        response=self.generate_message(bedrock_client,
                                   model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0",
                                   messages=payload,
                                   max_tokens=1024,temp=0.5,top_p=0.9)
        return response
    
    def local_image_to_data_url(self,image_path):
        with open(image_path, "rb") as image_path:
            binary_data_1 = image_path.read()
            base_64_encoded_data = base64.b64encode(binary_data_1)
            base64_string = base_64_encoded_data.decode('utf-8')

        return base64_string

    def gen_template(self,
                    images,
                    category,
                    ans_range,
                    prompting_strategy = "zeroshot",
                    ):
        cat_def = self.category_definitions[category]
        if self.prompting_strategy  == "step_back" :
            self.pre_image_prompt = self.inference_prompts["common_prefix"] + self.inference_prompts[prompting_strategy["meta_prompt"]]
            prompt = self.pre_image_prompt.format(category=category,
                                            category_definition = cat_def,
                                            answer_range = ans_range,
                                            step_back_category_prompt = self.inference_prompts[prompting_strategy["step_back_category_prompt"][category]])
        else :
            self.pre_image_prompt = self.inference_prompts["common_prefix"] + self.inference_prompts[prompting_strategy]
            prompt = self.pre_image_prompt.format(category=category,
                                                category_definition = cat_def,
                                                answer_range = ans_range)
        template = [
    #             {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                        "text": prompt},
                    ],
                },
        ]
        for idx,image in enumerate(images):
            if image == '': # no context present
                continue
            # print(f'------{image}----')
            base64_image = self.local_image_to_data_url(image)
            template[0]['content'].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data":base64_image

                }
            })

        return template
    
    def generate_response(self,imagelist,category,ans_range) :
        payload = self.gen_template(imagelist,category,ans_range) 
        response = self.query_claude(payload)
        ans = response['content'][0]['text']
        return ans
    
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

    def run_inference(self):
        # print( self.context_map)
        for folder,imagelist in tqdm(self.context_map.items()) :
            print(f"Folder : {folder}")
            response_csv_path = os.path.join(folder,'claude_3_5_sonnet_response.csv')
            self.response_csv_path_folder_map[folder.split('/')[-1]] = response_csv_path
            if os.path.exists(response_csv_path) : # inferencing already done - simply read it & update counts to get metrics
                print('Inferencing already done')
                # response_csv = pd.read_csv(response_csv_path) # read
                # self.get_matches(response_csv) # update counts for metrics
                # print(self.exact_matches,self.total_count)
                continue
                
            response_csv = pd.read_csv(os.path.join(folder,'annotations_w_claude_3_sonnet_response.csv'))


            for idx in tqdm(range(len(imagelist))):
                image_list = imagelist[idx]
                q_no = image_list[-1].split('/')[-1].split('.')[0].split('_')[0]
                q_list = response_csv['input_image_location-input'].values
                if q_no not in q_list :
                    continue
                category = response_csv.loc[response_csv['input_image_location-input']==q_no,'category'].values[0]
                ans_range = response_csv.loc[response_csv['input_image_location-input']==q_no,'final_answer_range-output'].values[0]
                if str(ans_range) == 'nan' :
                    ans_key = response_csv.loc[response_csv['input_image_location-input']==q_no,'final_answer-output'].values[0]
                    ans_range = self.encode_range(str(ans_key))
                    response_csv.loc[response_csv['input_image_location-input']==q_no,'final_answer_range-output'] = ans_range
                claude_response_dict = self.generate_response(image_list,category,ans_range)
                response_csv.loc[response_csv['input_image_location-input']==q_no,'claude_3_5_sonnet_response'] = str(claude_response_dict)
                # print("Wating before next call...")
                time.sleep(60)
            response_csv.to_csv(response_csv_path,index=False)
            
            # self.get_matches(response_csv)
        
        # acc = self.get_accuracy()
        # print(f'True Positives : {self.exact_matches} | Total Count : {self.total_count} | Accuracy : {acc}')
    
    def extract_regex(json_string):
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

    def response_post_processing(json_string) :
        json_string = json_string.replace("\n","")
        values = []
        try :
            dict = literal_eval(json_string)
        except :
            try :
                dict = json.loads(json_string)
            except :
                try : 
                    dict = yaml.safe_load(json_string)
                except :
                    dict = {}
        if len(dict) == 0:
            values = extract_regex(json_string)
        else: # coming from json parsing through try-catch
            for key in dict.keys() :
                values.append(dict[key])
        return pd.Series(values)
    
if __name__ == "__main__":
    datastore_path = os.path.join(ROOT_DIR,"datastore")
    folder_paths = os.listdir(datastore_path)
    folder_paths = [i for i in folder_paths if i not in ['metadata.json','.DS_Store']]
    # folder_paths = ['QP_Andhra Pradesh NTSE Stage 1 2017-18 (MAT_SAT_LANG)']
    # folder_paths = [i for i in folder_paths if 'annotations.csv' in os.listdir(os.path.join(datastore_path,i))]
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-f', '--folder_list', help='The list of folder paths to run inference on', required=True)
    # args = vars(parser.parse_args())
    # folder_paths = args['folder_list'].split(',')
    
    inference_obj = ClaudeInference(folder_paths)
    print('Starting Inference..')
    inference_obj.run_inference()

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
import re
import logging
import os
import json
import time
import boto3
import botocore
from typing import Type
# from botocore.client import BaseClient
# from botocore.config import Config

# config = Config(
#     retries = dict(
#         max_attempts = 10
#     )
# )


bedrock_client = boto3.client('bedrock-runtime',region_name='us-east-1') #, config=config
import pandas as pd
from tqdm.notebook import tqdm_notebook
import base64
import yaml

load_dotenv("../../.env")

ROOT_DIR = "../"
PROMPT = '''
You are given a mathematical/logical reasoning question. 
You are provided with two images, the first one is the context information for the second image.
The first image, if provided, is the context and the second image has the question along with 4 answer choices.

NOTE : If you are provided with just one image, there is no extra context needed.
NOTE : The answer is surely one amongst the four choices[A-D or 1-4] provided, it cannot be NONE.

Using on all this information provided, output a dictionary with the following fields - 
{
"question": a transcription of the question being asked based on the Question images,
"choices": a list containing each answer choice transcribed from the Question images,
"answer": the option out of the given choices that correctly answers the question. Just output the answer choice number [A-D or 1-4] and NOT the actual answer associated with that choice number,
"explanation": 100-word explanation of the solution to the question that results in the correct answer.
}
Output only the JSON and nothing else. Answer now.
'''

class ClaudeInference :
    def __init__(
            self,
            folders,
            prompting_strategy = "zeroshot",
            # pre_image_prompt = PROMPT
    ) :
        # self.model = model
        # self.pre_image_prompt = pre_image_prompt
        with open("../prompts.json") as f :
            self.inference_prompts = json.load(f)["inference"]
        self.category_definitions = self.inference_prompts["category_definition"]
        self.prompting_strategy = prompting_strategy

        self.folders = [os.path.join(ROOT_DIR,'datastore',folder) for folder in folders]
        self.ss_folders = [os.path.join(ROOT_DIR,'datastore',folder,'merged_screenshots') for folder in folders]
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
    
    def generate_message(self,bedrock_runtime, model_id, messages, max_tokens=512,top_p=1,temp=0.5,system=''):
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": temp,
                "top_p": top_p,
                "system": system
            }  
        )  
        
        response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
        response_body = json.loads(response.get('body').read())

        return response_body

    def query_claude(self,payload):
        """Sends a request to the OpenAI API and prints the response."""
        response=self.generate_message(bedrock_client,
                                   model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0",
                                   messages=payload,
                                   max_tokens=1024,temp=0.5,top_p=0.9)
        return response
    
    def local_image_to_data_url(self,image_path):
        with open(image_path, "rb") as image_path:
            binary_data_1 = image_path.read()
            base_64_encoded_data = base64.b64encode(binary_data_1)
            base64_string = base_64_encoded_data.decode('utf-8')

        return base64_string

    def gen_template(self,
                    images,
                    category,
                    ans_range,
                    prompting_strategy = "zeroshot",
                    ):
        cat_def = self.category_definitions[category]
        if self.prompting_strategy  == "step_back" :
            self.pre_image_prompt = self.inference_prompts["common_prefix"] + self.inference_prompts[prompting_strategy["meta_prompt"]]
            prompt = self.pre_image_prompt.format(category=category,
                                            category_definition = cat_def,
                                            answer_range = ans_range,
                                            step_back_category_prompt = self.inference_prompts[prompting_strategy["step_back_category_prompt"][category]])
        else :
            self.pre_image_prompt = self.inference_prompts["common_prefix"] + self.inference_prompts[prompting_strategy]
            prompt = self.pre_image_prompt.format(category=category,
                                                category_definition = cat_def,
                                                answer_range = ans_range)
        template = [
    #             {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                        "text": prompt},
                    ],
                },
        ]
        for idx,image in enumerate(images):
            if image == '': # no context present
                continue
            # print(f'------{image}----')
            base64_image = self.local_image_to_data_url(image)
            template[0]['content'].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data":base64_image

                }
            })

        return template
    
    def generate_response(self,imagelist,category,ans_range) :
        payload = self.gen_template(imagelist,category,ans_range) 
        response = self.query_claude(payload)
        ans = response['content'][0]['text']
        return ans
    
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

    def run_inference(self):
        # print( self.context_map)
        for folder,imagelist in tqdm(self.context_map.items()) :
            print(f"Folder : {folder}")
            response_csv_path = os.path.join(folder,'claude_3_5_sonnet_response.csv')
            self.response_csv_path_folder_map[folder.split('/')[-1]] = response_csv_path
            if os.path.exists(response_csv_path) : # inferencing already done - simply read it & update counts to get metrics
                print('Inferencing already done')
                # response_csv = pd.read_csv(response_csv_path) # read
                # self.get_matches(response_csv) # update counts for metrics
                # print(self.exact_matches,self.total_count)
                continue
                
            response_csv = pd.read_csv(os.path.join(folder,'annotations_w_claude_3_sonnet_response.csv'))


            for idx in tqdm(range(len(imagelist))):
                image_list = imagelist[idx]
                q_no = image_list[-1].split('/')[-1].split('.')[0].split('_')[0]
                q_list = response_csv['input_image_location-input'].values
                if q_no not in q_list :
                    continue
                category = response_csv.loc[response_csv['input_image_location-input']==q_no,'category'].values[0]
                ans_range = response_csv.loc[response_csv['input_image_location-input']==q_no,'final_answer_range-output'].values[0]
                if str(ans_range) == 'nan' :
                    ans_key = response_csv.loc[response_csv['input_image_location-input']==q_no,'final_answer-output'].values[0]
                    ans_range = self.encode_range(str(ans_key))
                    response_csv.loc[response_csv['input_image_location-input']==q_no,'final_answer_range-output'] = ans_range
                claude_response_dict = self.generate_response(image_list,category,ans_range)
                response_csv.loc[response_csv['input_image_location-input']==q_no,'claude_3_5_sonnet_response'] = str(claude_response_dict)
                # print("Wating before next call...")
                time.sleep(60)
            response_csv.to_csv(response_csv_path,index=False)
            
            # self.get_matches(response_csv)
        
        # acc = self.get_accuracy()
        # print(f'True Positives : {self.exact_matches} | Total Count : {self.total_count} | Accuracy : {acc}')
    
    def extract_regex(json_string):
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

    def response_post_processing(json_string) :
        json_string = json_string.replace("\n","")
        values = []
        try :
            dict = literal_eval(json_string)
        except :
            try :
                dict = json.loads(json_string)
            except :
                try : 
                    dict = yaml.safe_load(json_string)
                except :
                    dict = {}
        if len(dict) == 0:
            values = extract_regex(json_string)
        else: # coming from json parsing through try-catch
            for key in dict.keys() :
                values.append(dict[key])
        return pd.Series(values)
    
if __name__ == "__main__":
    datastore_path = os.path.join(ROOT_DIR,"datastore")
    folder_paths = os.listdir(datastore_path)
    folder_paths = [i for i in folder_paths if i not in ['metadata.json','.DS_Store']]
    # folder_paths = ['QP_Andhra Pradesh NTSE Stage 1 2017-18 (MAT_SAT_LANG)']
    # folder_paths = [i for i in folder_paths if 'annotations.csv' in os.listdir(os.path.join(datastore_path,i))]
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-f', '--folder_list', help='The list of folder paths to run inference on', required=True)
    # args = vars(parser.parse_args())
    # folder_paths = args['folder_list'].split(',')
    
    inference_obj = ClaudeInference(folder_paths)
    print('Starting Inference..')
    inference_obj.run_inference()
