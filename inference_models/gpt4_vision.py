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

load_dotenv("../../.env")

ROOT_DIR = "../"
API_KEY = os.environ.get("GPT4_API_KEY")
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

class GPT4VisionInference:
    def __init__(
            self,
            folders,
            model = "gpt-4-vision-preview",
            pre_image_prompt = PROMPT
    ) :
        self.model = model
        self.pre_image_prompt = pre_image_prompt
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
    
    def query_openai(self,payload):
        """Sends a request to the OpenAI API and prints the response."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=json.loads(payload))
        return response.json()
    
    def local_image_to_data_url(self,image_path):
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_path:
            base64_encoded_data = base64.b64encode(image_path.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    def gen_template(self,
                     images,
                     top_p = 0.1,
                     temp = 0.0
                     ):
        template = {
            'model' : self.model ,
            "top_p" : top_p,
            "temperature" : temp,
            "messages": [
                {"role": "system", "content": "You are a student solving maths and logical reasoning problems with multiple choice questions."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                        "text": self.pre_image_prompt},
                    ],
                },
            ],
            "max_tokens": 2000,
            "stream": False
        }
        for idx,image in enumerate(images):
            if image == '': # no context present
                continue
            
            base64_image = self.local_image_to_data_url(image)
            template["messages"][1]['content'].append({
                "type": "image_url",
                "image_url": {
                    "url": base64_image,
                    "detail": "high",
                }
            })

        return json.dumps(template)
    
    def generate_response(self,imagelist,q_no) :
        payload = self.gen_template(imagelist)
        response = self.query_openai(payload)
        ans = response['choices'][0]['message']['content'].replace("```json","").replace("\n","").replace("```","")
        try : 
            ans = literal_eval(ans)
            return ans
        except :
            ans += '"}'
            print(f"Error while parsing response for {q_no}.Retrying")
            ans = literal_eval(ans)
            return ans
    
    def get_matches(self,response_csv) :
        response_csv['final_answer-output'].fillna('',inplace=True)
        response_csv['final_answer-output'] = response_csv['final_answer-output'].astype(str)
        response_csv['final_answer-output_fmt'] = response_csv['final_answer-output'].apply(lambda val : [x.strip('[').strip(']') for x in val.split(',')])
        response_csv['GPT4_ans'] = response_csv.apply(lambda x : str(ord(x['GPT4_ans']) - ord('A') + 1).upper() if x['final_answer_range-output'] in ['1-4','1,2,3,4'] and not(x['GPT4_ans']>='1' and x['GPT4_ans']<='4' )else str(x['GPT4_ans']).upper(),axis=1)
        response_csv['exact_match'] = response_csv.apply(lambda row : 1 if row['GPT4_ans'] in row['final_answer-output_fmt'] else 0,axis=1)
        self.exact_matches += response_csv.loc[response_csv['exact_match']==1].shape[0]
        self.total_count += response_csv.shape[0]

    def get_accuracy(self) :
        return self.exact_matches/self.total_count
    
    def delete_response_csv(self,del_file):
        if os.path.exists(self.response_csv_path_folder_map[del_file]) :
            os.remove(self.response_csv_path_folder_map[del_file]) 
        return
    
    def run_inference(self):
        for folder,imagelist in self.context_map.items() :
            print(f"Folder : {folder}")
            response_csv_path = os.path.join(folder,'annotations_w_gpt4_response.csv')
            self.response_csv_path_folder_map[folder.split('/')[-1]] = response_csv_path
            if os.path.exists(response_csv_path) : # inferencing already done - simply read it & update counts to get metrics
                print('Inferencing already done')
                response_csv = pd.read_csv(response_csv_path) # read
                self.get_matches(response_csv) # update counts for metrics
                # print(self.exact_matches,self.total_count)
                continue
                
            response_csv = pd.read_csv(os.path.join(folder,'annotations.csv'))

            for idx in tqdm(range(len(imagelist))) :
                image_list = imagelist[idx]
                q_no = image_list[-1].split('/')[-1].split('.')[0].split('_')[0]
                gpt_response_dict = self.generate_response(image_list,q_no)
                response_csv.loc[response_csv['sample_id-input']==q_no,'GPT4_reasoning'] = gpt_response_dict['explanation']
                response_csv.loc[response_csv['sample_id-input']==q_no,'GPT4_ans'] = gpt_response_dict['answer']
                response_csv.loc[response_csv['sample_id-input']==q_no,'GPT4_response'] = str(gpt_response_dict)
            response_csv.to_csv(response_csv_path,index=False)
            self.get_matches(response_csv)
        
        acc = self.get_accuracy()
        print(f'True Positives : {self.exact_matches} | Total Count : {self.total_count} | Accuracy : {acc}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder_list', help='The list of folder paths to run inference on', required=True)
    args = vars(parser.parse_args())

    folders = args['folder_list'].split(',')
    inference_obj = GPT4inference(folders)
    print('Starting Inference..')
    inference_obj.run_inference()
