template = """
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
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import pandas as pd
from tqdm import tqdm
import json

root_path = ""
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3")
chain = prompt | model
response_json = "../../datastore/final_datasets/test/image_description_gpt-4o_.json"
with open("../../datastore/final_datasets/test/image_description_gpt-4o.json", "r") as file:
    all_responses = json.load(file) 

for key in tqdm(all_responses.keys()):
    try:
        with open(response_json, "r") as file:
            all_responses_ = json.load(file) 
    except FileNotFoundError:
        # If the file doesn't exist, create a new list
        all_responses_ = dict()
    
    item = all_responses[key]
    contains_diagram = bool(item["contains_diagram"])
    item["image_description_ollama"] = "None"
    if contains_diagram :
        resp = item["gpt-4o_response"]
        
        is_parsed = False
        num_tries = 1
        output = chain.invoke({"json_string": f"{resp}"})
        
        while not is_parsed or num_tries>3:
            try:
                dict_to_add = json.loads(output)
                is_parsed = True
            except:
                print("ORIGINAL OUTPUT: ",  f"{resp}")
                print("="*20)
                print("LLM FAILED OUTPUT: ", output)
                print()
                is_parsed=False
                num_tries+=1
                dict_to_add = {"description":"NotParseable"}
    item["image_description_ollama"] = dict_to_add["description"]
    merged_dict = {**all_responses_, **{key : item}}
    with open(response_json, 'w') as file:
        json.dump(merged_dict, file, indent=4)