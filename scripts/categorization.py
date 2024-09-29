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

load_dotenv("../.env")
API_KEY = os.environ.get("GEMINI_API_KEY")
# API_KEY = os.environ.get("GPT4_API_KEY")
do_gemini = True
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

with open("prompts.json") as f:
    file = json.load(f)
    f.close()
PROMPT = file["categorization"]


def get_template(
        img_paths,
        pre_context=PROMPT,
):
    img_list = []
    for img_path in img_paths:
        img_list.append(Image.open(img_path))

    template = [pre_context] + img_list
    return template


def gemini_response(img_paths):
    template = get_template(img_paths=img_paths)
    response = model.generate_content(template, stream=True)
    response.resolve()
    try:
        response_text = response.candidates[0].content.parts[0].text
    except Exception as e:
        logging.debug(f"GeminiApiException: {e}")
        response_text = response.prompt_feedback
    return response_text


def merge_files(cat_df):
    err_count = 0
    for row in cat_df.iterrows():
        try:
            cat_df.loc[row[0], 'category'] = json.loads(row[1]['category-reponse'])['category']
            cat_df.loc[row[0], 'contains_diagram'] = json.loads(row[1]['category-reponse'])['contains_diagram']
        except:
            try:
                cat_df.loc[row[0], 'category'] = literal_eval(row[1]['category-reponse'])['category']
                cat_df.loc[row[0], 'contains_diagram'] = literal_eval(row[1]['category-reponse'])['contains_diagram']
            except:
                err_count += 1
                cat_df.loc[row[0], 'category'] = None
                cat_df.loc[row[0], 'contains_diagram'] = None
    return cat_df, err_count


def encode_range(x):
    if x.isalpha() and len(x) == 1:
        return "A-D"
    elif x.isnumeric() and len(x) == 1:
        return "1-4"
    return None


if __name__ == "__main__":
    root_path = "../"
    datastore_path = os.path.join(root_path, 'datastore')
    folder_paths = os.listdir(datastore_path)
    folder_paths = [i for i in folder_paths if i not in ['metadata.json', '.DS_Store']]
    folder_paths = [i for i in folder_paths if 'merged_screenshots' in os.listdir(os.path.join(datastore_path, i))]
    for folder_path_idx in tqdm(range(len(folder_paths))):
        folder_path = folder_paths[folder_path_idx]
        print(f"FOLDER : {folder_path}")
        merged_ss_path = os.path.join(datastore_path, folder_path, 'merged_screenshots')
        categorizations_csv_path = os.path.join(os.path.join(datastore_path, folder_path), 'categorizations.csv')
        if os.path.exists(categorizations_csv_path):
            print(F"file exists : {categorizations_csv_path}\n")
            continue

        if (folder_path_idx % 3 == 0) and (folder_path_idx != 0):
            time.sleep(60)

        files = os.listdir(merged_ss_path)
        cat_dict_g = dict()
        for idx in tqdm(range(len(files))):
            file = files[idx]
            q_no = file.split(".")[0]
            response = gemini_response([os.path.join(merged_ss_path, file)])
            try:
                cat_dict_g[q_no] = response.replace("```json", "").replace("\n", "").replace("```", "")
            except:
                print(f"ERROR - STORING RAW REPONSE : {response}")
                cat_dict_g[q_no] = response
            time.sleep(5)
        annotations_csv_path = os.path.join(os.path.join(datastore_path, folder_path), 'annotations.csv')
        annotations_csv = pd.read_csv(annotations_csv_path)
        annotations_csv['category-reponse'] = annotations_csv['input_image_location-input'].apply(
            lambda x: cat_dict_g[x])

        annotations_csv.to_csv(categorizations_csv_path, index=False)  # writing categorizations.csv
        annotations_csv, err_count = merge_files(annotations_csv)
        # final_df.to_csv(annotations_file_name,index=False) # writing annotations.csv
        annotations_csv["final_answer_range-output"] = annotations_csv["final_answer-output"].apply(
            lambda x: encode_range(str(x)))
        annotations_csv.to_csv(annotations_csv_path, index=False)
        print('----------------------------------------\n')