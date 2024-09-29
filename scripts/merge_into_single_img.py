# from matplotlib import pyplot as plt
import cv2
import numpy as np
from collections import defaultdict
import pandas as pd
import base64
import json
import mimetypes
import os
import requests
import sys
import shutil
import argparse
from tqdm import tqdm

# parser = argparse.ArgumentParser(description='Prepare uniform directory structure')
# parser.add_argument('-pp', '--paper_path', help='The name of paper', required=False)
# args = vars(parser.parse_args())

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

if __name__ == "__main__" :

    # Enter a root path
    root_path = "./"

    # if args["paper_path"] is None:
    #         raise ValueError("Required --paper_path value not specified!")
    # else :
    #      folder_path = os.path.join("datastore", args["paper_path"])

    datastore_path = os.path.join(root_path,'datastore')
    folder_paths = os.listdir(datastore_path)
    folder_paths = [i for i in folder_paths if i not in ['metadata.json','.DS_Store']]
    folder_paths = [i for i in folder_paths if 'annotations.csv' in os.listdir(os.path.join(datastore_path,i))]
    folder_paths = [i for i in folder_paths if 'merged_screenshots' not in os.listdir(os.path.join(datastore_path,i))]

    for folder_path_idx in tqdm(range(len(folder_paths))) :
        folder_path = folder_paths[folder_path_idx]
        image_dict = defaultdict(list)
        annotation_csv = pd.read_csv(os.path.join(datastore_path,folder_path,'annotations.csv'))
        qc_map = dict((q+".png",c) for q,c in zip(annotation_csv["input_image_location-input"],annotation_csv["context-input"]))
        new_ss_folder_path = os.path.join(datastore_path,folder_path,'merged_screenshots')
        if not os.path.isdir(new_ss_folder_path):
            os.makedirs(new_ss_folder_path, exist_ok=True)
              
        for q_img,c_img in qc_map.items():
            if c_img == 'NO_CONTEXT' :
                shutil.copy2(os.path.join(datastore_path,folder_path,"screenshots",q_img),
                            os.path.join(new_ss_folder_path,q_img))
                continue
            merged_img = merge_images([cv2.imread(os.path.join(datastore_path,folder_path,"screenshots",c_img)),
                                    cv2.imread(os.path.join(datastore_path,folder_path,"screenshots",q_img))])
            cv2.imwrite(os.path.join(new_ss_folder_path,q_img), merged_img)
