import os
import cv2
import json
import numpy as np
import pandas as pd
import zipfile
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.filebase import DicomBytesIO
from sklearn.model_selection import StratifiedShuffleSplit

def read_xray(path_or_bytes, voi_lut = True, fix_monochrome = True):
    # Based on: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = dcmread(path_or_bytes)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # Depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    # CLAHE filter
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Adjust parameters
    clahe_img = clahe.apply(data)

    return clahe_img

def get_img_from_zip(study_id, image_id, myzip):
    with myzip.open(f"{dicom_dir[3:-4]}/images/{study_id}/{image_id}.dicom") as myfile:
        return read_xray(DicomBytesIO(myfile.read()))

dicom_dir = "D:/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0.zip"
metadata_dir = "./data/finding_annotations.csv"
annotations = pd.read_csv(metadata_dir)
annotations.finding_categories = annotations.finding_categories.replace("'", '"', regex=True).apply(json.loads)

# Sets negative coords to 0
for col in ['xmin', 'ymin', 'xmax', 'ymax']:
  annotations[col] = annotations[col].apply(lambda x: 0 if x < 0 else x)

### Filtering malign / benign
masses_calcif = pd.DataFrame([annotations.iloc[i] for i in range(len(annotations)) if 'Mass' in annotations.iloc[i].finding_categories or 'Suspicious Calcification' in annotations.iloc[i].finding_categories])
healthy = annotations[annotations['finding_birads'].isna()]

### Adjusting to compensate for class imbalance to include 85% sick, 15% healthy
healthy = healthy.sample(n=int(len(masses_calcif)/0.85*0.15), random_state=1)

### Join in new dataframe
masses_calcif['finding_categories'] = 1
healthy['finding_categories'] = 0
annotations = pd.concat([masses_calcif, healthy], ignore_index=True)

## Sort train/test indices with an 80/20 split
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

for i, (train_index, test_index) in enumerate(sss.split(
    annotations.drop('finding_categories', axis=1),
    annotations.finding_categories,
    )):
  annotations.loc[train_index, 'split'] = 'train'
  annotations.loc[test_index, 'split'] = 'test'

# Adds columns with bounding box information
annotations["bx"] = (annotations["xmax"] + annotations["xmin"]) / 2
annotations["by"] = (annotations["ymax"] + annotations["ymin"]) / 2
annotations["bh"] = annotations["ymax"] - annotations["ymin"]
annotations["bw"] = annotations["xmax"] - annotations["xmin"]

processed_base_path = "./data/processed_imgs"

os.makedirs(processed_base_path, exist_ok=True)

with zipfile.ZipFile(dicom_dir) as myzip:
    for i in range(annotations.shape[0]):
        row = annotations.iloc[i]
        
        im = get_img_from_zip(row.study_id, row.image_id, myzip)
        
        # Gets original image dimensions
        original_height, original_width = im.shape[:2]

        # Target dimensions
        target_width, target_height = 640, 640 # YOLO's internal input image size

        # Scaling factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        # Resizes image
        im_resized = cv2.resize(im, (target_width, target_height))

        # Scales bounding box information accordingly
        annotations.at[i, "xmin"] = row.xmin * scale_x
        annotations.at[i, "xmax"] = row.xmax * scale_x
        annotations.at[i, "ymin"] = row.ymin * scale_y
        annotations.at[i, "ymax"] = row.ymax * scale_y

        annotations.at[i, "bx"] = (annotations.at[i, "xmax"] + annotations.at[i, "xmin"]) / 2
        annotations.at[i, "by"] = (annotations.at[i, "ymax"] + annotations.at[i, "ymin"]) / 2
        annotations.at[i, "bh"] = annotations.at[i, "ymax"] - annotations.at[i, "ymin"]
        annotations.at[i, "bw"] = annotations.at[i, "xmax"] - annotations.at[i, "xmin"]

        # Saves resized image
        new_img_path = f"{processed_base_path}/{row.study_id}/{row.image_id}.png"
        
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
        
        cv2.imwrite(new_img_path, im_resized)

        # Updates image path on Dataframe
        annotations.at[i, "directory_path"] = new_img_path

# Saves annotations as csv
annotations.to_csv("pf-bula-castillo-garcia/annotationsv1.csv")