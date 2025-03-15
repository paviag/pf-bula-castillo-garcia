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

def read_xray(path_or_bytes, photometricInterpretation, voi_lut=True, fix_monochrome=True):
    """
    Returns processed image from dicom path.
    Fixes monochrome and applies CLAHE filter for segmentation.
    """
    # Based on: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = dcmread(path_or_bytes)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # Depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and photometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    # CLAHE filter
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Adjust parameters
    clahe_img = clahe.apply(data)

    return clahe_img

def get_img_from_zip(series_id, study_id, image_id, myzip):
    """
    Returns image file corresponding to series_id, study_id, image_id from dataset zip
    """
    photometricInterpretation = metadata[
        metadata['Series Instance UID'] == series_id
        ].iloc[0]['Photometric Interpretation']

    with myzip.open(f"{dicom_dir[3:-4]}/images/{study_id}/{image_id}.dicom") as myfile:
        return read_xray(DicomBytesIO(myfile.read()), photometricInterpretation)

def reproportion_class_distribution(annotations):
    """
    Returns reproportioned annotations with 85/15 positive/negative distribution
    """
    # Filtering positive/negative cases
    pos = pd.DataFrame([
        annotations.iloc[i] for i in range(len(annotations)) 
        if ('Mass' in annotations.iloc[i].finding_categories 
        or 'Suspicious Calcification' in annotations.iloc[i].finding_categories
        or 'Suspicious Lymph Node' in annotations.iloc[i].finding_categories)
        and not pd.isna(annotations.iloc[i].finding_birads)])
    
    neg = annotations[
        (annotations['finding_birads'].isna())
        & (~annotations['image_id'].isin(pos.image_id.unique()))
        ]
    
    if any(ipos in neg.image_id.unique() for ipos in pos.image_id.unique()):
        raise Exception("Overlapping pos and neg")
    
    if not neg.finding_birads.isna().values.any():
        raise Exception("Negative findings contain a positive")
    
    # Adjusting to compensate for class imbalance to include 85% pos, 15% neg
    neg = neg.sample(n=int(len(pos)/0.85*0.15), random_state=1)

    # Join into new dataframe
    pos['finding_categories'] = 0
    neg['finding_categories'] = 1
    annotations = pd.concat([pos, neg], ignore_index=True)

    return annotations

def set_splits(annotations):
    """
    Returns annotations with set splits for stratified train/test split of 80/20
    """
    # Sort train/test indices with an 80/20 split
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    # Stores annotations with no boxes for stratified split indices
    annot_no_boxes = annotations.drop(columns=['xmin', 'ymin', 'xmax', 'ymax'])
    # Stores indexes in annotations for each image_id
    imid_indexes = pd.Series()
    for im_id in annotations.image_id.unique():
        imid_indexes.loc[im_id] = annotations[annotations.image_id == im_id].index.tolist()
        annot_no_boxes.drop(  # Keep only one row per image
            index=imid_indexes[im_id][1:], 
            inplace=True)  
    
    # Split images into train/test and set split for all rows belonging to that
    # image in annotations
    for i, (train_index, test_index) in enumerate(sss.split(
          annot_no_boxes.drop(columns=['finding_categories']),
          annot_no_boxes.finding_categories,
          )):
        im_id = annot_no_boxes.iloc[train_index].image_id
        annotations.loc[
            [i for j in imid_indexes[im_id].values.tolist() for i in j], 
            'split'] = 'train'
        im_id = annot_no_boxes.iloc[test_index].image_id
        annotations.loc[
            [i for j in imid_indexes[im_id].values.tolist() for i in j], 
            'split'] = 'val'
    
    return annotations

def format_annotations(annotations):
    """
    Formats annotation file adjusting class distribution and setting train/test split indices
    """
    # Parses finding_categories strings to lists
    annotations.finding_categories = annotations.finding_categories.replace("'", '"', regex=True).apply(json.loads)
    # Sets negative coords to 0 and coords outside image to max width/height
    annotations[['xmin', 'xmax']] = annotations[['xmin', 'xmax']].apply(lambda x: x.clip(upper=annotations['width'], lower=0), axis=0)  
    annotations[['ymin', 'ymax']] = annotations[['ymin', 'ymax']].apply(lambda x: x.clip(upper=annotations['height'], lower=0), axis=0)
    # Reproportions class distribution
    annotations = reproportion_class_distribution(annotations)
    # Sets train/test split indices
    annotations = set_splits(annotations)
    return annotations

dicom_dir = "D:/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0.zip"
annot_dir = "./data/finding_annotations.csv"
metadata_dir = "./data/metadata.csv"

metadata = pd.read_csv(metadata_dir)
annotations = format_annotations(pd.read_csv(annot_dir))

processed_base_path = "./data/processed_imgs"
os.makedirs(processed_base_path, exist_ok=True)    # Creates processed images dir

with zipfile.ZipFile(dicom_dir) as myzip:
    for im_id, rows in annotations.groupby('image_id'):
        # Gets image from zip
        first = rows.head(1)
        im = get_img_from_zip(
            first.series_id.item(), 
            first.study_id.item(), 
            im_id, 
            myzip,
        )
            
        # Gets original image dimensions
        original_height, original_width = im.shape[:2]
    
        # Target dimensions
        target_width, target_height = 640, 640 # YOLO's internal input image size
    
        # Scaling factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height
    
        # Resizes image
        im_resized = cv2.resize(im, (target_width, target_height))
    
        # Saves resized image
        new_img_path = f"{processed_base_path}/{first.study_id.item()}/{first.image_id.item()}.jpg"
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
        cv2.imwrite(new_img_path, im_resized)
    
        for i, row in rows.iterrows():
            # Scales bounding box information accordingly
            annotations.at[i, "xmin"] = row.xmin = row.xmin * scale_x
            annotations.at[i, "xmax"] = row.xmax = row.xmax * scale_x
            annotations.at[i, "ymin"] = row.ymin = row.ymin * scale_y
            annotations.at[i, "ymax"] = row.ymax = row.ymax * scale_y

            annotations.at[i, "bx"] = (row.xmax + row.xmin) / (2*target_width)
            annotations.at[i, "by"] = (row.ymax + row.ymin) / (2*target_height)
            annotations.at[i, "bh"] = (row.ymax - row.ymin) / target_height
            annotations.at[i, "bw"] = (row.xmax - row.xmin) / target_width
    
            # Updates image path on Dataframe
            annotations.at[i, "directory_path"] = new_img_path

# Saves annotations as csv
annotations.to_csv("pf-bula-castillo-garcia/annotationsv1.csv", index=False)