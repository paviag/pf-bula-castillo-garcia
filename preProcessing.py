import os
import cv2
import json
from PIL import Image
import pandas as pd
import numpy as np


images_dir = "D:/processed_imgs"
metadata_dir = "./data/finding_annotations.csv"
annotations = pd.read_csv(metadata_dir)
annotations.finding_categories = annotations.finding_categories.replace("'", '"', regex=True).apply(json.loads)


categories = list(set([a for b in annotations.finding_categories for a in b]))
#print(len(categories))

annotations["bx"] = (annotations["xmax"] + annotations["xmin"]) / 2
annotations["by"] = (annotations["ymax"] + annotations["ymin"]) / 2
annotations["bh"] = annotations["ymax"] - annotations["ymin"]
annotations["bw"] = annotations["xmax"] - annotations["xmin"]

#redimensionar

len_df = annotations.shape[0]
resized_base_path = "./data/resized"

os.makedirs(resized_base_path, exist_ok=True)

for i in range(len_df):
    row = annotations.iloc[i]
    
    img_path = f'D:/processed_imgs/{row.study_id}/{row.image_id}.png'
    
    im = cv2.imread(img_path)
    
    if im is None:
        print(f"Warning: Image at {img_path} could not be loaded.")
        continue 
    
    # Obtener dimensiones originales de la imagen
    original_height, original_width = im.shape[:2]

    # Dimensiones objetivo
    target_width, target_height = 288, 288 #Para YOLO probé con esta escala, toca ver si podemos mejorar esto

    # Calcular factores de escala
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    # Redimensionar la imagen
    im_resized = cv2.resize(im, (target_width, target_height))

    # Aplicar la transformación de escala a las coordenadas de la bounding box
    annotations.at[i, "xmin"] = row.xmin * scale_x
    annotations.at[i, "xmax"] = row.xmax * scale_x
    annotations.at[i, "ymin"] = row.ymin * scale_y
    annotations.at[i, "ymax"] = row.ymax * scale_y

    # Recalcular las bounding boxes después del resize
    annotations.at[i, "bx"] = (annotations.at[i, "xmax"] + annotations.at[i, "xmin"]) / 2
    annotations.at[i, "by"] = (annotations.at[i, "ymax"] + annotations.at[i, "ymin"]) / 2
    annotations.at[i, "bh"] = annotations.at[i, "ymax"] - annotations.at[i, "ymin"]
    annotations.at[i, "bw"] = annotations.at[i, "xmax"] - annotations.at[i, "xmin"]

    # Guardar la imagen redimensionada
    new_img_path = f"{resized_base_path}/{row.study_id}/{row.image_id}.png"
    
    os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
    
    cv2.imwrite(new_img_path, im_resized)
    
    # Actualizar la ruta de la imagen en el DataFrame
    annotations.at[i, "directory_path"] = new_img_path

annotations.to_csv("pf-bula-castillo-garcia/annotationsv1.csv")