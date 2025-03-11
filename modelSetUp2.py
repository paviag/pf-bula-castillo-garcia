import shutil
import yaml
import os
import pandas as pd

annotations_dir = "pf-bula-castillo-garcia/annotationsv1.csv"
annotations = pd.read_csv(annotations_dir)

yolo_labels_path = "pf-bula-castillo-garcia/yolo_labels/"
os.makedirs(yolo_labels_path, exist_ok=True)

for im_id, rows in annotations.groupby('image_id'):
    first = rows.head(1)
    img_path = f'{first.directory_path.item()}'
    
    # Create YOLO labels
    if first.finding_categories.item() == 0:
        labels = "\n".join([f"{row.finding_categories} {row.bx} {row.by} {row.bw} {row.bh}" for i, row in rows.iterrows()])
        yolo_label_path = os.path.join(yolo_labels_path, f"{im_id}.txt")
        with open(yolo_label_path, "w") as f:
            f.write(labels)
    else:
        yolo_label_path = os.path.join(yolo_labels_path, f"{im_id}.txt")
        with open(yolo_label_path, "w") as f:
            f.write("")

base_yolo_path = "pf-bula-castillo-garcia/dataset/"

for im_id, rows in annotations.groupby('image_id'):
    # Images for training are saved into train folder
    # Images for validation are split into the val folder for YOLO validation with labeled images
    # and the val_neg folder for manual validation with unlabeled images
    first = rows.head(1)
    if first.split.item() == "train":
        split_folder = "train"
    elif first.finding_categories.item() == 0:
        split_folder = "val"
    else:
        split_folder = "val_neg"
    
    # Move image to respective images folder
    img_dest_path = os.path.join(base_yolo_path, "images", split_folder, f"{im_id}.jpg") 
    os.makedirs(os.path.dirname(img_dest_path), exist_ok=True)
    shutil.copy(first.directory_path.item(), img_dest_path)
    
    # Move annotations to YOLO labels folder
    label_src_path = os.path.join(yolo_labels_path, f"{im_id}.txt")
    label_dest_path = os.path.join(base_yolo_path, "labels", split_folder, f"{im_id}.txt")
    os.makedirs(os.path.dirname(label_dest_path), exist_ok=True)
    shutil.copy(label_src_path, label_dest_path)

# Define YOLO configuration
yolo_config = {
    "train": "C:/Users/Lab6k/Documents/PF/pf-bula-castillo-garcia/dataset/images/train",  # Path to train data for training
    "val": "C:/Users/Lab6k/Documents/PF/pf-bula-castillo-garcia/dataset/images/val",      # Path to val data for training
    "nc": 1,    # Number of classes to detect anomalies for (masses & calcifications are merged into one class)
    "names": ["0"],    # Numbered class names must start with 0
}
# Path to YAML model config file
yaml_path = "pf-bula-castillo-garcia/data.yaml"

# Save YAML file
with open(yaml_path, "w") as file:
    yaml.dump(yolo_config, file, default_flow_style=False)

print(f"Archivo YAML guardado en: {yaml_path}")

# Confirm folder creation
dataset_path = "pf-bula-castillo-garcia/dataset/"
expected_folders = ["images/train", "images/val", "images/val_neg", "labels/train", "labels/val"]

for folder in expected_folders:
    full_path = os.path.join(dataset_path, folder)
    print(f"{full_path}: {'✅ Existe' if os.path.exists(full_path) else '❌ No existe'}")