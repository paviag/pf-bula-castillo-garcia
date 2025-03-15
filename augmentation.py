import os
import pandas as pd
import albumentations as A
import cv2

N_AUGMENTATIONS = 3 # Number of augmentations per image

def augment_image(image_path, class_labels, bboxes, transform):
    """Returns transformed image and labels from input paths and transform function"""
    image = cv2.imread(image_path)
    transformed = transform(
        image=image, 
        bboxes=bboxes, 
        class_labels=class_labels, 
        )
    return transformed

def save_image(path, image):
    """Saves image to path"""
    #os.makedirs(path, exist_ok=True)
    cv2.imwrite(
        path, 
        image,
    )

annotations_dir = "pf-bula-castillo-garcia/annotationsv1.csv"
processed_base_path = "./data/processed_imgs"

# Transform function for augmentation
transform = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

annotations_dir = "pf-bula-castillo-garcia/annotationsv1.csv"
annotations = pd.read_csv(annotations_dir)
new_annotations = pd.DataFrame()
for im_id, rows in annotations.groupby('image_id'):
    first = rows.head(1)
    img_path = f"{first.directory_path.item()}"
    bboxes = rows[["bx", "by", "bw", "bh"]].values.tolist()
    if first.finding_categories.item() == 0:
        class_labels = [0]*len(bboxes)
    else:
        bboxes = []
        class_labels = []
    # Each image will be augmented N_AUGMENTATIONS number of times
    for i in range(N_AUGMENTATIONS):
        # Augment image
        transformed = augment_image(
            img_path, 
            class_labels,
            bboxes,
            transform,
        )
        # Save aumented image to proccesed images folder
        new_path = f"{processed_base_path}/{first.study_id.item()}/{im_id}_{i}.jpg"
        save_image(
            new_path,
            transformed["image"],
        )
        
        if len(bboxes) == 0:
            new_row = first.copy()
            new_row.image_id = f"{im_id}_{i}"
            new_row.directory_path = new_path
            new_annotations = pd.concat([new_annotations, new_row], ignore_index=True)
            continue

        # Add to annotations
        for bbox in transformed["bboxes"]:
            new_row = first.copy()
            new_row.image_id = f"{im_id}_{i}"
            new_row.directory_path = new_path
            new_row.bx, new_row.by, new_row.bw, new_row.bh = bbox
            new_annotations = pd.concat([new_annotations, new_row], ignore_index=True)

# Update annotations
new_annotations.to_csv("pf-bula-castillo-garcia/annotationsv2.csv", index=False)