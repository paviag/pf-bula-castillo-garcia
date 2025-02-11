import os
import shutil
import pydicom
import cv2
import numpy as np

def convert_dicom_to_png(dicom_path, png_path):
    try:
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array
        
        # Normalize to 8-bit grayscale
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
        image = image.astype(np.uint8)
        
        # Save as PNG
        cv2.imwrite(png_path, image)
    except Exception as e:
        print(f"Error converting {dicom_path}: {e}")

# Define paths
source_dir = "./data/images"
dest_dir = "D:\processed_imgs"

# Copy folders and convert DICOM to PNG
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)
    dest_folder_path = os.path.join(dest_dir, folder)
    
    if os.path.isdir(folder_path):
        os.makedirs(dest_folder_path, exist_ok=True)
        
        for file in os.listdir(folder_path):
            if (file.lower().endswith(".dicom") or file.lower().endswith(".dcm")):
                dicom_path = os.path.join(folder_path, file)
                png_filename = os.path.splitext(file)[0] + ".png"
                png_path = os.path.join(dest_folder_path, png_filename)
                convert_dicom_to_png(dicom_path, png_path)
            else:
                # Copy non-DICOM files as-is
                shutil.copy(os.path.join(folder_path, file), os.path.join(dest_folder_path, file))

print("Conversion completed.")