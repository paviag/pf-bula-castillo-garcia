import os
import cv2
import zipfile
import pandas as pd
from pydicom.filebase import DicomBytesIO


class ImageDatasetHandler:
    def __init__(self, dicom_zip, metadata_path, output_dir, target_size=(640, 640)):
        self.target_width, self.target_height = target_size
        self.dicom_zip = dicom_zip
        self.metadata = pd.read_csv(metadata_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_image(self, series_id, study_id, image_id, processor):
        """Returns an image of a processed dicom file"""
        photometric_interpretation = self.metadata[self.metadata['Series Instance UID']
                                                   == series_id].iloc[0]['Photometric Interpretation']
        with zipfile.ZipFile(f"{self.dicom_zip}.zip") as myzip:
            with myzip.open(f"{self.dicom_zip.split('/')[-1]}/images/{study_id}/{image_id}.dicom") as myfile:
                return processor.process(DicomBytesIO(myfile.read()), photometric_interpretation)

    def resize(self, image):
        """Returns resized image"""
        return cv2.resize(image, (self.target_width, self.target_height))

    def save_image(self, image, study_id, image_id):
        """Saves input image into its respective path"""
        path = f"{self.output_dir}/{study_id}/{image_id}.jpg"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)
        return path
