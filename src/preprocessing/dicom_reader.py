import cv2
import numpy as np
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut

class DicomProcessor:
    def __init__(self, voi_lut=True, clahe_clip=2.0, clahe_grid=(8, 8)):
        self.voi_lut = voi_lut
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)

    def read_xray(self, dicom_path, photometric_interpretation):
        """
        Returns processed image from dicom path.
        Fixes monochrome and applies CLAHE filter for segmentation.
        
        Based on: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
        """
        dicom = dcmread(dicom_path)
        # Transforms raw DICOM data to "human-friendly" view
        data = apply_voi_lut(dicom.pixel_array, dicom) if self.voi_lut else dicom.pixel_array
        # Inverts X-ray if needed
        if photometric_interpretation == "MONOCHROME1":
            data = np.amax(data) - data
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255  # Normalization
        clahe = self.clahe.apply(data.astype(np.uint8)) # Clahe filter
        return clahe
