from preprocessing.dicom_reader import DicomProcessor
from preprocessing.annotation_processor import ProcessedAnnotation
from preprocessing.image_handler import ImageDatasetHandler
from config import config


def run_preprocessing():
    dicom_zip = config.dicom_zip
    annotations_path = f"{config.input_data_path}/finding_annotations.csv"
    metadata_path = f"{config.input_data_path}/metadata.csv"
    output_image_dir = config.output_image_path
    output_data_dir = config.output_data_path

    processor = DicomProcessor()
    # default target size 640x640
    image_handler = ImageDatasetHandler(
        dicom_zip, metadata_path, output_image_dir)
    # default test size 0.2 and target size 640x640
    annotations = ProcessedAnnotation(annotations_path).annotations

    for im_id, rows in annotations.groupby('image_id'):
        first = rows.iloc[0]
        # Extracts processed dicom file from zip
        image = image_handler.extract_image(first.series_id, first.study_id,
                                            im_id, processor)
        # Resizes image
        resized_image = image_handler.resize(image)
        # Saves image
        img_path = image_handler.save_image(
            resized_image, first.study_id, im_id)
        # Modifies directory path to newly processed image
        annotations.loc[annotations.image_id ==
                        im_id, 'directory_path'] = img_path

    # Saves annotations as csv
    annotations.to_csv(output_data_dir+"/annotations.csv", index=False)
    annotations.to_csv(output_data_dir+"/annotations_processed.csv", index=False)
