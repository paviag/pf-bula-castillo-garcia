from augmentation.image_augmentation import ImageAugmentor
from augmentation.annotation_handler import AnnotationHandler
from config import config

def run_augmentations():
    annotations_dir = f"{config.output_data_path}/annotations.csv"
    processed_base_path = config.output_image_path
    output_annotations_path = f"{config.output_data_path}/annotations.csv"

    augmentor = ImageAugmentor()    # default number of augmentations is 3
    annotation_handler = AnnotationHandler(annotations_dir)
    annotation_handler.process_annotations(augmentor, processed_base_path)
    annotation_handler.save_annotations(output_annotations_path)