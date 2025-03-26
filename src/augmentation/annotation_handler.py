import pandas as pd
import os
import cv2


class AnnotationHandler:
    """Handles loading, processing, and saving annotation data"""

    def __init__(self, annotations_path):
        self.annotations = pd.read_csv(annotations_path)
        self.new_annotations = pd.DataFrame()

    def process_annotations(self, augmentor, processed_base_path):
        """Processes annotations and augments images."""
        for im_id, rows in self.annotations.groupby('image_id'):
            first = rows.head(1)
            img_path = first.directory_path.item()
            if first.finding_categories.item() == 0:
                bboxes = rows[['bx', 'by', 'bw', 'bh']].values.tolist()
                class_labels = [0]*len(bboxes)
            else:
                bboxes = []
                class_labels = []
            # Each image will be augmented a number of times
            for i in range(augmentor.num_augmentations):
                # Augment image
                transformed = augmentor.augment_image(
                    img_path, class_labels, bboxes)
                new_path = f"{processed_base_path}/{first.study_id.item()}/{im_id}_{i}.jpg"
                # Save augmented image to proccesed images folder
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                cv2.imwrite(new_path, transformed['image'])
                # Add to annotations
                self._update_annotations(
                    first, im_id, i, new_path, transformed['bboxes'])

    def _update_annotations(self, first, im_id, i, new_path, transformed_bboxes):
        """Updates annotation dataframe with new augmented image data."""
        # Image with no boxes
        if not transformed_bboxes:
            new_row = first.copy()
            new_row.image_id = f"{im_id}_{i}"
            new_row.directory_path = new_path
            self.new_annotations = pd.concat(
                [self.new_annotations, new_row], ignore_index=True)
            return
        # Image with boxes
        for bbox in transformed_bboxes:
            new_row = first.copy()
            new_row.image_id = f"{im_id}_{i}"
            new_row.directory_path = new_path
            new_row.bx, new_row.by, new_row.bw, new_row.bh = bbox
            self.new_annotations = pd.concat(
                [self.new_annotations, new_row], ignore_index=True)

    def save_annotations(self, output_path):
        """Saves the updated annotations."""
        self.new_annotations.to_csv(output_path, index=False)
