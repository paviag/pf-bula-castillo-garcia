import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


class ProcessedAnnotation:
    def __init__(self, annotations, test_size=0.2, target_size=(640, 640)):
        self.annotations = pd.read_csv(annotations)
        self.test_size = test_size
        self.target_width, self.target_height = target_size

        self._reproportion_class_distribution()
        self._set_splits()
        self._fix_invalid_boxes()
        self._scale_bounding_boxes()
        self._add_yolo_label_cols()

    def _reproportion_class_distribution(self, neg_size=0.15):
        """Returns reproportioned annotations with adjusted positive/negative distribution"""
        # Filtering positive/negative cases
        pos = self.annotations[self.annotations.finding_categories.apply(lambda x: any(
            c in x for c in ["Mass", "Suspicious Calcification", "Suspicious Lymph Node"]))]
        pos = pos.dropna(subset=['finding_birads', 'xmin'])
        neg = self.annotations[(self.annotations['finding_birads'].isna()) & (
            ~self.annotations['image_id'].isin(pos.image_id))]

        # Adjusting to compensate for class imbalance to include input neg_size
        neg = neg.sample(n=int(len(pos) / (1-neg_size)
                         * neg_size), random_state=0)
        pos.loc[:, 'finding_categories'] = 0
        neg.loc[:, 'finding_categories'] = 1

        self.annotations = pd.concat([pos, neg], ignore_index=True)

    from sklearn.model_selection import train_test_split

    def _set_splits(self):
        """Returns annotations with set train, val, test splits for stratified train/test split"""
        # Get unique image IDs with finding categories
        unique_images = self.annotations.drop(columns=['xmin', 'ymin', 'xmax', 'ymax']).drop_duplicates(subset=['image_id'])

        # Separate based on finding_categories
        images_0 = unique_images[unique_images['finding_categories'] == 0]
        images_1 = unique_images[unique_images['finding_categories'] == 1]

        # Create mapping of image_id to annotation indices
        imid_indexes = {im_id: self.annotations[self.annotations.image_id == im_id].index.tolist()
                        for im_id in self.annotations.image_id.unique()}

        # Stratify positive cases (finding_categories = 0) into train, val, test
        train_0, temp_0 = train_test_split(images_0, test_size=0.30, random_state=0)
        val_0, test_0 = train_test_split(temp_0, test_size=0.50, random_state=0)  # Splitting remaining 30% into 15/15

        # Stratify negative cases (finding_categories = 1) into train, test
        train_1, test_1 = train_test_split(images_1, test_size=0.30, random_state=0)  # No validation for this group

        # Combine train, val, and test sets
        train_images = pd.concat([train_0, train_1])
        val_images = val_0  # Only positive cases for model validation during training
        test_images = pd.concat([test_0, test_1])

        # Convert image IDs to lists
        train_ids = train_images['image_id'].tolist()
        val_ids = val_images['image_id'].tolist()
        test_ids = test_images['image_id'].tolist()

        # Assign split values to annotations
        self.annotations.loc[sum([imid_indexes[im] for im in train_ids], []), 'split'] = 'train'
        self.annotations.loc[sum([imid_indexes[im] for im in val_ids], []), 'split'] = 'val'
        self.annotations.loc[sum([imid_indexes[im] for im in test_ids], []), 'split'] = 'test'

    def _set_splitsold(self):
        """Returns annotations with set splits for stratified train/test split"""
        # Get unique image IDs and their corresponding finding categories
        unique_images = self.annotations.drop(
            columns=['xmin', 'ymin', 'xmax', 'ymax']).drop_duplicates(subset=['image_id'])
        
        # Create a mapping of image_id to all corresponding annotation indices
        imid_indexes = {im_id: self.annotations[self.annotations.image_id == im_id].index.tolist()
                        for im_id in self.annotations.image_id.unique()}
        
        # Split the unique images dataset while stratifying by finding_categories
        train_images, test_images = train_test_split(
            unique_images,
            test_size=self.test_size,
            random_state=0,
            stratify=unique_images['finding_categories']
        )
        
        # Get the train and test image IDs
        train_ids = train_images['image_id'].tolist()
        test_ids = test_images['image_id'].tolist()
        
        # Set split values for all annotations based on their image_id
        self.annotations.loc[sum([imid_indexes[im] for im in train_ids], []), 'split'] = 'train'
        self.annotations.loc[sum([imid_indexes[im] for im in test_ids], []), 'split'] = 'val'

    def _scale_bounding_boxes(self):
        """Scales annotations' bounding boxes to target dimensions"""
        for col in ['xmin', 'xmax']:
            self.annotations[col] *= self.target_width / self.annotations.width
        for col in ['ymin', 'ymax']:
            self.annotations[col] *= self.target_height / self.annotations.height
        self.annotations.loc[:, 'width'] = self.target_width
        self.annotations.loc[:, 'height'] = self.target_height

    def _fix_invalid_boxes(self):
        """Clips box coordinates between 0 and width (x-coordinates) or height (y-coordinates)"""
        self.annotations[['xmin', 'xmax']] = self.annotations[['xmin', 'xmax']].apply(
            lambda x: x.clip(upper=self.annotations['width'], lower=0), axis=0)
        self.annotations[['ymin', 'ymax']] = self.annotations[['ymin', 'ymax']].apply(
            lambda x: x.clip(upper=self.annotations['height'], lower=0), axis=0)

    def _add_yolo_label_cols(self):
        """Adds columns of yolo label fields for bounding boxes"""
        self.annotations.loc[:, "bx"] = (self.annotations["xmax"] + self.annotations["xmin"]) / (2 * self.annotations["width"])
        self.annotations.loc[:, "by"] = (self.annotations["ymax"] + self.annotations["ymin"]) / (2 * self.annotations["height"])
        self.annotations.loc[:, "bh"] = (self.annotations["ymax"] - self.annotations["ymin"]) / self.annotations["height"]
        self.annotations.loc[:, "bw"] = (self.annotations["xmax"] - self.annotations["xmin"]) / self.annotations["width"]
