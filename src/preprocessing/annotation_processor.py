import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


class ProcessedAnnotation:
    def __init__(self, annotations, target_size=(640, 640)):
        self.annotations = pd.read_csv(annotations)
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

    def _set_splits(self):
        """Returns annotations with set train, val, test splits for stratified 75/15/15 split"""
        # Get unique image IDs with finding categories
        unique_images = self.annotations.drop(columns=['xmin', 'ymin', 'xmax', 'ymax']).drop_duplicates(subset=['image_id'])
        
        # Create a stratification label by combining all relevant columns
        relevant_columns = ['laterality', 'view_position', 'breast_density', 'finding_categories']
        unique_images['stratify_label'] = unique_images.apply(lambda row: '_'.join(map(str, row[relevant_columns])), axis=1)
        
        # Identify rare labels (those with fewer than 2 instances) and replace them with 'Rare'
        label_counts = unique_images['stratify_label'].value_counts()
        rare_labels = label_counts[label_counts < 2].index
        unique_images['stratify_label'] = unique_images['stratify_label'].apply(lambda x: 'Rare' if x in rare_labels else x)
        
        # Split the data into train and temp sets (70% train, 30% temp)
        train_images, temp_images = train_test_split(unique_images, test_size=0.3, stratify=unique_images['stratify_label'], random_state=0)
        
        # Identify rare labels (those with fewer than 2 instances) and replace them with 'Rare'
        label_counts = temp_images['stratify_label'].value_counts()
        rare_labels = label_counts[label_counts < 2].index
        temp_images['stratify_label'] = temp_images['stratify_label'].apply(lambda x: 'Rare' if x in rare_labels else x)
        
        # Split the temp set into validation and test sets
        val_images, test_images = train_test_split(temp_images, test_size=0.5, stratify=temp_images['stratify_label'], random_state=0)

        # Create mapping of image_id to annotation indices
        imid_indexes = {im_id: self.annotations[self.annotations.image_id == im_id].index.tolist()
                        for im_id in self.annotations.image_id.unique()}

        # Convert image IDs to lists
        train_ids = train_images['image_id'].tolist()
        val_ids = val_images['image_id'].tolist()
        test_ids = test_images['image_id'].tolist()

        # Assign split values to annotations
        self.annotations.loc[sum([imid_indexes[im] for im in train_ids], []), 'split'] = 'train'
        self.annotations.loc[sum([imid_indexes[im] for im in val_ids], []), 'split'] = 'val'
        self.annotations.loc[sum([imid_indexes[im] for im in test_ids], []), 'split'] = 'test'

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
