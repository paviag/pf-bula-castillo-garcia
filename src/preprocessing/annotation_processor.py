import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


class ProcessedAnnotation:
    def __init__(self, annotations, num_groups, target_size=(640, 640)):
        self.annotations = pd.read_csv(annotations)
        self.target_width, self.target_height = target_size

        self._reproportion_class_distribution(num_groups)
        self._set_splits(num_groups)
        self._fix_invalid_boxes()
        self._scale_bounding_boxes()
        self._add_yolo_label_cols()
    
    def _reproportion_class_distribution(self, num_groups):
        """Returns reproportioned annotations with adjusted positive/negative distribution"""

        # Filtering positive cases
        pos = self.annotations[self.annotations.finding_categories.apply(lambda x: any(
            c in x for c in ["Mass", "Suspicious Calcification", "Suspicious Lymph Node"]))]
        pos = pos.dropna(subset=['finding_birads', 'xmin'])

        # Filtering negative cases
        neg = self.annotations[(self.annotations['finding_birads'].isna()) & 
                                (~self.annotations['image_id'].isin(pos.image_id))]

        # Initialize group columns
        self.annotations[[f'group_{i}' for i in range(num_groups)]] = 0

        # Assign all positive cases across all groups
        self.annotations.loc[pos.index, [f'group_{i}' for i in range(num_groups)]] = 1

        # Get unique negative image ids and shuffle them
        unique_neg_images = neg.drop_duplicates(subset=['image_id'])
        neg_image_ids = unique_neg_images.image_id.to_numpy()
        np.random.shuffle(neg_image_ids)

        # Get a random subsample ensuring 70/30 split for positive/negative
        neg_image_ids = np.random.choice(
            neg_image_ids, size=int(len(pos) / 0.7 * 0.3), replace=False)
        # Filter negatives to include only sampled images
        neg = neg[neg.image_id.isin(neg_image_ids)]

        split_size = len(neg_image_ids) // num_groups
        # Assign negatives to groups in equal splits by image_id
        for i in range(num_groups):
            start = i * split_size
            end = (i + 1) * split_size
            assigned_images = neg_image_ids[start:end]
            self.annotations.loc[self.annotations.image_id.isin(assigned_images), f'group_{i}'] = 1
        
        # Distribute leftover samples across groups
        if len(neg_image_ids) % num_groups > 0:
            remaining_images = neg_image_ids[num_groups * split_size:]  # Capture remaining images
            for i, image_id in enumerate(remaining_images):
                self.annotations.loc[self.annotations.image_id == image_id, f'group_{i % num_groups}'] = 1

        # Reassigning finding categories
        self.annotations.loc[pos.index, 'finding_categories'] = 0   # Positive 
        self.annotations.loc[neg.index, 'finding_categories'] = 1   # Negative 

        # Filter annotations to include only assigned rows
        self.annotations = self.annotations.loc[pos.index.tolist() + neg.index.tolist()].reset_index(drop=True)

        return self.annotations

    def _set_splits(self, num_groups):
        """Returns annotations with set train, val, test splits for stratified 75/15/15 split"""

        # Create a stratification label by combining all relevant columns
        relevant_columns = ['laterality', 'view_position', 'breast_density', 'finding_categories']

        # Create mapping of image_id to annotation indices
        imid_indexes = {im_id: self.annotations[self.annotations.image_id == im_id].index.tolist()
                        for im_id in self.annotations.image_id.unique()}
        
        for i in range(num_groups):
            # Create a mask for the current group
            annot_group = self.annotations[self.annotations[f'group_{i}'] == 1].copy()
            
            # Get unique image IDs with finding categories
            annot_group = annot_group.drop(columns=[f'group_{j}' for j in range(num_groups) if f'group_{j}' in annot_group.columns])
            unique_images = annot_group.drop(columns=['xmin', 'ymin', 'xmax', 'ymax']).drop_duplicates(subset=['image_id'])
            
            # Create a stratification label by combining all relevant columns
            unique_images['stratify_label'] = unique_images.apply(lambda row: '_'.join(map(str, row[relevant_columns])), axis=1)
            # Identify rare labels (those with fewer than 2 instances) and replace them with 'Rare'
            label_counts = unique_images['stratify_label'].value_counts()
            rare_labels = label_counts[label_counts < 2].index
            if len(rare_labels) == 1:
                # Replace rare labels with the second most common label
                unique_images[unique_images['stratify_label'] == rare_labels[0]]['stratify_label'] = label_counts.index[-2]
            else:
                # Replace rare labels with 'Rare'
                unique_images['stratify_label'] = unique_images['stratify_label'].apply(lambda x: 'Rare' if x in rare_labels else x)
           
            # Split the data into train and temp sets (70% train, 30% temp)
            train_images, temp_images = train_test_split(unique_images, test_size=0.3, stratify=unique_images['stratify_label'], random_state=0)
            
            # Identify rare labels (those with fewer than 2 instances) and replace them with 'Rare'
            label_counts = temp_images['stratify_label'].value_counts()
            rare_labels = label_counts[label_counts < 2].index
            if len(rare_labels) == 1:
                # Replace rare labels with the second most common label
                temp_images[unique_images['stratify_label'] == rare_labels[0]]['stratify_label'] = label_counts.index[-2]
            else:
                # Replace rare labels with 'Rare'
                temp_images['stratify_label'] = temp_images['stratify_label'].apply(lambda x: 'Rare' if x in rare_labels else x)
            
            # Split the temp set into validation and test sets
            val_images, test_images = train_test_split(temp_images, test_size=0.5, stratify=temp_images['stratify_label'], random_state=0)
            
            # Convert image IDs to lists
            train_ids = train_images['image_id'].tolist()
            val_ids = val_images['image_id'].tolist()
            test_ids = test_images['image_id'].tolist()

            # Assign split values to annotations
            self.annotations.loc[sum([imid_indexes[im] for im in train_ids], []), [f'group_{i}', 'split']] = 'train'  
            self.annotations.loc[sum([imid_indexes[im] for im in val_ids], []), [f'group_{i}', 'split']] = 'val'
            self.annotations.loc[sum([imid_indexes[im] for im in test_ids], []), [f'group_{i}', 'split']] = 'test'
        
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
