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
        
        # Reassign finding categories
        self.annotations.loc[pos.index, 'finding_categories'] = 0   # Positive 
        self.annotations.loc[neg.index, 'finding_categories'] = 1   # Negative 

        # Initialize group columns
        self.annotations[[f'group_{i}' for i in range(num_groups)]] = 0

        # Relevant columns for stratification
        relevant_columns = ['laterality', 'view_position', 'breast_density']

        # Get unique negative images
        unique_neg_images = neg.drop(columns=['xmin', 'ymin', 'xmax', 'ymax']).drop_duplicates(subset=['image_id'])
        unique_neg_images = self._add_stratify_label(unique_neg_images, relevant_columns)
        unique_neg_images = unique_neg_images.drop(columns=relevant_columns)
        unique_neg_images = self._fix_rare_labels(unique_neg_images, 2)

        # Get unique pos images
        unique_pos_images = pos.drop(columns=['xmin', 'ymin', 'xmax', 'ymax']).drop_duplicates(subset=['image_id'])
        unique_pos_images = self._add_stratify_label(unique_pos_images, relevant_columns)
        unique_pos_images = unique_pos_images.drop(columns=relevant_columns)
        unique_pos_images = self._fix_rare_labels(unique_pos_images, 2)

        # Get a random subsample ensuring 80 pos/20 neg split for each group 
        # and 15% relative test data
        test_pos_size = int(len(unique_pos_images) * 0.15)
        group_pos_size = len(unique_pos_images) - test_pos_size
        group_neg_size = int(group_pos_size / 0.85 * 0.15)
        test_neg_size = int(group_neg_size / 0.85 * 0.15)
        test_pos_size = int(group_pos_size / 0.85 * 0.15)

        # Split pos test images from pos group images (final pos_test_images)
        pos_group_images, pos_test_images = train_test_split(unique_pos_images, 
                                                            test_size=test_pos_size, 
                                                            stratify=unique_pos_images['stratify_label'],
                                                            random_state=0)
        
        # Extract necessary neg images to obtain 80% pos/20% neg 
        _, unique_neg_images = train_test_split(unique_neg_images, 
                                                test_size=(num_groups * group_neg_size) + test_neg_size, 
                                                stratify=unique_neg_images['stratify_label'],
                                                random_state=0)
        unique_neg_images = self._fix_rare_labels(unique_neg_images, 2)
        
        # Filter negatives to include only sampled images
        neg = neg[neg.image_id.isin(unique_neg_images.image_id.to_numpy())]
        
        # Split neg test images from neg group images (final neg_test_images)
        neg_group_images, neg_test_images = train_test_split(unique_neg_images, 
                                                            test_size=test_neg_size, 
                                                            stratify=unique_neg_images['stratify_label'],
                                                            random_state=0)
        neg_group_images = self._fix_rare_labels(neg_group_images, 2)
        
        # Split group images among groups
        neg_split_groups = []
        remaining_df = neg_group_images.copy()
        # Iteratively split the dataset while ensuring stratification
        for _ in range(num_groups - 1):
            split_df, remaining_df = train_test_split(
                remaining_df, test_size=1 - (1 / (num_groups - len(neg_split_groups))), 
                stratify=remaining_df['stratify_label'])
            remaining_df = self._fix_rare_labels(remaining_df, 2)
            neg_split_groups.append(split_df)
        neg_split_groups.append(remaining_df)  # Add the last remaining group

        # Assign negatives to groups
        for i, assigned_images in enumerate(neg_split_groups):
            assigned_image_ids = assigned_images.image_id.to_numpy()
            self.annotations.loc[self.annotations.image_id.isin(assigned_image_ids), f'group_{i}'] = 1
            self.annotations.loc[self.annotations.image_id.isin(assigned_image_ids), 'split'] = 'group'

        # Assign positives to groups
        assigned_image_ids = pos_group_images.image_id.to_numpy()
        self.annotations.loc[self.annotations.image_id.isin(assigned_image_ids), [f'group_{i}' for i in range(num_groups)]] = 1
        self.annotations.loc[self.annotations.image_id.isin(assigned_image_ids), 'split'] = 'group'

        # Assign negatives and positives to test
        self.annotations.loc[self.annotations.image_id.isin(neg_test_images.image_id.to_numpy()), 'split'] = 'test'
        self.annotations.loc[self.annotations.image_id.isin(pos_test_images.image_id.to_numpy()), 'split'] = 'test'

        # Filter annotations to include only assigned rows
        self.annotations = self.annotations.loc[pos.index.tolist() + neg.index.tolist()].reset_index(drop=True)

        return self.annotations

    def _fix_rare_labels(self, df, num):
        # Identify rare labels (those with fewer than num instances) and replace them with 'Rare'
        label_counts = df['stratify_label'].value_counts()
        rare_labels = label_counts[label_counts < num].index
        if len(rare_labels) < num:
            # Replace rare labels with the second most common label
            df.loc[df['stratify_label'].isin(rare_labels), 'stratify_label'] = label_counts.index[-num]
        else:
            # Replace rare labels with 'Rare'
            df['stratify_label'] = df['stratify_label'].apply(lambda x: 'Rare' if x in rare_labels else x)
        return df

    def _add_stratify_label(self, unique_images, relevant_columns):
        """Adds stratify label to df with relevant columns"""
        # Create a stratification label by combining all relevant columns
        unique_images['stratify_label'] = unique_images.apply(lambda row: '_'.join(map(str, row[relevant_columns])), axis=1)
        return unique_images

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
            annot_group = annot_group[annot_group['split'] == 'group']
            unique_images = annot_group.drop(columns=['xmin', 'ymin', 'xmax', 'ymax']).drop_duplicates(subset=['image_id'])
            unique_images = self._add_stratify_label(unique_images, relevant_columns)
            unique_images = self._fix_rare_labels(unique_images, 2)
           
            # Split the data into train and val sets (85% train, 15% val)
            train_images, val_images = train_test_split(unique_images, test_size=0.15, stratify=unique_images['stratify_label'], random_state=0)
            
            # Assign split values to annotations
            self.annotations.loc[sum([imid_indexes[im] for im in train_images['image_id'].tolist()], []), f'group_{i}'] = 'train'  
            self.annotations.loc[sum([imid_indexes[im] for im in val_images['image_id'].tolist()], []), f'group_{i}'] = 'val'
        
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
