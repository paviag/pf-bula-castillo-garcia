import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


class ProcessedAnnotation:
    def __init__(self, annotations, target_size=(640, 640)):
        self.annotations = pd.read_csv(annotations)
        self.target_width, self.target_height = target_size

        #self._reproportion_class_distribution()
        self._set_splits()
        self._fix_invalid_boxes()
        self._scale_bounding_boxes()
        self._add_yolo_label_cols()

    def _set_splits(self, train_frac=0.7, val_frac=0.15, pos_frac=0.6):
        """Returns reproportioned annotations with adjusted positive/negative distribution and set splits"""
        # Filtering positive/negative cases
        pos = self.annotations[self.annotations.finding_categories.apply(lambda x: any(
            c in x for c in ["Mass", "Suspicious Calcification", "Suspicious Lymph Node"]))]
        pos = pos.dropna(subset=['finding_birads', 'xmin'])
        neg = self.annotations[(self.annotations['finding_birads'].isna()) & (
            ~self.annotations['image_id'].isin(pos.image_id))]
        
        pos_unique = pos.drop_duplicates(subset=['image_id'])
        neg_unique = neg.drop_duplicates(subset=['image_id'])

        # Create mapping of image_id to annotation indices
        imid_indexes = {im_id: self.annotations[self.annotations.image_id == im_id].index.tolist()
                        for im_id in self.annotations.image_id.unique()}

        # Set findings to 0 for positive cases
        self.annotations.loc[sum([imid_indexes[im] for im in pos_unique.image_id], []), 'finding_categories'] = 0
        # Set findings to 1 for negative cases
        self.annotations.loc[sum([imid_indexes[im] for im in neg_unique.image_id], []), 'finding_categories'] = 1

        # Create a stratification label by combining all relevant columns
        relevant_columns = ['laterality', 'view_position', 'breast_density']
        pos_unique = self._add_stratify_label(pos_unique, relevant_columns)
        pos_unique = self._fix_rare_labels(pos_unique, 2)
        neg_unique = self._add_stratify_label(neg_unique, relevant_columns)
        neg_unique = self._fix_rare_labels(neg_unique, 2)

        # Define total known value
        total_pos = len(pos_unique)
        ratio_pos = train_frac*pos_frac + val_frac + (1-train_frac-val_frac)*pos_frac
        # Calculate total
        total_size = total_pos / ratio_pos

        # Training set
        n_train = int(total_size * train_frac)
        n_train_pos = int(n_train * pos_frac)
        n_train_neg = n_train - n_train_pos

        # Validation set
        n_val = int(total_size * val_frac)
        n_val_pos = n_val  # Given explicitly

        # Test set
        n_test = int(total_size * (1 - train_frac - val_frac))
        n_test_pos = int(n_test * pos_frac)
        n_test_neg = n_test - n_test_pos

        print(f"Total size: {total_size}, Train: {n_train}, Val: {n_val}, Test: {n_test}")
        print(f"Pos Train: {n_train_pos}, Pos Val: {n_val_pos}, Pos Test: {n_test_pos}")
        print(f"Neg Train: {n_train_neg}, Neg Test: {n_test_neg}")

        # Selecting neg needed
        neg_selected, _ = train_test_split(
            neg_unique, train_size=n_train_neg + n_test_neg,
            stratify=neg_unique['stratify_label'], random_state=0
        )
        neg_selected = self._fix_rare_labels(neg_selected, 2)

        # Splitting pos into train, val, test sets
        pos_trainval, pos_test = train_test_split(
            pos_unique,
            train_size=n_train_pos + n_val_pos,
            stratify=pos_unique['stratify_label'],
            random_state=0
        )
        pos_trainval = self._fix_rare_labels(pos_trainval, 2)

        pos_train, pos_val = train_test_split(
            pos_trainval,
            train_size=n_train_pos,
            stratify=pos_trainval['stratify_label'],
            random_state=0
        )

        # Splitting neg into train and test sets
        neg_train, neg_test = train_test_split(
            neg_selected,
            train_size=n_train_neg,
            stratify=neg_selected['stratify_label'],
            random_state=0
        )

        # Set all splits to NaN initially
        self.annotations.loc[:, 'split'] = pd.NA

        # Assigning splits to annotations
        self.annotations.loc[sum([imid_indexes[im] for im in pos_train.image_id], []), 'split'] = 'train'
        self.annotations.loc[sum([imid_indexes[im] for im in pos_val.image_id], []), 'split'] = 'val'
        self.annotations.loc[sum([imid_indexes[im] for im in pos_test.image_id], []), 'split'] = 'test'

        self.annotations.loc[sum([imid_indexes[im] for im in neg_train.image_id], []), 'split'] = 'train'
        self.annotations.loc[sum([imid_indexes[im] for im in neg_test.image_id], []), 'split'] = 'test'

        # Filter out unused annotations
        self.annotations = self.annotations[self.annotations['split'].notna()]
        # Reset index
        self.annotations.reset_index(drop=True, inplace=True)

    def _fix_rare_labels(self, df, num):
        # Identify rare labels (those with fewer than num instances) and replace them with 'Rare'
        label_counts = df['stratify_label'].value_counts()
        rare_labels = label_counts[label_counts < num].index
        if len(rare_labels) < num:
            # Replace rare labels with the second most common label
            df.loc[df['stratify_label'].isin(rare_labels), 'stratify_label'] = label_counts.index[-num]
        else:
            # Replace rare labels with 'Rare'
            df.loc[:, 'stratify_label'] = df['stratify_label'].apply(lambda x: 'Rare' if x in rare_labels else x)
        return df

    def _add_stratify_label(self, unique_images, relevant_columns):
        """Adds stratify label to df with relevant columns"""
        # Create a stratification label by combining all relevant columns
        unique_images.loc[:, 'stratify_label'] = unique_images.apply(lambda row: '_'.join(map(str, row[relevant_columns])), axis=1)
        return unique_images

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
