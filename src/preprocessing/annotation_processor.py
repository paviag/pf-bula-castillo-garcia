import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class ProcessedAnnotation:
    def __init__(self, annotations, test_size=0.2, target_size=(640, 640)):
        self.annotations = pd.read_csv(annotations)
        self.test_size = test_size
        self.target_width, self.target_height = target_size

        self.fix_invalid_boxes()
        self.reproportion_class_distribution()
        self.set_splits()
        self.scale_bounding_boxes()

    def reproportion_class_distribution(self, neg_size=0.15):
        """Returns reproportioned annotations with adjusted positive/negative distribution"""
        # Filtering positive/negative cases
        pos = self.annotations[self.annotations.finding_categories.apply(lambda x: any(
            c in x for c in ["Mass", "Suspicious Calcification", "Suspicious Lymph Node"]))]
        pos.dropna(subset=['finding_birads'])
        neg = self.annotations[(self.annotations['finding_birads'].isna()) & (
            ~self.annotations['image_id'].isin(pos.image_id))]

        # Adjusting to compensate for class imbalance to include input neg_size
        neg = neg.sample(n=int(len(pos) / (1-neg_size)
                         * neg_size), random_state=1)
        pos['finding_categories'] = 0
        neg['finding_categories'] = 1

        self.annotations = pd.concat([pos, neg], ignore_index=True)

    def set_splits(self):
        """Returns annotations with set splits for stratified train/test split"""
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=0)
        # Stores annotations with no boxes for stratified split indices
        annot_no_boxes = self.annotations.drop(
            columns=['xmin', 'ymin', 'xmax', 'ymax'])
        # Stores indexes in annotations for each image_id
        imid_indexes = {im_id: self.annotations[self.annotations.image_id == im_id].index.tolist()
                        for im_id in self.annotations.image_id.unique()}
        annot_no_boxes = annot_no_boxes.drop_duplicates(
            subset=['image_id'])    # Keep only one row per image

        # Split images into train/test and set split for all rows belonging to that
        # image in annotations
        for train_idx, test_idx in sss.split(annot_no_boxes.drop(columns=['finding_categories']),
                                             annot_no_boxes.finding_categories):
            train_ids = annot_no_boxes.iloc[train_idx].image_id
            test_ids = annot_no_boxes.iloc[test_idx].image_id

            self.annotations.loc[sum([imid_indexes[im]
                                      for im in train_ids], []), 'split'] = 'train'
            self.annotations.loc[sum([imid_indexes[im]
                                      for im in test_ids], []), 'split'] = 'val'
        """ OLD CODE IN CASE IT BREAKS
        # Stores annotations with no boxes for stratified split indices
        annot_no_boxes = annotations.drop(columns=['xmin', 'ymin', 'xmax', 'ymax'])
        # Stores indexes in annotations for each image_id
        imid_indexes = pd.Series()
        for im_id in annotations.image_id.unique():
            imid_indexes.loc[im_id] = annotations[annotations.image_id == im_id].index.tolist()
            annot_no_boxes.drop(  # Keep only one row per image
                index=imid_indexes[im_id][1:], 
                inplace=True)  
        
        # Split images into train/test and set split for all rows belonging to that
        # image in annotations
        for train_index, test_index in sss.split(annot_no_boxes.drop(columns=['finding_categories']),
                                                 annot_no_boxes.finding_categories):
            im_id = annot_no_boxes.iloc[train_index].image_id
            annotations.loc[
                [i for j in imid_indexes[im_id].values.tolist() for i in j], 
                'split'] = 'train'
            im_id = annot_no_boxes.iloc[test_index].image_id
            annotations.loc[
                [i for j in imid_indexes[im_id].values.tolist() for i in j], 
                'split'] = 'val'"""

    def scale_bounding_boxes(self):
        """Scales annotations' bounding boxes to target dimensions"""
        for col in ['xmin', 'xmax']:
            self.annotations[col] *= self.target_width / self.annotations.width
        for col in ['ymin', 'ymax']:
            self.annotations[col] *= self.target_height / \
                self.annotations.height

    def fix_invalid_boxes(self):
        """Clips box coordinates between 0 and width (x-coordinates) or height (y-coordinates)"""
        self.annotations[['xmin', 'xmax']] = self.annotations[['xmin', 'xmax']].apply(
            lambda x: x.clip(upper=self.annotations['width'], lower=0), axis=0)
        self.annotations[['ymin', 'ymax']] = self.annotations[['ymin', 'ymax']].apply(
            lambda x: x.clip(upper=self.annotations['height'], lower=0), axis=0)
