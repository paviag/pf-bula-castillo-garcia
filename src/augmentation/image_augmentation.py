import albumentations as A
import cv2

class ImageAugmentor:
    """Handles image augmentation"""
    def __init__(self, num_augmentations=3):
        self.num_augmentations = num_augmentations
        self.transform = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def augment_image(self, image_path, class_labels, bboxes):
        """Returns transformed image and labels from input paths and transform function"""
        image = cv2.imread(image_path)
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return transformed