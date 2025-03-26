import os
import shutil
import yaml
import pandas as pd
from config import config


class YOLOLabelGenerator:
    """Generates YOLO labels from annotations"""

    def __init__(self, annotations, output_dir):
        self.annotations = annotations
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_labels(self):
        for im_id, rows in self.annotations.groupby('image_id'):
            first = rows.head(1)
            yolo_label_path = os.path.join(self.output_dir, f"{im_id}.txt")
            labels = "\n".join(
                [f"{row.finding_categories} {row.bx} {row.by} {row.bw} {row.bh}" for _,
                    row in rows.iterrows()]
            ) if first.finding_categories.item() == 0 else ""
            with open(yolo_label_path, "w") as f:
                f.write(labels)


class DatasetOrganizer:
    """Organizes images and labels in folders structured for YOLO models"""

    def __init__(self, annotations_file, base_yolo_path, label_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.base_yolo_path = base_yolo_path
        self.label_dir = label_dir
        os.makedirs(self.base_yolo_path, exist_ok=True)

    def move_files(self):
        for im_id, rows in self.annotations.groupby('image_id'):
            first = rows.head(1)
            split_folder = self._determine_split_folder(first)
            self._move_image(first, im_id, split_folder)
            self._move_label(im_id, split_folder)

    def _determine_split_folder(self, first):
        if first.split.item() == "train":
            return "train"
        return "val" if first.finding_categories.item() == 0 else "val_neg"

    def _move_image(self, first, im_id, split_folder):
        img_dest_path = os.path.join(
            self.base_yolo_path, "images", split_folder, f"{im_id}.jpg")
        os.makedirs(os.path.dirname(img_dest_path), exist_ok=True)
        shutil.copy(first.directory_path.item(), img_dest_path)

    def _move_label(self, im_id, split_folder):
        label_src_path = os.path.join(self.label_dir, f"{im_id}.txt")
        label_dest_path = os.path.join(
            self.base_yolo_path, "labels", split_folder, f"{im_id}.txt")
        os.makedirs(os.path.dirname(label_dest_path), exist_ok=True)
        shutil.copy(label_src_path, label_dest_path)


class YOLOConfigGenerator:
    """Generates YAML config file for YOLO model"""

    def __init__(self, yaml_path, train_path, val_path, num_classes, class_names):
        self.yaml_path = yaml_path
        self.config = {
            "train": train_path,
            "val": val_path,
            "nc": num_classes,
            "names": class_names,
        }

    def save_config(self):
        with open(self.yaml_path, "w") as file:
            yaml.dump(self.config, file, default_flow_style=False)
        print(f"Archivo YAML guardado en: {self.yaml_path}")


class FolderValidator:
    """Checks that expected folders exist"""

    def __init__(self, dataset_path, expected_folders):
        self.dataset_path = dataset_path
        self.expected_folders = expected_folders

    def validate_folders(self):
        for folder in self.expected_folders:
            full_path = os.path.join(self.dataset_path, folder)
            print(
                f"{full_path}: {'✅ Exists' if os.path.exists(full_path) else '❌ Does not exist'}")


def run_model_setup():
    annotations_file = f"{config.output_data_path}/annotations.csv"
    yolo_labels_path = config.yolo_labels_path
    yolo_dataset_path = config.yolo_dataset_path
    yaml_path = config.yolo_config_path
    expected_folders = ["images/train", "images/val",
                        "images/val_neg", "labels/train", "labels/val"]

    # Generate labels
    yolo_label_gen = YOLOLabelGenerator(annotations_file, yolo_labels_path)
    yolo_label_gen.generate_labels()
    # Organize images and labels
    dataset_organizer = DatasetOrganizer(
        annotations_file, yolo_dataset_path, yolo_labels_path)
    dataset_organizer.move_files()
    # Make YOLO model config file
    yolo_config = YOLOConfigGenerator(
        yaml_path, yolo_dataset_path + "images/train", yolo_dataset_path + "images/val", 1, ["0"])
    yolo_config.save_config()
    # Check expected folders were created
    folder_validator = FolderValidator(yolo_dataset_path, expected_folders)
    folder_validator.validate_folders()
