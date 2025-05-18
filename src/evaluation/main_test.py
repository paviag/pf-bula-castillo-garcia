import os
import glob
import numpy as np
from ultralytics import YOLO
from config import config
from evaluation.ensemble import ensemble_predict_wbf
from evaluation.visualization import generate_confusion_matrix, generate_classification_report
from evaluation.metric_utils import calculate_map_from_boxes, xywh_to_xyxy


def _get_ypred(model, image_files):
    """
    Returns predicted values of test data for a single model.
    """
    ypred = []
    for r in model.predict(image_files, verbose=False, save=False, stream=True, device='cpu', imgsz=640):
        if len(r.boxes.cls.unique()) > 0:
            ypred.append(0)
        else:
            ypred.append(1)
    return np.array(ypred)

def _get_ytrue(label_files):
    """
    Returns true class labels and boxes from test data.
    Boxes are returned in the xyxy format.
    """
    ytrue = []
    boxes = []
    for label_file in label_files:
        with open(label_file, "r") as f:
            lines = f.readlines()
            if lines:
                ytrue.append(0)
                boxes.append([np.array(list(map(float, line.split()[1:]))) for line in lines])
            else:
                ytrue.append(1)
                boxes.append(np.array([]))
    # Convert boxes to xyxy format
    return np.array(ytrue), xywh_to_xyxy(boxes)   

def _get_box_metrics(model, idx):
    """
    Returns box validation metrics from single model val 
    as dict with keys 'mAP50' and 'mAP50-95'
    """
    results = model.val(data=f"data_{idx}.yaml", split="test", plots=False)

    # Extract mAP metrics
    return {k: results.results_dict[f"metrics/{k}(B)"] for k in ['mAP50', 'mAP50-95']}

def run_validation(train_index):
    output_val_path = config.output_validation_path
    test_images_path = f"{config.yolo_dataset_path}/images/test"
    test_labels_path = f"{config.yolo_dataset_path}/labels/test"

    # Sort image and label files to ensure they are in the same order
    image_files = sorted(glob.glob(f"{test_images_path}/*.jpg"))
    label_files = sorted(glob.glob(f"{test_labels_path}/*.txt"))
    
    # Gets true values of test data
    ytrue, true_boxes = _get_ytrue(label_files)
    
    if isinstance(train_index, list):
        # Load models
        model_paths = [f"{config.runs_path}/train{idx}/weights/best.pt" for idx in train_index]
        models = [YOLO(model_path) for model_path in model_paths]
        # Get ensemble predictions 
        ypred, pred_boxes, conf_scores = ensemble_predict_wbf(models, image_files)
        
        # Get box metrics
        box_metrics = calculate_map_from_boxes(pred_boxes, true_boxes, conf_scores)
        
        # Join train indexes for saving
        train_index = "_".join([str(idx) for idx in train_index])    
    else:
        # Load single model
        model_path = f"{config.runs_path}/train{train_index}/weights/best.pt"
        model = YOLO(model_path)
        # Get predictions
        ypred = _get_ypred(model, image_files)
        print("ypred", len(ypred))
        # Get box metrics
        box_metrics = _get_box_metrics(model)

    # Class labels for plots
    class_labels = ["0", "1"]  

    print(len(ypred), len(ytrue), len(true_boxes))
    # Confusion matrix
    generate_confusion_matrix(
        ytrue,
        ypred,
        f"{output_val_path}/confusion_matrix_run{train_index}",
    )
    # Classification report
    generate_classification_report(
        ytrue,
        ypred,
        class_labels,
        box_metrics,
        f"{output_val_path}/classification_report_run{train_index}",
    )