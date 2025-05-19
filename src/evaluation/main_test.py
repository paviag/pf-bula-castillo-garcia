import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from ultralytics import YOLO
from config import config
import torch
import gc


def _get_ypred(model, test_images, batch_size=16):
    """
    Returns predicted values of test data.
    """
    ypred = []
    
    for i in range(0, len(test_images), batch_size):
        batch = test_images[i:i+batch_size]
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Process batch
        results = model.predict(
            batch,
            save=False,
            verbose=False,
            conf=0.45,
            stream=True
        )
        
        for r in results:
            if len(r.boxes.cls.unique()) > 0:
                ypred.append(0)  # Object detected
            else:
                ypred.append(1)  # No object detected
    
    return np.array(ypred)

def _get_ytrue(labels):
    """
    Returns true values of test data.
    """
    ytrue = []
    for label_file in labels:
        with open(label_file, "r") as f:
            lines = f.readlines()
            if lines:
                ytrue.append(0)
            else:
                ytrue.append(1)
                
    return np.array(ytrue)

def _get_box_metrics(model):
    """
    Returns box validation metrics as dict with keys 'mAP50' and 'mAP50-95'
    """
    results = model.val(data="data.yaml", split="test", plots=False)

    # Extract mAP metrics
    return {k: results.results_dict[f"metrics/{k}(B)"] for k in ['mAP50', 'mAP50-95']}

def _generate_classification_report(ytrue, ypred, class_labels, box_metrics, save_path):
    """
    Saves fig of classification report.
    """
    # Generate classification report
    report = classification_report(
        ytrue.astype(int), ypred.astype(int), output_dict=True)
    
    for k in box_metrics:
        report[class_labels[0]][k] = box_metrics[k]
        report[class_labels[1]][k] = box_metrics[k]

    # Plot classification report
    plt.figure(figsize=(8, 5))
    sns.heatmap(pd.DataFrame(
        report).iloc[:, :].T.drop(columns=["support"]), annot=True, cmap="Purples", linewidths=0.5, vmin=0, vmax=1)
    plt.title("Classification Report Heatmap")
    plt.xlabel("Metrics")
    plt.ylabel("Labels")

    # Save the image to output path
    plt.savefig(save_path)

def _generate_confusion_matrix(ytrue, ypred, class_labels, save_path):
    """
    Saves fig of confusion matrix.
    """

    plt.figure(figsize=(8, 5), facecolor='#00000000')
    # Generate the confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        ytrue, ypred, cmap='Purples'#, display_labels=class_labels
    )
    plt.title("Confusion Matrix")

    # Save the image to output path
    plt.savefig(save_path)

def run_validation(train_index):
    model_path = f"{config.runs_path}/train{train_index}/weights/best.pt"
    output_val_path = config.output_validation_path
    test_images_path = f"{config.yolo_dataset_path}/images/test"
    test_labels_path = f"{config.yolo_dataset_path}/labels/test"

    test_images = sorted(glob.glob(f"{test_images_path}/*.jpg"))
    test_labels = sorted(glob.glob(f"{test_labels_path}/*.txt"))

    model = YOLO(model_path)  # YOLO working model created from best weights

    box_metrics = _get_box_metrics(model)  # Obtains box metrics

    # Gets true values and predicted values of validation data
    ytrue = _get_ytrue(test_labels)
    ypred = _get_ypred(model, test_images)
    print(len(ypred), len(ytrue))
    # Class labels for plots
    class_labels = ["0", "1"]  

    # Confusion matrix
    _generate_confusion_matrix(
        ytrue,
        ypred,
        class_labels,
        f"{output_val_path}/confusion_matrix_run{train_index}",
    )
    # Classification report
    _generate_classification_report(
        ytrue,
        ypred,
        class_labels,
        box_metrics,
        f"{output_val_path}/classification_report_run{train_index}",
    )