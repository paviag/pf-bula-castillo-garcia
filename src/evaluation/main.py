import pandas as pd
from ultralytics import YOLO
from ultralytics import YOLO
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import numpy as np
import matplotlib.pyplot as plt
from config import config
import seaborn as sns


def _get_ytrue_ypred(model, test_images_path):
    """
    Saves classification report of confusion matrix image of model from train_index.

    The matrix is built from predictions on model's pre-established
    validation data.
    """
    neg_results = model.predict(
        f"{test_images_path}/val_neg")  # Predictions for neg cases
    # Predictions for pos cases
    pos_results = model.predict(f"{test_images_path}/val")

    ytrue = np.concatenate([    # Array for true class values
        np.ones(len(neg_results)),  # neg cases should be 1
        np.zeros(len(pos_results)),  # pos cases should be 0
    ])
    # If there is more than one kind of predicted classes for a case, that case is pos
    # If there are none, that case is neg
    # This happens because the model is only aware of one class: pos (1)
    ypred = np.array([          # Array for predicted classes
        0 if len(r.boxes.cls.unique()) > 0 else 1
        for r in neg_results+pos_results
    ])

    return ytrue, ypred


def _generate_confusion_matrix(ytrue, ypred, class_labels, save_path):
    """
    Saves fig of confusion matrix.
    """

    plt.figure(figsize=(8, 5), facecolor='#00000000')
    # Generate the confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        ytrue, ypred, cmap='Blues', display_labels=class_labels
    )
    plt.title("Confusion Matrix")

    # Save the image to output path
    plt.savefig(save_path)


def _get_box_metrics(model):
    """
    Returns box validation metrics as dict with keys 'mAP50' and 'mAP50-95'
    """
    results = model.val(plots=False, data='data.yaml')  # Runs validation and returns metrics
    # Obtains box metrics
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
        report[class_labels[1]][k] = 1
        
    # Plot classification report
    plt.figure(figsize=(8, 5), facecolor='#00000000')
    sns.heatmap(pd.DataFrame(
        report).iloc[:, :].T, annot=True, cmap="Blues", linewidths=0.5, vmin=0, vmax=1)
    plt.title("Classification Report Heatmap")
    plt.xlabel("Metrics")
    plt.ylabel("Labels")

    # Save the image to output path
    plt.savefig(save_path)


def run_validation(train_index):
    # if __name__ == '__main__':
    model_path = f"{config.runs_path}/train{train_index}/weights/best.pt"
    output_val_path = config.output_validation_path
    test_images_path = f"{config.yolo_dataset_path}/images"

    model = YOLO(model_path)  # YOLO working model created from best weights

    box_metrics = _get_box_metrics(model)  # Obtains box metrics

    # Gets true values and predicted values of validation data
    ytrue, ypred = _get_ytrue_ypred(model, test_images_path)
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