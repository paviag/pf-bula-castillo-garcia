import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

def generate_classification_report(ytrue, ypred, class_labels, box_metrics, save_path):
    """
    Saves figure of classification report.
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
    plt.close()

def generate_confusion_matrix(ytrue, ypred, save_path):
    """
    Saves figure of confusion matrix.
    """
    plt.figure(figsize=(8, 5))
    
    # Generate the confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        ytrue, ypred, cmap='Purples'
    )
    plt.title("Confusion Matrix")

    # Save the image to output path
    plt.savefig(save_path)
    plt.close()