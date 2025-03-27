from ultralytics import YOLO
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from config import config


def confusion_matrix(model, test_images_path, save_path):
    """
    Saves and displays fig of confusion matrix image of model from train_index.

    The matrix is built from predictions on model's pre-established
    validation data.
    """
    neg_results = model.predict(f"{test_images_path}/val_neg")  # Predictions for neg cases
    pos_results = model.predict(f"{test_images_path}/val")      # Predictions for pos cases

    ytrue = np.concatenate([    # Array for true class values
        np.ones(len(neg_results)),  # neg cases should be 1
        np.zeros(len(pos_results)), # pos cases should be 0
    ])
    # If there is more than one kind of predicted classes for a case, that case is pos
    # If there are none, that case is neg
    # This happens because the model is only aware of one class: pos (1)
    ypred = np.array([          # Array for predicted classes
        0 if len(r.boxes.cls.unique()) > 0 else 1 
        for r in neg_results+pos_results
        ])

    # Generate the confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        ytrue, ypred, cmap='Blues', display_labels=["positive (0)", "negative (1)"]
    )

    # Save the image to output path
    plt.savefig(save_path)

    # Display
    plt.show()
