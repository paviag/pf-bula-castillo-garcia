from ultralytics import YOLO
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

model_path = "runs/detect/train121/weights/best.pt"
test_path = "C:/Users/Lab6k/Documents/PF/pf-bula-castillo-garcia/dataset/images"

model = YOLO(model_path)

#print(model.val(save_json=True))

neg_results = model.predict(test_path+"/val_neg")
pos_results = model.predict(test_path+"/val")
for r in pos_results:
    if len(r.boxes.cls.unique()) > 1:
        print(Exception(r.boxes.cls))

ytrue = np.concatenate([
    np.ones(len(neg_results)),
    np.zeros(len(pos_results)),
])
ypred = np.array([
    0 if len(r.boxes.cls.unique()) > 0 else 1 
    for r in neg_results+pos_results
    ])

ConfusionMatrixDisplay.from_predictions(ytrue, ypred, cmap='Blues', display_labels=["positive (0)", "negative (1)"])

plt.show()
#results = model("pf-bula-castillo-garcia/dataset/images/val/0af3e36d8d050e92149a6b79e4181db4.jpg")
#results.show()
