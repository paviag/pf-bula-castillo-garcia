import pandas as pd
from ultralytics import YOLO

if __name__ == '__main__':
    model_path = "runs/detect/train6/weights/best.pt"

    model = YOLO(model_path) # YOLO working model created from best weights

    results = model.val(plots=False)#save_to_json=True)  # This runs validation and returns metrics
    key_metrics = ['precision', 'recall', 'mAP50', 'mAP50-95']
    series = pd.Series({k: results.results_dict[f"metrics/{k}(B)"] for k in key_metrics})
    #series.to_csv(config.output)


##
#model_path = f"{runs_path}/train{train_index}/weights/best.pt"
#model = YOLO(model_path) # YOLO working model created from best weights