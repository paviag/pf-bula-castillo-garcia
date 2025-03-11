import optuna
import os
import subprocess
import pandas as pd
import numpy as np

# Gets index of next train iteration
global train_index 
train_index = max(
    list(map(
        lambda x: int(x[5:]) if len(x) > 5 else 0,
        list(filter(lambda x: x.startswith("train"), os.listdir("runs/detect"))),
    ))
) + 1

def objective(trial):
    global train_index

    # Gets initial/suggested values for parameters
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-1, log=True)
    lrf = trial.suggest_float("lrf", 0.01, 1.0)
    momentum = trial.suggest_float("momentum", 0.6, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.001)
    warmup_epochs = trial.suggest_float("warmup_epochs", 0.0, 5.0)
    warmup_momentum = trial.suggest_float("warmup_momentum", 0.0, 0.95)
    box = trial.suggest_float("box", 0.02, 0.2)
    cls = trial.suggest_float("cls", 0.2, 4.0)
    batch = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    
    # Runs train command for YOLO model with generated parameters
    train_command = [
        "yolo", "train", 
        "data=pf-bula-castillo-garcia/data.yaml",
        "optimizer=AdamW",
        f"lr0={lr0}",
        f"lrf={lrf}",
        f"momentum={momentum}",
        f"weight_decay={weight_decay}",
        f"warmup_epochs={warmup_epochs}",
        f"warmup_momentum={warmup_momentum}",
        f"box={box}",
        f"cls={cls}",
        f"batch={batch}",
        "model=yolov8n.pt",
        "epochs=10", 
        "imgsz=640",
        "device=cuda:0",
        "workers=1",
    ]

    subprocess.run(train_command)
    
    # Obtains metrics of current run 
    results = pd.read_csv(f"runs/detect/train{train_index}/results.csv")
    p, r, mAP50, mAP50_95 = results[
        ['metrics/precision(B)', 
         'metrics/recall(B)', 
         'metrics/mAP50(B)', 
         'metrics/mAP50-95(B)']
         ].iloc[-1].values
    
    # Increases train_index for next iteration
    train_index += 1
    
    # Returns metrics
    return p, r, mAP50, mAP50_95

# Creates optuna study
study = optuna.create_study(directions=['maximize', 'maximize', 'maximize', 'maximize'])
study.optimize(objective, n_trials=50)

# DataFrame for best trials in study
best_trials = pd.DataFrame(columns=[
    "number", 
    "learning_rate", 
    "batch_size",
    "weight_decay",
    "precision",
    "recall",
    "mAP50", 
    "mAP50_95",
])
for t in study.best_trials:
    row = [t.number]
    row.extend(t.params.values())
    row.extend(t.values)

    row = {k: v for k, v in zip(best_trials.columns, row)}
    best_trials.loc[len(best_trials)] = row

# Saves as csv
best_trials.to_csv("pf-bula-castillo-garcia/optuna_best_trials.csv", index=False)
print(best_trials)