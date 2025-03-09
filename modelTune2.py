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
        list(filter(lambda x: x.startswith("train"), reversed(os.listdir("runs/detect")))),
    ))
) + 1


def maximize_balanced(model_iterations):
    # Find the maximum value for each metric across all model iterations
    max_values = [max(model_iterations[j][i] for j in range(len(model_iterations))) for i in range(4)]
    
    # For each model iteration, calculate the "balance score"
    best_model = None
    best_balance_score = float('-inf')
    for mi in model_iterations:
        # Calculate the sum of differences from the max values to determine how close the list of metrics is
        differences = [abs(i - v) for i, v in zip(mi, max_values)]
        balance_score = -sum(differences)  # Negate the sum of differences (higher = better balance)
        
        if balance_score > best_balance_score:
            best_balance_score = balance_score
            best_model = mi
    
    return best_model

def objective(trial):
    global train_index

    # Gets initial/suggested values for parameters
    lr0 = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Runs train command for YOLO model with generated parameters
    train_command = [
        "yolo", "train", 
        "data=pf-bula-castillo-garcia/data.yaml",
        "optimizer=AdamW",
        f"lr0={lr0}",
        #f"patience={patience}",
        f"batch={batch_size}",
        f"weight_decay={weight_decay}",
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