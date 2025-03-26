import pandas as pd
import optuna


class OptunaStudyManager:
    """Manages the Optuna study and stores best trials"""

    def __init__(self, trainer):
        self.trainer = trainer
        self.study = optuna.create_study(directions=4*['maximize'])

    def run_optimization(self, n_trials=50):
        """Optimizes study for an input number of trials with trainer"""
        self.study.optimize(self.trainer.train, n_trials=n_trials)

    def save_best_trials(self, output_path):
        """Stores best trials as csv and prints them"""
        columns = ["number"] + self.trainer.params + \
            ["precision", "recall", "mAP50", "mAP50_95"]
        best_trials = pd.DataFrame(columns=columns)
        for t in self.study.best_trials:
            row = {k: v for k, v in zip(
                columns, [t.number] + list(t.params.values()) + list(t.values))}
            best_trials.loc[len(best_trials)] = row
        best_trials.to_csv(output_path, index=False)
        print(best_trials)
