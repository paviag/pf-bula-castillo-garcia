from tuning.study_manager import OptunaStudyManager
from tuning.train_index import TrainIndexManager
from tuning.trainer import YOLOTrainer
from config import config


def run_tuning():
    runs_path = config.runs_path
    output_data_dir = config.output_data_path
    config_path = config.yolo_config_path

    train_index_manager = TrainIndexManager()
    trainer = YOLOTrainer(train_index_manager, runs_path=runs_path, # default 10 epochs per trial
                          config_path=config_path)
    optuna_manager = OptunaStudyManager(trainer)
    optuna_manager.run_optimization()   # default 50 trials
    
    best_trials_path = f"{output_data_dir}/optuna_best_trials.csv"
    optuna_manager.save_best_trials(best_trials_path)
    return best_trials_path
