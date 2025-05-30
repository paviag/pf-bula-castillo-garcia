import argparse
from preprocessing.main import run_preprocessing
from augmentation.main import run_augmentations
from model.model_setup import run_model_setup
from tuning.main import run_tuning
from tuning.train_index import TrainIndexManager
from model.model import run_model
from evaluation.main_test import run_validation


def main(run_tuning_sp=False, omit_setup=True):
    if not omit_setup:
        # Setup
        print("Running preprocessing...")
        run_preprocessing()
        print("Running augmentations...")
        run_augmentations()
        print("Running model setup...")
        run_model_setup()
    if run_tuning_sp:
        print("Running tuning...")
        run_tuning()    # Tuning
    train_index = TrainIndexManager().index
    print("Running model training...")
    run_model()  # Model
    print("Running model evaluation...")
    run_validation(train_index)  # Evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline")
    parser.add_argument("--tuning", action="store_true",
                        help="Enable hyperparameter tuning step")
    parser.add_argument("--omit_setup", action="store_true",
                        help="Omit data setup steps")
    args = parser.parse_args()

    main(run_tuning_sp=args.tuning, omit_setup=args.omit_setup)
