import argparse
from preprocessing.main import run_preprocessing
from augmentation.main import run_augmentations
from model.model_setup import run_model_setup
from tuning.main import run_tuning
from model.model import run_model
# from evaluation.main import run_evaluation


def main(run_tuning=False, omit_setup=True):
    if not omit_setup:
        # Setup
        run_preprocessing()
        run_augmentations()
        run_model_setup()
    if run_tuning:
        best_trials_path = run_tuning()    # Tuning
        run_model(best_trials_path=best_trials_path)  # Model
    run_model()  # Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline")
    parser.add_argument("--tuning", action="store_true",
                        help="Enable hyperparameter tuning step")
    parser.add_argument("--omit_setup", action="store_true",
                        help="Omit data setup steps")
    args = parser.parse_args()

    main(run_tuning=args.tuning, omit_setup=args.omit_setup)
