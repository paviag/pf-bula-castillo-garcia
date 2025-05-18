import argparse
from preprocessing.main import run_preprocessing
from augmentation.main import run_augmentations
from model.model_setup import run_model_setup
from tuning.main import run_tuning
from tuning.train_index import TrainIndexManager
from model.model import run_model
from evaluation.main_test import run_validation


def main(run_tuning_sp=False, omit_setup=True):
    """Main function to run the pipeline"""
    NUM_GROUPS = 3
    if not omit_setup:
        # Setup
        print("Running preprocessing...")
        run_preprocessing(NUM_GROUPS)
        print("Running augmentations...")
        run_augmentations()
        print("Running model setup...")
        run_model_setup(NUM_GROUPS)
    if run_tuning_sp:
        print("Running tuning...")
        run_tuning()    # Tuning
    tim = TrainIndexManager()
    train_indices = []
    # Run model training and evaluation for each group
    for i in range(NUM_GROUPS):
        train_indices.append(228)#tim.get_index())
        print(f"Running model training and evaluation for group {i} (index {train_indices[-1]})...")
        # Run model training and evaluation for each group
        run_model(group=i)
        print("Running model evaluation...")
        run_validation(train_indices[-1])  # Evaluation
    # Run ensemble validation
    print("Running ensemble validation...")
    run_validation(train_indices)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline")
    parser.add_argument("--tuning", action="store_true",
                        help="Enable hyperparameter tuning step")
    parser.add_argument("--omit_setup", action="store_true",
                        help="Omit data setup steps")
    args = parser.parse_args()

    main(run_tuning_sp=args.tuning, omit_setup=args.omit_setup)
