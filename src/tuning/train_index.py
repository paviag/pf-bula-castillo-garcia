import os


class TrainIndexManager:
    """Manages the training index for YOLO runs"""

    def __init__(self, runs_base_path):
        # Sets starting index of Optuna study as index of next YOLO run
        self.index = self._get_next_index(runs_base_path)

    def _get_next_index(self, base_path):
        """Returns the index that the next YOLO run will have"""
        existing_runs = [int(x[5:]) for x in os.listdir(
            base_path) if x.startswith("train") and len(x) > 5]
        return max(existing_runs, default=0) + 1

    def get_index(self):
        """Gets index and increments for next run"""
        idx = self.index
        self.index += 1
        return idx
