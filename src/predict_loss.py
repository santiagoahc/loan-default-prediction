import pandas as pd
from pathlib import Path


class PredictLoss:
    def __init__(self, data_path):
        self.data_path: Path = data_path

    def main(self):
        self._load_data()

    def _load_data(self):
        train_data = pd.read_csv(self.data_path / "train_v2.csv.zip")
        test_data = pd.read_csv(self.data_path / "test_v2.csv.zip")
