import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class PredictLoss:
    def __init__(self, data_path: Path):
        self._data_path: Path = data_path
        self._train: pd.DataFrame = pd.DataFrame()
        self._test: pd.DataFrame = pd.DataFrame()
        self.x_train: Optional[np.array] = None
        self.x_test: Optional[np.array] = None
        self.y_train: Optional[np.array] = None
        self.y_test: Optional[np.array] = None
        self.model_name: Path = self._data_path / "xgboost_regressor.pkl"

    def main(self):
        self._load_data()
        self._basic_preprocessing()
        self._data_preparation()
        self._training()
        self._predict()

    def _training(self):
        if self._load_model():
            return
        self.model = XGBRegressor(n_jobs=12)
        mean_score = cross_val_score(
            estimator=self.model, X=self.x_train, y=self.y_train, cv=5
        ).mean()
        print("Mean score from Cross Validation:", mean_score)
        print("\nTraining started...\n")
        self.model.fit(self.x_train, self.y_train)
        print("Saving model")
        pickle.dump(self.model, open(self.model_name, "wb"))

    def _predict(self):
        test_predictions = self.model.predict(self.x_test)
        submission_df = pd.DataFrame(test_predictions, columns=["loss"])
        submission_df.insert(loc=0, column="id", value=self._test["id"])
        submission_df.to_csv(self._data_path / "submission.csv", index=False)

    def _load_model(self):
        if not self.model_name.is_file():
            return
        print("Loading model")
        self.model = pickle.load(open(self.model_name, "rb"))
        return True

    def _data_preparation(self):
        sc = StandardScaler()

        import ipdb

        ipdb.set_trace()

        self.x_train = sc.fit_transform(self._train.drop(columns=["id", "loss"]))
        self.x_test = sc.transform(self._test.drop(columns="id"))
        self.y_train = np.array(self._train["loss"]).reshape((-1,))

    def _load_data(self):
        self._train = pd.read_csv(self._data_path / "train_v2.csv.zip")
        self._test = pd.read_csv(self._data_path / "test_v2.csv.zip")

    def _basic_preprocessing(self, visualize_hist: bool = False):
        self._calculate_null_values(self._train)
        print("Filling values with its mean")
        object_cols = self._train.select_dtypes(include=["object"]).columns.tolist()
        self._train.drop(columns=object_cols, inplace=True)
        self._test.drop(columns=object_cols, inplace=True)
        self._train.fillna(self._train.mean(), inplace=True)
        self._test.fillna(self._test.mean(), inplace=True)
        self._calculate_null_values(self._train)
        _ = plt.hist(self._train["loss"], bins=100, edgecolor="k", log=True)
        if visualize_hist:
            plt.show()
        self._remove_collinear_features(0.7)

    def _remove_collinear_features(self, threshold: float):
        """
        Objective:
            Remove collinear features in a dataframe with a correlation coefficient
            greater than the threshold. Removing collinear features can help a model
            to generalize and improves the interpretability of the model.

        Inputs:
            threshold: any features with correlations greater than this value are removed

        Output:
            dataframe that contains only the non-highly-collinear features
        """
        print("Number of columns:", len(self._train.columns))
        # Dont want to remove correlations between loss
        y = self._train["loss"]
        self._train = self._train.drop(columns=["loss"])

        # Calculate the correlation matrix
        corr_matrix = self._train.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        # Iterate through the correlation matrix and compare correlations
        for i in iters:
            for j in range(i):
                item = corr_matrix.iloc[j : (j + 1), (i + 1) : (i + 2)]
                col = item.columns
                val = abs(item.values)
                # If correlation exceeds the threshold
                if val >= threshold:
                    drop_cols.append(col.values[0])

        # Drop one of each pair of correlated columns
        drops = set(drop_cols)
        self._train = self._train.drop(columns=drops)
        self._test = self._test.drop(columns=drops)

        # Add the score back in to the data
        self._train["loss"] = y
        print(
            "Number of columns after removing collinear features:",
            len(self._train.columns),
        )

    def _calculate_null_values(self, df: pd.DataFrame, ratio: float = 0.1):
        print(
            "Columns with more than 10% of null values: ",
            len(df.isnull().sum()[df.isnull().sum() / len(df) > ratio]),
        )
        print(
            "Columns with more than 5% of null values: ",
            len(df.isnull().sum()[df.isnull().sum() / len(df) > ratio / 2]),
        )


if __name__ == "__main__":
    # Data Path
    data = Path("data")

    prediction = PredictLoss(data)
    prediction.main()
