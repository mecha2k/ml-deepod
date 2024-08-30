import torch
import numpy as np
import pandas as pd
import pickle
import warnings

from pathlib import Path
from deepod.models.time_series import AnomalyTransformer, TranAD, TimesNet


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"{device} is available in torch")


def load_datasets(data_path, mode="open"):
    if mode == "open":
        train_df = pickle.load(open(data_path / "train.pkl", "rb"))
        test_df = pickle.load(open(data_path / "test.pkl", "rb"))
        print(train_df.head())
        X_train = train_df[:10000].values
        X_test = test_df[:10000].values
        y_test = np.zeros_like(X_test.shape[0])
    elif mode == "swat":
        X_train = np.load(data_path / "train.npy")
        X_test = np.load(data_path / "test.npy")
        y_test = np.load(data_path / "test_labels.npy")
    return X_train, X_test, y_test


if __name__ == "__main__":
    mode = "open"
    data_path = Path("datasets/open")
    # mode = "SMD"
    # data_path = Path("datasets/SMD")
    # mode = "swat"
    # data_path = Path("datasets/SWaT")

    X_train, X_test, y_test = load_datasets(data_path, mode=mode)
    print(X_train.shape, X_test.shape, y_test.shape)

    clf = AnomalyTransformer(device=device)
    clf.fit(X_train)

    scores = clf.decision_function(X_test)
    print(scores.shape)
    print(scores[:50])
    # scores = pd.DataFrame(scores)
    # scores.to_pickle(data_path / "test_scores.pkl")

    # scores = pd.read_pickle(data_path / "test_scores.pkl")
    # print(scores.describe())
    # print(scores.info())
    # print(scores.head())
    # print(scores.shape)

    # # evaluation of time series anomaly detection
    # from deepod.metrics import ts_metrics
    # from deepod.metrics import (
    #     point_adjustment,
    # )  # execute point adjustment for time series ad
    #
    # scores = pd.read_pickle(data_path / "test_scores.pkl")
    # print(scores.describe())
    # scores = scores.values.flatten()
    # print(scores.shape)
    #
    # # eval_metrics = ts_metrics(y_test, scores)
    # adj_eval_metrics = ts_metrics(y_test, point_adjustment(y_test, scores))
    # print(eval_metrics)
    # print(adj_eval_metrics)
    # # results : roc_auc_score, average_precision_score, best_f1, best_p, best_r
    # # (0.76625, 0.26313, 0.30726, 0.42068, 0.24201)
