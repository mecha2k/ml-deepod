import torch
import numpy as np
import pandas as pd
import pickle
import warnings

from pathlib import Path

from menuinst.utils import data_path

from deepod.models.time_series import AnomalyTransformer, TranAD, TimesNet
from testbed.testbed_unsupervised_ad import y_test

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"{device} is available in torch")


def load_datasets(data_path, mode="open"):
    if mode == "open":
        train_df = pickle.load(open(data_path / "train.pkl", "rb"))
        test_df = pickle.load(open(data_path / "test.pkl", "rb"))
        X_train = train_df.values
        X_test = test_df.values
        y_test = np.zeros_like(X_test.shape[0])
    elif mode == "swat":
        X_train = np.load(data_path / "train.npy")
        X_test = np.load(data_path / "test.npy")
        y_test = np.load(data_path / "test_label.npy")
    print(X_train.shape, X_test.shape, y_test.shape)
    return X_train, X_test, y_test


if __name__ == "__main__":
    # mode = "open"
    # data_path = Path("datasets/open")
    # mode = "SMD"
    # data_path = Path("datasets/SMD")

    mode = "swat"
    data_path = Path("datasets/SWaT")

    X_train, X_test, y_test = load_datasets(data_path, mode=mode)

    clf = AnomalyTransformer(device=device)
    clf.fit(X_train)

    train_scores = clf.decision_function(X_train)
    train_scores = pd.DataFrame(train_scores)
    train_scores.to_pickle(data_path / "train_scores.pkl")

    test_scores = clf.decision_function(X_test)
    test_scores = pd.DataFrame(test_scores)
    test_scores.to_pickle(data_path / "test_scores.pkl")

    # evaluation of time series anomaly detection
    from deepod.metrics import ts_metrics
    from deepod.metrics import (
        point_adjustment,
    )  # execute point adjustment for time series ad

    scores = clf.decision_function(test)
    eval_metrics = ts_metrics(labels, scores)
    adj_eval_metrics = ts_metrics(labels, point_adjustment(labels, scores))
    print(eval_metrics)
    print(adj_eval_metrics)
    # results : roc_auc_score, average_precision_score, best_f1, best_p, best_r
    # (0.76625, 0.26313, 0.30726, 0.42068, 0.24201)
