# time series anomaly detection methods
from deepod.models.time_series import AnomalyTransformer, TranAD, TimesNet
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

# data_path = Path("datasets/open")
# train_df = pickle.load(open(data_path / "train.pkl", "rb"))
# test_df = pickle.load(open(data_path / "test.pkl", "rb"))
# X_train = train_df.values
# X_test = test_df.values
#
# clf = AnomalyTransformer()
# clf.fit(X_train)
#
# train_scores = clf.decision_function(X_train)
# train_scores = pd.DataFrame(train_scores, index=train_df.index)
# train_scores.to_pickle(data_path / "train_scores.pkl")
#
# test_scores = clf.decision_function(X_test)
# test_scores = pd.DataFrame(test_scores, index=test_df.index)
# test_scores.to_pickle(data_path / "test_scores.pkl")

# evaluation of time series anomaly detection
from deepod.metrics import ts_metrics
from deepod.metrics import (
    point_adjustment,
)  # execute point adjustment for time series ad

data_path = Path("datasets/SMD")
with open(data_path / "SMD_train.npy", "rb") as f:
    train = np.load(f)
with open(data_path / "SMD_test.npy", "rb") as f:
    test = np.load(f)
with open(data_path / "SMD_test_label.npy", "rb") as f:
    labels = np.load(f)
print(train.shape, test.shape, labels.shape)

# eval_metrics = ts_metrics(labels, scores)
# adj_eval_metrics = ts_metrics(labels, point_adjustment(labels, scores))
