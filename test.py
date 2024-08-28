# time series anomaly detection methods
from deepod.models.time_series import AnomalyTransformer, TranAD, TimesNet
from pathlib import Path
import pickle
import pandas as pd

data_path = Path("datasets")
train_df = pickle.load(open(data_path / "train.pkl", "rb"))
test_df = pickle.load(open(data_path / "test.pkl", "rb"))
X_train = train_df.values
X_test = test_df.values

# clf = TimesNet()
clf = AnomalyTransformer()
clf.fit(X_train)
scores = clf.decision_function(X_test)
scores = pd.DataFrame(scores, index=test_df.index)
scores.to_pickle(data_path / "scores.pkl")
print(scores)

# # evaluation of time series anomaly detection
# from deepod.metrics import ts_metrics
# from deepod.metrics import (
#     point_adjustment,
# )  # execute point adjustment for time series ad
#
# eval_metrics = ts_metrics(labels, scores)
# adj_eval_metrics = ts_metrics(labels, point_adjustment(labels, scores))
