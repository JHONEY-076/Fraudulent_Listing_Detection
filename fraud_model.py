from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier


TARGET = "허위매물여부"
CATEGORICAL_COLUMNS = ["매물확인방식", "방향", "주차가능여부", "중개사무소", "제공플랫폼"]
NUMERIC_COLUMNS = ["보증금", "월세", "전용면적", "해당층", "총층", "방수", "욕실수", "총주차대수", "관리비"]
REQUIRED_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + ["게재일"]


class ListingFeatures(TransformerMixin, BaseEstimator):
    def fit(self, frame, target=None):
        self.feature_names_out_ = np.asarray(self._build(frame).columns, dtype=object)
        return self

    def transform(self, frame):
        check_is_fitted(self, "feature_names_out_")
        return self._build(frame).loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    @staticmethod
    def _build(frame):
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"필수 컬럼이 없습니다: {missing}")
        features = pd.DataFrame(index=frame.index)
        for column in CATEGORICAL_COLUMNS:
            features[column] = frame[column].fillna("__MISSING__").astype(str)
        for column in NUMERIC_COLUMNS:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            features[column] = values.astype(float)
            features[f"{column}_missing"] = values.isna().astype(int)

        dates = pd.to_datetime(frame["게재일"], errors="coerce")
        features["listing_month"] = dates.dt.month
        features["listing_day"] = dates.dt.day
        features["listing_weekday"] = dates.dt.dayofweek
        features["listing_days_since_2020"] = (dates - pd.Timestamp("2020-01-01")).dt.days
        features["listing_date_missing"] = dates.isna().astype(int)

        ratios = {
            "deposit_per_area": ("보증금", "전용면적"),
            "rent_per_area": ("월세", "전용면적"),
            "maintenance_per_area": ("관리비", "전용면적"),
            "area_per_room": ("전용면적", "방수"),
            "area_per_bathroom": ("전용면적", "욕실수"),
            "floor_ratio": ("해당층", "총층"),
            "rent_to_deposit": ("월세", "보증금"),
            "parking_per_floor": ("총주차대수", "총층"),
        }
        for name, (numerator, denominator) in ratios.items():
            features[name] = features[numerator] / features[denominator].where(features[denominator] > 0)
        for column in ["보증금", "월세", "전용면적", "총주차대수", "관리비"]:
            features[f"{column}_log1p"] = np.log1p(features[column].clip(lower=0))
        features["zero_rent"] = features["월세"].eq(0).astype(int)
        features["floor_exceeds_total"] = features["해당층"].gt(features["총층"]).astype(int)
        return features.replace([np.inf, -np.inf], np.nan)


@dataclass
class Candidate:
    name: str
    family: str
    params: dict


def candidate_grid():
    candidates = []
    for regularization in [0.1, 1.0]:
        candidates.append(Candidate(f"logistic_c{regularization}", "LogisticRegression", {"C": regularization}))
    for family in ["RandomForest", "ExtraTrees"]:
        for leaf_size in [1, 3]:
            candidates.append(Candidate(f"{family}_leaf{leaf_size}", family, {"min_samples_leaf": leaf_size}))
    for leaf_nodes in [7, 15, 31]:
        candidates.append(Candidate(f"hist_leaves{leaf_nodes}", "HistGradientBoosting", {"max_leaf_nodes": leaf_nodes}))
    for depth, weight, regularization in [(4, 1, 3), (4, 3, 5), (5, 1, 5), (5, 3, 5), (6, 1, 5), (6, 3, 5), (7, 3, 7), (6, 6, 7)]:
        candidates.append(Candidate(f"cat_d{depth}_w{weight}", "CatBoost", {
            "depth": depth, "class_weights": [1.0, float(weight)], "l2_leaf_reg": regularization,
        }))
    for leaves, weight in [(7, 1), (15, 1), (15, 3), (31, 3)]:
        candidates.append(Candidate(f"lgb_leaves{leaves}_w{weight}", "LightGBM", {
            "num_leaves": leaves, "scale_pos_weight": weight,
        }))
    for depth, weight in [(2, 1), (3, 1), (3, 3), (5, 3)]:
        candidates.append(Candidate(f"xgb_d{depth}_w{weight}", "XGBoost", {
            "max_depth": depth, "scale_pos_weight": weight,
        }))
    return candidates


def numeric_feature_columns(frame):
    return [column for column in frame if column not in CATEGORICAL_COLUMNS]


def make_pipeline(candidate, seed=42, threads=2):
    common = {"random_state": seed}
    if candidate.family == "CatBoost":
        model = CatBoostClassifier(
            iterations=700, learning_rate=0.045, loss_function="Logloss",
            cat_features=CATEGORICAL_COLUMNS, thread_count=threads,
            allow_writing_files=False, verbose=False, **common, **candidate.params,
        )
        return Pipeline([("features", ListingFeatures()), ("model", model)])

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, numeric_feature_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=3), CATEGORICAL_COLUMNS),
    ])
    if candidate.family == "LogisticRegression":
        model = LogisticRegression(class_weight="balanced", max_iter=3000, **common, **candidate.params)
    elif candidate.family in ["RandomForest", "ExtraTrees"]:
        estimator = RandomForestClassifier if candidate.family == "RandomForest" else ExtraTreesClassifier
        model = estimator(n_estimators=400, class_weight="balanced", max_features=0.7,
                          n_jobs=threads, **common, **candidate.params)
    elif candidate.family == "HistGradientBoosting":
        model = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05,
                                              l2_regularization=5, early_stopping=False,
                                              **common, **candidate.params)
    elif candidate.family == "LightGBM":
        model = LGBMClassifier(n_estimators=500, learning_rate=0.035, min_child_samples=25,
                               reg_lambda=5, verbosity=-1, n_jobs=threads,
                               **common, **candidate.params)
    elif candidate.family == "XGBoost":
        model = XGBClassifier(n_estimators=500, learning_rate=0.04, min_child_weight=5,
                              subsample=0.9, colsample_bytree=0.9, reg_lambda=5,
                              tree_method="hist", eval_metric="logloss", n_jobs=threads,
                              **common, **candidate.params)
    else:
        raise ValueError(f"지원하지 않는 모델: {candidate.family}")
    return Pipeline([("features", ListingFeatures()), ("preprocess", preprocess), ("model", model)])


def threshold_table(target, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("예측 확률은 0~1 사이의 유한한 값이어야 합니다.")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("임계값 선택에는 0/1 클래스가 모두 필요합니다.")
    precisions, recalls, thresholds = precision_recall_curve(target, probabilities)
    precisions, recalls = precisions[:-1], recalls[:-1]
    denominator = precisions + recalls
    scores = np.divide(2 * precisions * recalls, denominator,
                       out=np.zeros_like(denominator), where=denominator > 0)
    return pd.DataFrame({"threshold": thresholds, "precision": precisions, "recall": recalls, "f1": scores})


def select_threshold(target, probabilities, min_recall=0.8):
    if not 0 < min_recall <= 1:
        raise ValueError("min_recall은 0보다 크고 1 이하여야 합니다.")
    table = threshold_table(target, probabilities)
    feasible = table.loc[table["recall"] >= min_recall]
    return feasible.sort_values(["f1", "recall", "precision", "threshold"], ascending=False).iloc[0].to_dict()


def evaluate(target, probabilities, threshold):
    predicted = np.asarray(probabilities) >= threshold
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        target, predicted, labels=[0, 1],
    ).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "tn": int(true_negative), "fp": int(false_positive),
        "fn": int(false_negative), "tp": int(true_positive),
    }


def predict_listings(artifact, frame):
    probabilities = artifact["pipeline"].predict_proba(frame)[:, 1]
    result = pd.DataFrame(index=frame.index)
    if "ID" in frame:
        result["ID"] = frame["ID"]
    result["허위매물점수"] = probabilities
    result["허위매물예측"] = (probabilities >= artifact["threshold"]).astype(int)
    return result
