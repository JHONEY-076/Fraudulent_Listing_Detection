import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from fraud_model import REQUIRED_COLUMNS, TARGET, ListingFeatures, evaluate, predict_listings
from train_model import bootstrap_intervals


def listing_fingerprints(frame):
    features = ListingFeatures().fit_transform(frame)
    return pd.util.hash_pandas_object(features, index=False)


def date_range(frame):
    dates = pd.to_datetime(frame["게재일"], errors="coerce")
    return {
        "min": dates.min().isoformat() if dates.notna().any() else None,
        "max": dates.max().isoformat() if dates.notna().any() else None,
        "missing_or_invalid": int(dates.isna().sum()),
    }


def write_report(output, report):
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "operations_validation.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def run_validation(data_path, model_path="artifacts/fraud_detector.joblib",
                   reference_path="train.csv", output_dir="artifacts/operations",
                   label_column=TARGET):
    data_path, model_path, reference_path = Path(data_path), Path(model_path), Path(reference_path)
    output = Path(output_dir)
    destinations = [output / "operations_validation.json", output / "operations_predictions.csv"]
    if any(destination.resolve() in {path.resolve() for path in [data_path, model_path, reference_path]}
           for destination in destinations):
        raise ValueError("입력 파일과 결과 파일의 경로는 달라야 합니다.")
    artifact = joblib.load(model_path)
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    if reference_sha != artifact["manifest"]["data_sha256"]:
        raise ValueError("원본 학습·선택 데이터의 해시가 모델 기록과 다릅니다. 정확한 원본을 지정하세요.")
    frame, reference = pd.read_csv(data_path), pd.read_csv(reference_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"새 데이터가 비어 있거나 필수 컬럼이 없습니다: {missing}")
    if label_column in REQUIRED_COLUMNS:
        raise ValueError("실제 허위 여부 라벨은 입력 변수 컬럼과 달라야 합니다.")
    labels = pd.Series(np.nan, index=frame.index)
    if label_column in frame:
        labels = pd.to_numeric(frame[label_column], errors="coerce")
        invalid = (frame[label_column].notna() & labels.isna()) | (labels.notna() & ~labels.isin([0, 1]))
        if invalid.any():
            raise ValueError("확정 라벨은 0/1, 미확정 라벨은 빈 값이어야 합니다.")

    fingerprints = listing_fingerprints(frame)
    feature_overlap = fingerprints.isin(listing_fingerprints(reference))
    id_overlap = pd.Series(False, index=frame.index)
    duplicate_ids = pd.Series(False, index=frame.index)
    if "ID" in frame:
        ids = frame["ID"].astype("string")
        duplicate_ids = ids.notna() & ids.duplicated(keep=False)
        if "ID" in reference:
            id_overlap = ids.notna() & ids.isin(reference["ID"].astype("string").dropna())
    duplicate_features = fingerprints.duplicated(keep=False)
    overlap = feature_overlap | id_overlap
    report = {
        "status": "pending",
        "data_path": str(data_path), "model_path": str(model_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "reference_sha256": reference_sha,
        "candidate": artifact["candidate"], "fixed_threshold": float(artifact["threshold"]),
        "rows": len(frame), "label_column": label_column,
        "labeled_rows": int(labels.notna().sum()), "label_coverage": float(labels.notna().mean()),
        "reference_listing_dates": date_range(reference), "batch_listing_dates": date_range(frame),
        "overlap_with_project_data": int(overlap.sum()),
        "overlap_by_id": int(id_overlap.sum()), "overlap_by_features": int(feature_overlap.sum()),
        "duplicate_id_rows": int(duplicate_ids.sum()), "duplicate_feature_rows": int(duplicate_features.sum()),
        "metrics": None, "bootstrap_95_ci": None,
        "notes": [
            "The saved model and threshold stay fixed; this command performs no fitting or tuning.",
            "The reference includes all historical project rows, including previously used validation rows.",
            "Exact ID/feature checks cannot detect all relisted properties or near-duplicate listings.",
            "Listing dates are not collection dates. External collection provenance must be verified separately.",
        ],
    }
    if overlap.any() or duplicate_ids.any() or duplicate_features.any():
        report["status"] = "rejected_overlap_or_duplicates"
        report["notes"].append("No operational metrics were computed because the batch overlaps historical data or repeats listings.")
        return write_report(output, report)

    predictions = predict_listings(artifact, frame.drop(columns=[label_column], errors="ignore"))
    predictions["actual_label"] = labels
    predictions["label_confirmed"] = labels.notna()
    confirmed = labels.notna()
    if confirmed.any():
        target = labels.loc[confirmed].astype(int).to_numpy()
        probabilities = predictions.loc[confirmed, "허위매물점수"].to_numpy()
        if set(np.unique(target)) == {0, 1}:
            report["metrics"] = evaluate(target, probabilities, artifact["threshold"])
            report["bootstrap_95_ci"] = bootstrap_intervals(target, probabilities, artifact["threshold"], seed=20261017)
        else:
            predicted = probabilities >= artifact["threshold"]
            true_negative, false_positive, false_negative, true_positive = confusion_matrix(
                target, predicted, labels=[0, 1],
            ).ravel()
            precision_denominator = int(true_positive + false_positive)
            recall_denominator = int(true_positive + false_negative)
            f1_denominator = int(2 * true_positive + false_positive + false_negative)
            report["metrics"] = {
                "threshold": float(artifact["threshold"]), "accuracy": float((target == predicted).mean()),
                "precision": float(true_positive / precision_denominator) if precision_denominator else None,
                "recall": float(true_positive / recall_denominator) if recall_denominator else None,
                "f1": float(2 * true_positive / f1_denominator) if f1_denominator else None,
                "roc_auc": None, "average_precision": None,
                "tn": int(true_negative), "fp": int(false_positive),
                "fn": int(false_negative), "tp": int(true_positive),
            }
            report["notes"].append("Only one confirmed class is present; some metrics and bootstrap intervals are unavailable.")
        report["confirmed_positives"] = int(target.sum())
        report["status"] = "evaluated" if confirmed.all() else "evaluated_partial_labels"
        if not confirmed.all():
            report["notes"].append("Metrics cover confirmed labels only; selective label verification can bias operating performance.")
    else:
        report["status"] = "awaiting_labels"
        report["notes"].append("Scores were generated, but recall/F1 cannot be measured without confirmed ground truth.")
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "operations_predictions.csv", index=False, encoding="utf-8-sig")
    return write_report(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="수집 매물에서 저장 모델·임계값을 고정하여 운영 검증")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="artifacts/fraud_detector.joblib")
    parser.add_argument("--reference", default="train.csv")
    parser.add_argument("--output", default="artifacts/operations")
    parser.add_argument("--label-column", default=TARGET)
    arguments = parser.parse_args()
    try:
        report = run_validation(arguments.data, arguments.model, arguments.reference, arguments.output, arguments.label_column)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"검증 중단: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if report["status"] == "rejected_overlap_or_duplicates":
        parser.exit(2, "기존 데이터 중복 또는 배치 내 중복으로 운영 평가를 중단했습니다.\n")
