import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from fraud_model import TARGET, candidate_grid, evaluate, make_pipeline, select_threshold, threshold_table


def save_json(path, contents):
    path.write_text(json.dumps(contents, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def bootstrap_intervals(target, probabilities, threshold, seed, iterations=1000):
    random = np.random.default_rng(seed)
    target = np.asarray(target)
    predicted = probabilities >= threshold
    positives = np.flatnonzero(target == 1)
    negatives = np.flatnonzero(target == 0)
    scores = []
    for _ in range(iterations):
        sample = np.concatenate([random.choice(positives, len(positives)), random.choice(negatives, len(negatives))])
        true_positive = np.sum(predicted[sample] & (target[sample] == 1))
        false_positive = np.sum(predicted[sample] & (target[sample] == 0))
        false_negative = len(positives) - true_positive
        scores.append([true_positive / len(positives),
                       2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)])
    limits = np.quantile(scores, [0.025, 0.975], axis=0)
    return {metric: limits[:, index].tolist() for index, metric in enumerate(["recall", "f1"])}


def save_plot(output, comparison, target, probabilities, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay

    best_by_family = comparison.drop_duplicates("family").set_index("family")
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    best_by_family[["recall", "f1"]].plot.barh(ax=axes[0], color=["#2a9d8f", "#264653"])
    axes[0].set(xlim=(0, 1), title="Development OOF: model selection", ylabel="")
    PrecisionRecallDisplay.from_predictions(target, probabilities, ax=axes[1])
    axes[1].scatter(metrics["recall"], metrics["precision"], color="#e76f51", label="Selected threshold")
    axes[1].set_title("Holdout: precision vs. recall")
    axes[1].legend(loc="lower left")
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    axes[2].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[2].text(column, row, str(matrix[row, column]), ha="center", va="center",
                         color="white" if matrix[row, column] > matrix.max() / 2 else "black", fontsize=18)
    axes[2].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"], xlabel="Predicted", ylabel="Actual", title="Holdout: confusion matrix")
    figure.savefig(output / "evaluation.png", dpi=160)
    plt.close(figure)


def run_training(data_path="train.csv", output_dir="artifacts", min_recall=0.8,
                 folds=5, seed=20261017, threads=2):
    if not 0 < min_recall <= 1 or folds < 2 or threads < 1:
        raise ValueError("min_recall: (0, 1], folds: 2 이상, threads: 1 이상이어야 합니다.")
    data_path, output = Path(data_path), Path(output_dir)
    frame = pd.read_csv(data_path)
    if TARGET not in frame or frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError(f"{TARGET} 컬럼에는 결측치 없이 0과 1이 모두 있어야 합니다.")
    if frame.drop(columns=[TARGET, "ID"], errors="ignore").duplicated().any():
        raise ValueError("중복 매물의 분할 누수를 방지하려면 먼저 중복을 확인하거나 그룹 분할을 적용하세요.")
    features, target = frame.drop(columns=TARGET), frame[TARGET].astype(int)
    development, holdout = train_test_split(np.arange(len(frame)), test_size=0.2, stratify=target, random_state=seed)
    development_features = features.iloc[development].reset_index(drop=True)
    development_target = target.iloc[development].reset_index(drop=True)
    if development_target.value_counts().min() < folds:
        raise ValueError("각 클래스의 개발 데이터 수보다 folds가 클 수 없습니다.")
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(development_features, development_target))
    output.mkdir(parents=True, exist_ok=True)
    candidates = candidate_grid()
    manifest = {
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "seed": seed, "folds": folds, "min_recall": min_recall, "threads": threads,
        "selection_rule": "Maximize development OOF F1 subject to recall >= min_recall; ties: recall, AP, candidate name.",
        "development_rows": development.tolist(), "holdout_rows": holdout.tolist(),
        "candidates": [asdict(candidate) for candidate in candidates],
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "xgboost", "joblib"]},
    }
    save_json(output / "run_manifest.json", manifest)
    results, predictions, fold_results = [], {}, []
    with threadpool_limits(limits=threads):
        for candidate_index, candidate in enumerate(candidates, start=1):
            started = perf_counter()
            probabilities = np.full(len(development), np.nan)
            print(f"[{candidate_index}/{len(candidates)}] {candidate.name}", flush=True)
            for fold_index, (training_rows, validation_rows) in enumerate(splits, start=1):
                pipeline = make_pipeline(candidate, seed=seed + fold_index, threads=threads)
                pipeline.fit(development_features.iloc[training_rows], development_target.iloc[training_rows])
                probabilities[validation_rows] = pipeline.predict_proba(development_features.iloc[validation_rows])[:, 1]
                fold_results.append({"candidate": candidate.name, "fold": fold_index,
                                     **evaluate(development_target.iloc[validation_rows], probabilities[validation_rows], 0.5)})
            selected = select_threshold(development_target, probabilities, min_recall)
            metrics = evaluate(development_target, probabilities, selected["threshold"])
            defaults = evaluate(development_target, probabilities, 0.5)
            results.append({"candidate": candidate.name, "family": candidate.family, **metrics,
                            "default_recall": defaults["recall"], "default_f1": defaults["f1"],
                            "seconds": round(perf_counter() - started, 2)})
            predictions[candidate.name] = probabilities
            pd.DataFrame(results).to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
            print(f"  OOF recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, AP={metrics['average_precision']:.4f}", flush=True)

        comparison = pd.DataFrame(results).sort_values(
            ["f1", "recall", "average_precision", "candidate"], ascending=[False, False, False, True],
        ).reset_index(drop=True)
        winner = comparison.iloc[0]
        chosen = next(candidate for candidate in candidates if candidate.name == winner["candidate"])
        selected_threshold = float(winner["threshold"])
        print(f"Selected before holdout evaluation: {chosen.name}, threshold={selected_threshold:.6f}", flush=True)
        pipeline = make_pipeline(chosen, seed=seed, threads=threads)
        pipeline.fit(development_features, development_target)
        artifact = {"pipeline": pipeline, "threshold": selected_threshold,
                    "candidate": asdict(chosen), "min_recall": min_recall, "manifest": manifest}
        joblib.dump(artifact, output / "fraud_detector.joblib", compress=3)
        holdout_probabilities = pipeline.predict_proba(features.iloc[holdout])[:, 1]

    holdout_target = target.iloc[holdout].to_numpy()
    tuned = evaluate(holdout_target, holdout_probabilities, selected_threshold)
    default = evaluate(holdout_target, holdout_probabilities, 0.5)
    report = {
        "selected_candidate": asdict(chosen), "threshold": selected_threshold,
        "selection_min_recall": min_recall, "development_size": len(development), "holdout_size": len(holdout),
        "development_positives": int(development_target.sum()), "holdout_positives": int(holdout_target.sum()),
        "oof_selection_metrics": evaluate(development_target, predictions[chosen.name], selected_threshold),
        "holdout_default": default, "holdout_tuned": tuned,
        "holdout_recall_target_met": bool(tuned["recall"] >= min_recall),
        "holdout_bootstrap_95_ci": bootstrap_intervals(holdout_target, holdout_probabilities, selected_threshold, seed),
        "evaluation_notes": [
            "OOF scores are used for model/threshold selection and are optimistic selection scores, not independent test scores.",
            "Holdout labels were excluded from preprocessing, fitting, candidate selection, and threshold tuning in this run.",
            "All rows come from the existing project dataset; this is not a new external validation dataset.",
            "Recall constraint applies to development OOF predictions; it is not a guarantee on new data.",
            "Bootstrap intervals condition on the fixed model and this holdout; they exclude model selection uncertainty.",
            "Saved model is trained on development rows only, matching the evaluated model. No all-data refit is performed.",
        ],
    }
    save_json(output / "metrics.json", report)
    comparison.to_csv(output / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_results).to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    threshold_table(development_target, predictions[chosen.name]).to_csv(output / "threshold_curve.csv", index=False)
    oof = pd.DataFrame({"source_row": development, "y_true": development_target, **predictions})
    oof.to_csv(output / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    holdout_predictions = pd.DataFrame({"source_row": holdout, "y_true": holdout_target,
                                        "probability": holdout_probabilities,
                                        "prediction": (holdout_probabilities >= selected_threshold).astype(int)})
    if "ID" in frame:
        holdout_predictions.insert(1, "ID", frame.iloc[holdout]["ID"].to_numpy())
    holdout_predictions.to_csv(output / "holdout_predictions.csv", index=False, encoding="utf-8-sig")
    save_plot(output, comparison, holdout_target, holdout_probabilities, tuned)
    print(json.dumps({"winner": chosen.name, "holdout_default": default, "holdout_tuned": tuned}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOF 재현율 조건 아래 F1을 최대화하는 허위매물 모델 선택")
    parser.add_argument("--data", default="train.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args()
    run_training(arguments.data, arguments.output, arguments.min_recall,
                 arguments.folds, arguments.seed, arguments.threads)
