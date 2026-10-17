import argparse
from pathlib import Path

import joblib
import pandas as pd

from fraud_model import predict_listings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="저장된 전처리·모델·임계값으로 새 매물 예측")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="artifacts/fraud_detector.joblib")
    parser.add_argument("--output", default="predictions.csv")
    arguments = parser.parse_args()
    if Path(arguments.input).resolve() == Path(arguments.output).resolve():
        parser.error("입력 데이터와 출력 경로는 달라야 합니다.")
    artifact = joblib.load(arguments.model)
    predictions = predict_listings(artifact, pd.read_csv(arguments.input))
    predictions.to_csv(arguments.output, index=False, encoding="utf-8-sig")
    print(f"{len(predictions)}개 매물 예측 저장: {arguments.output}")
