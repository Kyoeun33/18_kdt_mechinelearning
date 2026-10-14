"""CH03 3_2: 당뇨병 데이터 로지스틱 회귀 분석"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def print_metrics(y_true, y_pred, y_score, title):
    """분류 성능 지표를 화면에 출력합니다."""
    print(f"\n[{title}]")
    print("혼동 행렬:\n", confusion_matrix(y_true, y_pred))
    print(f"정확도: {accuracy_score(y_true, y_pred):.4f}")
    print(f"정밀도: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"재현율: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 점수: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_score):.4f}")


def main():
    data_path = Path(__file__).with_name("diabetes.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    df = pd.read_csv(data_path)
    if "Outcome" not in df.columns:
        raise ValueError("CSV에 정답 열 'Outcome'이 필요합니다.")

    print("데이터 미리보기:")
    print(df.head())
    print("\nOutcome 분포:")
    print(df["Outcome"].value_counts().sort_index())

    X = df.drop(columns="Outcome")
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 스케일러는 학습 데이터에만 맞춘 뒤 테스트 데이터에 적용합니다.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_train_scaled, y_train)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    print_metrics(y_test, predictions, probabilities, "기본 임계값 0.50")

    print("\n[임계값별 결과]")
    for threshold in (0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.50):
        threshold_predictions = (probabilities >= threshold).astype(int)
        print(
            f"임계값 {threshold:.2f} | "
            f"정확도 {accuracy_score(y_test, threshold_predictions):.4f} | "
            f"정밀도 {precision_score(y_test, threshold_predictions, zero_division=0):.4f} | "
            f"재현율 {recall_score(y_test, threshold_predictions, zero_division=0):.4f} | "
            f"F1 {f1_score(y_test, threshold_predictions, zero_division=0):.4f}"
        )

    print("\n[기술 통계]")
    print(df.describe())


if __name__ == "__main__":
    main()
