import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# パス設定
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "../data/Titanic.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "baseline_model.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込み、存在しない場合は取得して保存"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 使用するカラムを限定
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプライン（数値とカテゴリ変数の処理）"""
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデル学習とテストデータ返却"""
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが保存されているかを確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度が閾値以上かを確認（デフォルト0.75）"""
    model, X_test, y_test = train_model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    threshold = float(os.getenv("ACCURACY_THRESHOLD", 0.75))
    assert accuracy >= threshold, f"精度が閾値未満です: {accuracy} < {threshold}"


def test_model_inference_time(train_model):
    """モデルの推論時間が1秒未満であるかを検証"""
    model, X_test, _ = train_model
    start_time = time.time()
    model.predict(X_test)
    duration = time.time() - start_time
    assert duration < 1.0, f"推論時間が長すぎます: {duration:.4f}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """同一条件下で予測結果が再現されるか確認"""
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)

    assert np.array_equal(
        pred1, pred2
    ), "モデルの再現性に問題があります（予測結果が一致しません）"


def test_model_regression(train_model):
    """ベースラインモデルと比較して精度が劣化していないか確認"""
    model, X_test, y_test = train_model

    if not os.path.exists(BASELINE_MODEL_PATH):
        pytest.skip("ベースラインモデルが存在しません")

    with open(BASELINE_MODEL_PATH, "rb") as f:
        baseline_model = pickle.load(f)

    acc_new = accuracy_score(y_test, model.predict(X_test))
    acc_old = accuracy_score(y_test, baseline_model.predict(X_test))

    assert acc_new >= acc_old, f"モデルの精度が退化しています: {acc_new} < {acc_old}"