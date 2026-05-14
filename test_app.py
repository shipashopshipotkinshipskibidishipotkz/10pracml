import os
import pytest
import pandas as pd
from sklearn.metrics import accuracy_score
from app import mask_author, mask_address, predict_sentiment

def test_mask_author():
    assert mask_author("AMRENDRA") == "AM***"
    assert mask_author("A") == "*"
    assert mask_author("Jo") == "Jo***"

def test_mask_address():
    assert mask_address("New delhi") == "Ne***hi"
    assert mask_address("NY") == "***"
    assert mask_address("Moscow") == "Mo***ow"

def test_data_loading():
    import pickle
    assert os.path.exists("models.pkl")
    with open("models.pkl", "rb") as f:
        artifacts = pickle.load(f)
    for key in ("encoder", "main_model", "metrics", "df", "banks"):
        assert key in artifacts

def test_metrics_calculation():
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 0, 1, 0]
    acc = accuracy_score(y_true, y_pred)
    assert acc == 0.8

def test_recommendation_logic():
    df = pd.DataFrame({
        "bank": ["BankA", "BankA", "BankB", "BankC"],
        "rating": [5, 4, 3, 5]
    })
    liked_banks = ["BankA"]
    recs = df.groupby("bank")["rating"].mean().sort_values(ascending=False)
    recs = recs[~recs.index.isin(liked_banks)].index.tolist()
    assert "BankC" in recs
    assert "BankA" not in recs

def test_predict_sentiment():
    label, confidence = predict_sentiment("Отличный банк, всё понравилось!")
    assert label in ("Позитивный", "Негативный")
    assert 0.0 <= confidence <= 100.0
    label2, confidence2 = predict_sentiment("Ужасный сервис")
    assert isinstance(label2, str)
    assert isinstance(confidence2, float)