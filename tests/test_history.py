"""Unit tests for prediction history logging and dashboard metrics."""

import pandas as pd
import pytest
from app import log_prediction, summarize_dashboard


def test_log_prediction_and_summarize(tmp_path, monkeypatch):
    """Verify logging a prediction and reading it via summarize_dashboard."""
    test_csv = tmp_path / "test_history.csv"
    monkeypatch.setattr("app.HISTORY_PATH", test_csv)

    # Empty check
    df_initial = summarize_dashboard()
    assert df_initial.empty

    # Log first prediction
    log_prediction(
        class_name="plastic",
        confidence=0.95,
        weight_kg=2.0,
        value_inr=36.0,
        co2_saved_kg=3.6,
    )

    assert test_csv.exists()
    df_after = summarize_dashboard()
    assert len(df_after) == 1
    assert df_after.iloc[0]["class"] == "plastic"
    assert float(df_after.iloc[0]["weight_kg"]) == 2.0
    assert float(df_after.iloc[0]["value_inr"]) == 36.0

    # Log second prediction
    log_prediction(
        class_name="organic",
        confidence=0.88,
        weight_kg=1.5,
        value_inr=13.5,
        co2_saved_kg=3.3,
    )

    df_multi = summarize_dashboard()
    assert len(df_multi) == 2


def test_summarize_dashboard_corrupt_file(tmp_path, monkeypatch):
    """Verify graceful handling if history.csv is corrupted or empty."""
    corrupt_csv = tmp_path / "corrupt_history.csv"
    corrupt_csv.write_text("corrupted,unparseable\n1,2,3,4,5,6,7,8", encoding="utf-8")
    monkeypatch.setattr("app.HISTORY_PATH", corrupt_csv)

    df = summarize_dashboard()
    assert isinstance(df, pd.DataFrame)
