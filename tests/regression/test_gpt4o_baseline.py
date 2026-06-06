"""Regression tests for GPT-4o baseline metrics. Fail if metrics drop > 5%."""
import pytest
import json
from pathlib import Path

BASELINE = {
    "hotpotqa_f1": 0.74,
    "gsm8k_accuracy": 0.88,
    "hallucination_rate": 0.062,
    "faithfulness": 0.91,
}
TOLERANCE = 0.05  # 5% regression threshold

def load_latest_metrics() -> dict:
    p = Path("metrics/latest_eval.json")
    if not p.exists():
        pytest.skip("No metrics file found")
    return json.loads(p.read_text())

def test_hotpotqa_no_regression():
    metrics = load_latest_metrics()
    score = metrics.get("hotpotqa_f1", 0)
    assert score >= BASELINE["hotpotqa_f1"] * (1 - TOLERANCE), \
        f"HotpotQA F1 regressed: {score:.3f} < {BASELINE['hotpotqa_f1'] * (1 - TOLERANCE):.3f}"

def test_gsm8k_no_regression():
    metrics = load_latest_metrics()
    score = metrics.get("gsm8k_accuracy", 0)
    assert score >= BASELINE["gsm8k_accuracy"] * (1 - TOLERANCE)

def test_hallucination_rate_no_regression():
    metrics = load_latest_metrics()
    rate = metrics.get("hallucination_rate", 1.0)
    assert rate <= BASELINE["hallucination_rate"] * (1 + TOLERANCE), \
        f"Hallucination rate worsened: {rate:.3f}"

def test_faithfulness_no_regression():
    metrics = load_latest_metrics()
    score = metrics.get("faithfulness", 0)
    assert score >= BASELINE["faithfulness"] * (1 - TOLERANCE)
