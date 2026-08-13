import json
from pathlib import Path

import pytest

from bayesian_rag_evaluator.bn.calibration import (
    learn_cpds_from_data,
    load_labeled_examples,
    tune_thresholds_from_labels,
)


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "examples"


def test_load_labeled_examples():
    examples = load_labeled_examples(EXAMPLES_DIR / "labeled.json")
    assert len(examples) == 3
    assert examples[0].labels.query_relevance == "high"


def test_learn_cpds_from_labeled_data():
    examples = load_labeled_examples(EXAMPLES_DIR / "labeled.json")
    model = learn_cpds_from_data(examples)
    assert model.check_model()


def test_tune_thresholds():
    examples = load_labeled_examples(EXAMPLES_DIR / "labeled.json")
    suggested = tune_thresholds_from_labels(examples)
    assert "bins" in suggested


def test_batch_requests_file_valid_json():
    data = json.loads((EXAMPLES_DIR / "batch_requests.json").read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 2
