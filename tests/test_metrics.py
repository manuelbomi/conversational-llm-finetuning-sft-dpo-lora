from src.eval.metrics import compute_label_metrics, compute_preference_win_rate, parse_completion


def test_parse_completion_extracts_label():
    assert parse_completion("Label: OBJECTION\nResponse: something") == "OBJECTION"


def test_parse_completion_returns_none_when_unparseable():
    assert parse_completion("garbage output with no structure") is None


def test_parse_completion_handles_leading_whitespace_and_case():
    assert parse_completion("   Label:   BUYING_SIGNAL\nResponse: ok") == "BUYING_SIGNAL"


def test_compute_label_metrics_perfect_predictions():
    labels = ["OBJECTION", "NEUTRAL"]
    predictions = ["OBJECTION", "NEUTRAL", "OBJECTION"]
    golds = ["OBJECTION", "NEUTRAL", "OBJECTION"]
    result = compute_label_metrics(predictions, golds, labels)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["parse_failure_rate"] == 0.0


def test_compute_label_metrics_known_worked_example():
    # 4 examples, 2 labels. gold: A,A,B,B. pred: A,B,B,B.
    # A: tp=1 fp=0 fn=1 -> precision=1.0, recall=0.5, f1=0.667
    # B: tp=2 fp=1 fn=0 -> precision=0.667, recall=1.0, f1=0.8
    labels = ["A", "B"]
    predictions = ["A", "B", "B", "B"]
    golds = ["A", "A", "B", "B"]
    result = compute_label_metrics(predictions, golds, labels)
    assert result["per_label"]["A"]["precision"] == 1.0
    assert result["per_label"]["A"]["recall"] == 0.5
    assert round(result["per_label"]["A"]["f1"], 3) == 0.667
    assert round(result["per_label"]["B"]["precision"], 3) == 0.667
    assert result["per_label"]["B"]["recall"] == 1.0
    assert result["accuracy"] == 0.75


def test_compute_label_metrics_unparseable_prediction_counts_as_wrong():
    labels = ["A", "B"]
    predictions = [None, "B"]
    golds = ["A", "B"]
    result = compute_label_metrics(predictions, golds, labels)
    assert result["accuracy"] == 0.5
    assert result["parse_failure_rate"] == 0.5


def test_compute_preference_win_rate_all_correct():
    chosen = [1.0, 2.0, 3.0]
    rejected = [0.5, 1.0, 2.0]
    assert compute_preference_win_rate(chosen, rejected) == 1.0


def test_compute_preference_win_rate_all_wrong():
    chosen = [0.5, 1.0]
    rejected = [1.0, 2.0]
    assert compute_preference_win_rate(chosen, rejected) == 0.0


def test_compute_preference_win_rate_mixed():
    chosen = [1.0, 0.5, 3.0, 0.1]
    rejected = [0.5, 1.0, 2.0, 0.2]
    assert compute_preference_win_rate(chosen, rejected) == 0.5


def test_compute_preference_win_rate_empty():
    assert compute_preference_win_rate([], []) == 0.0
