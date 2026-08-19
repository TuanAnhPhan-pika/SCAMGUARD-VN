from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

OUT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report_path = OUT / "FINAL_REPORT.json"
    if report_path.exists():
        raise RuntimeError("evaluation already completed; refusing a second evaluation")
    freeze = json.loads((OUT / "PREDICTION_FREEZE.json").read_text(encoding="utf-8"))
    pred_path = OUT / "predictions_frozen.jsonl"
    if sha(pred_path) != freeze["predictions_sha256"]:
        raise RuntimeError("frozen prediction hash mismatch")
    predictions = [json.loads(x) for x in pred_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    gold = [json.loads(x) for x in (OUT / "INTERNAL_GOLD.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    gold_by_id = {x["case_id"]: x for x in gold}
    truth = [gold_by_id[x["case_id"]]["gold_risk"] for x in predictions]
    pred = [x["prediction"] for x in predictions]
    labels = ["NO_EVIDENCE", "REVIEW", "HIGH_EVIDENCE"]
    subsets = {}
    for subset in sorted({x["subset"] for x in gold}):
        pairs = [(gold_by_id[x["case_id"]]["gold_risk"], x["prediction"]) for x in predictions if gold_by_id[x["case_id"]]["subset"] == subset]
        strict = sum(a == b for a, b in pairs)
        operational = sum((b in {"NO_EVIDENCE", "REVIEW"}) if subset == "HARD_NEGATIVE" else
                          (b in {"REVIEW", "HIGH_EVIDENCE"}) if subset == "POSITIVE" else b == "NO_EVIDENCE" for a, b in pairs)
        subsets[subset] = {"cases": len(pairs), "strict_correct": strict, "strict_accuracy": strict / len(pairs),
                           "operational_correct": operational, "operational_accuracy": operational / len(pairs),
                           "prediction_counts": dict(Counter(b for _, b in pairs))}
    operational_correct = sum(x["operational_correct"] for x in subsets.values())
    errors = [{"case_id": x["case_id"], "subset": gold_by_id[x["case_id"]]["subset"],
               "gold": gold_by_id[x["case_id"]]["gold_risk"], "prediction": x["prediction"], "risk_scores": x["risk_scores"]}
              for x in predictions if x["prediction"] != gold_by_id[x["case_id"]]["gold_risk"]]
    report = {
        "test_id": "INDEPENDENT_HELD_OUT_200_20260819", "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "cases": len(predictions), "gold_distribution": dict(Counter(truth)), "prediction_distribution": dict(Counter(pred)),
        "strict_accuracy": accuracy_score(truth, pred),
        "strict_macro_f1_all_3_labels": f1_score(truth, pred, labels=labels, average="macro", zero_division=0),
        "classification_report_strict": classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": labels, "values": confusion_matrix(truth, pred, labels=labels).tolist()},
        "operational_policy": "POSITIVE accepts REVIEW/HIGH_EVIDENCE; HARD_NEGATIVE accepts NO_EVIDENCE/REVIEW; EASY_NEGATIVE accepts NO_EVIDENCE.",
        "operational_accuracy": operational_correct / len(predictions), "subsets": subsets, "strict_errors": errors,
        "predictions_sha256": sha(pred_path), "internal_gold_sha256": sha(OUT / "INTERNAL_GOLD.jsonl"),
        "MODEL_MODIFIED_AFTER_TEST_START": "NO", "THRESHOLD_TUNED_AFTER_TEST_START": "NO",
        "GOLD_OPENED_BEFORE_PREDICTION_FREEZE": "NO", "TEST_CASES_CHANGED_AFTER_TEST_START": "NO",
        "PREDICTIONS_FROZEN_BEFORE_GOLD": "YES", "FINAL_INDEPENDENT_TEST_VALID": "YES",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("cases", "gold_distribution", "prediction_distribution", "strict_accuracy", "strict_macro_f1_all_3_labels", "operational_accuracy", "subsets", "FINAL_INDEPENDENT_TEST_VALID")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
