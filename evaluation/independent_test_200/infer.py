from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from risk_retrain_e2e.runtime import EndToEndRiskRuntime
OUT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    input_path = OUT / "FINAL_MODEL_INPUT.jsonl"
    pred_path = OUT / "predictions_frozen.jsonl"
    if pred_path.exists():
        raise RuntimeError("predictions are already frozen; refusing to rerun")
    rows = [json.loads(x) for x in input_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    runtime = EndToEndRiskRuntime()
    predictions = []
    for row in rows:
        result = runtime.analyze_turns(row["turns"]) if "turns" in row else runtime.analyze(row["text"])
        predictions.append({"case_id": row["case_id"], "prediction": result["risk_level"], "risk_scores": result["risk_scores"]})
    pred_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in predictions), encoding="utf-8")
    freeze = {"phase": "PREDICTIONS_FROZEN_GOLD_NOT_OPENED", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "cases": len(predictions), "predictions_sha256": sha(pred_path), "final_model_input_sha256": sha(input_path)}
    (OUT / "PREDICTION_FREEZE.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(freeze, ensure_ascii=False))


if __name__ == "__main__":
    main()
