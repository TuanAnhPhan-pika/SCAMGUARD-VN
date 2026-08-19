from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "evaluation/independent_test_200"
EXPECTED = {
    "FINAL_MODEL_INPUT.jsonl": "bf28821ad7a00fdecdbf323a6fe4db26a74ecf21c7a4a63f9194f25682bc3d87",
    "predictions_frozen.jsonl": "67aba49b6fb118d6b5c5ba4e335442a6bba7e5842706fe3477609bddf2c311cf",
    "INTERNAL_GOLD.jsonl": "40807e46939000a487e5975b5b51942a11ffffb1a4ced3e66a5fa62b5fa7d2ef",
}
CHECKPOINTS = {
    "checkpoint_best.pt": "ae0420e7efbdbe791b71907131c480e6fcb9d7cef9f1e4f6382cdc392eb66019",
    "risk_e2e_best.pt": "824b58d32366d746adc32b4350c73cad49054f538d365f6661da4036535c8328",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", action="store_true")
    args = parser.parse_args()
    for name, expected in EXPECTED.items():
        actual = sha(TEST / name)
        if actual != expected:
            raise SystemExit(f"HASH MISMATCH: {name}: {actual}")
        print(f"OK {name}: {actual}")
    if args.checkpoints:
        for name, expected in CHECKPOINTS.items():
            path = ROOT / "artifacts/checkpoints" / name
            actual = sha(path)
            if actual != expected:
                raise SystemExit(f"HASH MISMATCH: {name}: {actual}")
            print(f"OK {name}: {actual}")
    predictions = [json.loads(line) for line in (TEST / "predictions_frozen.jsonl").read_text(encoding="utf-8").splitlines() if line]
    gold = [json.loads(line) for line in (TEST / "INTERNAL_GOLD.jsonl").read_text(encoding="utf-8").splitlines() if line]
    gold_by_id = {row["case_id"]: row for row in gold}
    truth = [gold_by_id[row["case_id"]]["gold_risk"] for row in predictions]
    pred = [row["prediction"] for row in predictions]
    supported_labels = sorted(set(truth))
    print(json.dumps({
        "cases": len(truth),
        "gold_distribution": Counter(truth),
        "strict_accuracy": accuracy_score(truth, pred),
        "macro_f1_supported_labels": f1_score(truth, pred, labels=supported_labels, average="macro", zero_division=0),
        "status": "VERIFIED_READ_ONLY",
    }, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()

