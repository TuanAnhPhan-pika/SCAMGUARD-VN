from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
TRANSFER = ROOT / "runtime"
SNAPSHOT = os.environ.get("SCAMGUARD_BASE_MODEL", "xlm-roberta-base")
BASE_CHECKPOINT = Path(os.environ["SCAMGUARD_BASE_CHECKPOINT"]) if os.environ.get("SCAMGUARD_BASE_CHECKPOINT") else None
DATA = Path(os.environ.get("SCAMGUARD_DATA_SPLITS", ROOT / "data/splits"))
OUTPUT = Path(os.environ.get("SCAMGUARD_SEMANTIC_OUTPUT", ROOT / "artifacts/checkpoints"))
LOCAL_FILES_ONLY = os.environ.get("SCAMGUARD_LOCAL_FILES_ONLY", "0") == "1"
sys.path.insert(0, str(TRANSFER))

from research.v3.jarp_vn.collator import TeacherJarpCollator  # noqa: E402
from research.v3.jarp_vn.conversation_encoder import HuggingFaceBackbone, SharedConversationEncoder  # noqa: E402
from research.v3.jarp_vn.schema import ConversationInput, SpeakerRole, Turn  # noqa: E402
from semantic_retrain_10class.model import JarpV05Semantic  # noqa: E402
from semantic_retrain_10class.schema import ACTION_LABELS, SEMANTIC_FIELDS  # noqa: E402


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def conversation(row: dict[str, Any]) -> ConversationInput:
    turns = tuple(Turn(
        turn_index=int(item["turn_index"]),
        speaker_role=SpeakerRole(str(item.get("speaker_role", "UNKNOWN"))),
        text=str(item["text"]),
    ) for item in row["turns"])
    return ConversationInput(turns=turns, current_turn=int(row["observable_turn"]))


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def labels(rows: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "candidate_action": torch.tensor(
            [ACTION_LABELS.index(row["labels"]["candidate_action"]) for row in rows], device=device
        ),
        **{
            field: torch.tensor([int(row["labels"][field] == "YES") for row in rows], device=device)
            for field in SEMANTIC_FIELDS
        },
    }


def sqrt_balanced_weights(rows: list[dict[str, Any]], field: str, classes: tuple[str, ...], device: torch.device) -> torch.Tensor:
    counts = Counter(row["labels"][field] for row in rows)
    total = len(rows)
    raw = [math.sqrt(total / (len(classes) * max(1, counts[name]))) for name in classes]
    mean = sum(raw) / len(raw)
    return torch.tensor([value / mean for value in raw], dtype=torch.float32, device=device)


def loss_for(outputs: dict[str, torch.Tensor], target: dict[str, torch.Tensor], weights: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = {
        name: F.cross_entropy(outputs[name], target[name], weight=weights[name])
        for name in outputs
    }
    return losses["candidate_action"] + 2.0 * losses["requested"] + sum(
        losses[field] for field in ("negated", "quoted", "reported")
    )


def build_model() -> tuple[JarpV05Semantic, TeacherJarpCollator, dict[str, Any]]:
    backbone = HuggingFaceBackbone(str(SNAPSHOT), local_files_only=LOCAL_FILES_ONLY)
    encoder = SharedConversationEncoder(backbone, max_turns=12)
    model = JarpV05Semantic(encoder)
    checkpoint = None
    incompatible = None
    if BASE_CHECKPOINT and BASE_CHECKPOINT.is_file():
        checkpoint = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
        state = checkpoint["model_state"]
        initial = {key: value for key, value in state.items() if key.startswith("encoder.")}
        incompatible = model.load_state_dict(initial, strict=False)
    # Six lower transformer blocks stay fixed; upper six adapt to the new semantic objective.
    for parameter in model.encoder.backbone.model.embeddings.parameters():
        parameter.requires_grad_(False)
    for layer in model.encoder.backbone.model.encoder.layer[:6]:
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
    collator = TeacherJarpCollator.from_pretrained(
        str(SNAPSHOT), revision=None, local_files_only=LOCAL_FILES_ONLY, max_tokens=192, max_turns=12
    )
    metadata = {
        "base_run_id": checkpoint.get("run_id", "RUN004A_ENCODER_ONLY") if checkpoint else "XLM_ROBERTA_BASE",
        "head_initialization": "FRESH_RANDOM_ALL_SEMANTIC_HEADS",
        "base_checkpoint_sha256": hashlib.sha256(BASE_CHECKPOINT.read_bytes()).hexdigest() if BASE_CHECKPOINT and BASE_CHECKPOINT.is_file() else None,
        "missing_keys": incompatible.missing_keys if incompatible else [],
        "unexpected_keys": incompatible.unexpected_keys if incompatible else [],
    }
    return model, collator, metadata


@torch.inference_mode()
def evaluate(model: JarpV05Semantic, collator: TeacherJarpCollator, rows: list[dict[str, Any]], device: torch.device, batch_size: int) -> dict[str, Any]:
    model.eval()
    truth: dict[str, list[int]] = {name: [] for name in ("candidate_action", *SEMANTIC_FIELDS)}
    pred: dict[str, list[int]] = {name: [] for name in truth}
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        output = model(to_device(collator([conversation(row) for row in selected]), device))
        target = labels(selected, device)
        for name in truth:
            truth[name].extend(target[name].cpu().tolist())
            pred[name].extend(output[name].argmax(-1).cpu().tolist())
    per_head = {
        name: {
            "accuracy": accuracy_score(truth[name], pred[name]),
            "macro_f1": f1_score(truth[name], pred[name], average="macro", zero_division=0),
        }
        for name in truth
    }
    true_supported = [
        int(a != 0 and r == 1 and n == q == p == 0)
        for a, r, n, q, p in zip(*(truth[name] for name in ("candidate_action", *SEMANTIC_FIELDS)))
    ]
    pred_supported = [
        int(a != 0 and r == 1 and n == q == p == 0)
        for a, r, n, q, p in zip(*(pred[name] for name in ("candidate_action", *SEMANTIC_FIELDS)))
    ]
    supported = {
        "accuracy": accuracy_score(true_supported, pred_supported),
        "macro_f1": f1_score(true_supported, pred_supported, average="macro", zero_division=0),
    }
    selection_score = (
        0.40 * per_head["candidate_action"]["macro_f1"]
        + 0.30 * per_head["requested"]["macro_f1"]
        + 0.30 * supported["macro_f1"]
    )
    return {"per_head": per_head, "supported_action": supported, "selection_score": selection_score}


def save_checkpoint(path: Path, model: JarpV05Semantic, epoch: int, metrics: dict[str, Any], metadata: dict[str, Any]) -> None:
    value = {
        "format": "JARP_VN_SEMANTIC_RETRAIN_10CLASS_V1",
        "model_state": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
        "epoch": epoch,
        "metrics": metrics,
        "action_labels": ACTION_LABELS,
        "semantic_fields": SEMANTIC_FIELDS,
        **metadata,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for XLM-R semantic training")
    random.seed(40417)
    torch.manual_seed(40417)
    torch.cuda.manual_seed_all(40417)
    device = torch.device("cuda")
    train_rows = load_rows(DATA / "SEMANTIC_TRAIN.jsonl")
    dev_rows = load_rows(DATA / "SEMANTIC_DEV.jsonl")
    if args.smoke:
        train_rows, dev_rows = train_rows[:4], dev_rows[:8]
        args.epochs, args.accumulation = 1, 1
    model, collator, metadata = build_model()
    model.to(device)
    weights = {
        "candidate_action": sqrt_balanced_weights(train_rows, "candidate_action", ACTION_LABELS, device),
        **{
            field: sqrt_balanced_weights(train_rows, field, ("NO", "YES"), device)
            for field in SEMANTIC_FIELDS
        },
    }
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1.5e-5, weight_decay=0.01
    )
    pair_groups: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows:
        if row.get("pair_id"):
            pair_groups.setdefault(row["pair_id"], []).append(row)
    paired_batches = [rows for rows in pair_groups.values() if len(rows) == 2]
    batches_per_epoch = math.ceil(len(train_rows) / args.batch_size) + len(paired_batches)
    optimizer_steps = math.ceil(batches_per_epoch / args.accumulation) * args.epochs
    warmup = max(1, int(optimizer_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(
        (step + 1) / warmup, max(0.0, (optimizer_steps - step) / max(1, optimizer_steps - warmup))
    ))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = random.Random(40417 + epoch)
        rng.shuffle(train_rows)
        rng.shuffle(paired_batches)
        epoch_batches = [train_rows[i:i + args.batch_size] for i in range(0, len(train_rows), args.batch_size)] + paired_batches
        rng.shuffle(epoch_batches)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for batch_index, selected in enumerate(epoch_batches, 1):
            batch = to_device(collator([conversation(row) for row in selected]), device)
            target = labels(selected, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)
                raw_loss = loss_for(output, target, weights)
                if len(selected) == 2 and selected[0].get("pair_id") == selected[1].get("pair_id"):
                    roles = [row.get("pair_role") for row in selected]
                    if set(roles) == {"REQUEST", "NON_REQUEST"}:
                        req_i, non_i = roles.index("REQUEST"), roles.index("NON_REQUEST")
                        score = output["requested"][:, 1] - output["requested"][:, 0]
                        raw_loss = raw_loss + 0.30 * F.relu(1.0 - (score[req_i] - score[non_i]))
                loss = raw_loss / args.accumulation
            loss.backward()
            running += float(loss.detach()) * args.accumulation
            if batch_index % args.accumulation == 0 or batch_index == len(epoch_batches):
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(model, collator, dev_rows, device, max(1, args.batch_size * 2))
        metrics["epoch"] = epoch
        metrics["mean_train_batch_loss"] = running / batches_per_epoch
        metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        if metrics["selection_score"] > best:
            best = metrics["selection_score"]
            save_checkpoint(OUTPUT / ("smoke.pt" if args.smoke else "checkpoint_best.pt"), model, epoch, metrics, metadata)
    report = {
        "status": "SMOKE_PASS" if args.smoke else "TRAINING_COMPLETE",
        "device": torch.cuda.get_device_name(0),
        "train_rows": len(train_rows), "dev_rows": len(dev_rows),
        "history": history, "best_selection_score": best,
    }
    name = "SMOKE_REPORT.json" if args.smoke else "TRAINING_REPORT.json"
    (OUTPUT / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
