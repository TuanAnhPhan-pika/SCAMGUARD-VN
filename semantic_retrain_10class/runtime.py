from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import decode
from .train_semantic import OUTPUT, build_model, conversation, to_device


class SemanticRuntime:
    def __init__(self, checkpoint_path: Path | None = None, device: str = "auto"):
        self.checkpoint_path = checkpoint_path or OUTPUT / "checkpoint_best.pt"
        requested = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.device = torch.device(requested)
        self.model, self.collator, _ = build_model()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "JARP_VN_SEMANTIC_RETRAIN_10CLASS_V1":
            raise ValueError("unsupported semantic checkpoint format")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.to(self.device).eval()
        self.metadata = {
            "model": "JARP-VN SEMANTIC RETRAIN 10CLASS V1",
            "checkpoint_epoch": checkpoint["epoch"],
            "device": str(self.device),
        }

    @torch.inference_mode()
    def analyze_turns(self, turns: list[dict[str, Any]], observable_turn: int | None = None) -> dict[str, Any]:
        normalized = []
        for index, turn in enumerate(turns, 1):
            normalized.append({
                "turn_index": int(turn.get("turn_index", index)),
                "speaker_role": str(turn.get("speaker_role", "OTHER_PARTY")),
                "text": str(turn["text"]),
            })
        row = {"turns": normalized, "observable_turn": observable_turn or normalized[-1]["turn_index"]}
        batch = to_device(self.collator([conversation(row)]), self.device)
        result = decode(self.model(batch))[0]
        return {**self.metadata, **result}

    def analyze_text(self, text: str) -> dict[str, Any]:
        return self.analyze_turns([{"speaker_role": "OTHER_PARTY", "text": text}])
