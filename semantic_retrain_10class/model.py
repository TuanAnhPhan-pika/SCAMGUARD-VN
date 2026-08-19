from __future__ import annotations
import torch
from torch import nn
from .schema import ACTION_LABELS, SEMANTIC_FIELDS

class JarpV05Semantic(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        width = int(encoder.hidden_size) * 2
        self.dropout = nn.Dropout(0.1)
        self.candidate_action = nn.Linear(width, len(ACTION_LABELS))
        self.semantic_heads = nn.ModuleDict({f: nn.Linear(width, 2) for f in SEMANTIC_FIELDS})

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(batch)
        rep = self.dropout(torch.cat((encoded.prefix, encoded.current), dim=-1))
        return {"candidate_action": self.candidate_action(rep),
                **{f: h(rep) for f, h in self.semantic_heads.items()}}

def decode(outputs: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    probs = {k: v.softmax(-1) for k, v in outputs.items()}
    result = []
    for row in range(outputs["candidate_action"].shape[0]):
        action_id = int(probs["candidate_action"][row].argmax())
        semantics = {f: "YES" if int(probs[f][row].argmax()) else "NO" for f in SEMANTIC_FIELDS}
        candidate = ACTION_LABELS[action_id]
        supported = candidate != "NONE" and semantics == {
            "requested": "YES", "negated": "NO", "quoted": "NO", "reported": "NO"
        }
        result.append({"candidate_action": candidate, **semantics, "supported_action": supported,
            "scores": {"candidate_action": float(probs["candidate_action"][row, action_id]),
                       **{f: float(probs[f][row, int(semantics[f] == "YES")]) for f in SEMANTIC_FIELDS}}})
    return result
