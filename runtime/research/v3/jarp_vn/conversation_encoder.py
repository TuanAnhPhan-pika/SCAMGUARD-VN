from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .turn_encoder import masked_mean, pool_turns


@dataclass
class EncodedConversation:
    prefix: torch.Tensor
    current: torch.Tensor
    turns: torch.Tensor
    turn_mask: torch.Tensor


class TinyBackbone(nn.Module):
    """No-download backbone used only for schema/unit smoke tests."""

    def __init__(self, hidden_size: int = 32, vocab_size: int = 4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(input_ids))


class HuggingFaceBackbone(nn.Module):
    """Lazy teacher backbone; construction is the only operation that may access model files."""

    def __init__(self, model_name: str, revision: str | None = None, local_files_only: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name, revision=revision, local_files_only=local_files_only)
        self.hidden_size = int(self.model.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class SharedConversationEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, max_turns: int):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(backbone.hidden_size)
        self.max_turns = max_turns

    def forward(self, batch: dict[str, torch.Tensor]) -> EncodedConversation:
        token_states = self.backbone(batch["input_ids"], batch["attention_mask"])
        attention = batch["attention_mask"].bool()
        current_mask = batch["current_mask"].bool() & attention
        prefix = masked_mean(token_states, attention, dim=1)
        current = masked_mean(token_states, current_mask, dim=1)
        turns, turn_mask = pool_turns(token_states, batch["turn_ids"], self.max_turns)
        return EncodedConversation(prefix, current, turns, turn_mask)
