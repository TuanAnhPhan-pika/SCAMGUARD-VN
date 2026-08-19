from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .conversation_encoder import EncodedConversation
from .schema import LABELS


@dataclass
class HeadOutputs:
    categorical: dict[str, torch.Tensor]
    anchor_turn: torch.Tensor
    supporting_turns: torch.Tensor
    topic_embedding: torch.Tensor
    relation_embedding: torch.Tensor
    turn_mask: torch.Tensor


class JointPredictionHeads(nn.Module):
    """Thirteen heads over one shared conversational representation."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        joint_size = hidden_size * 2
        self.dropout = nn.Dropout(dropout)
        self.categorical = nn.ModuleDict({name: nn.Linear(joint_size, len(labels)) for name, labels in LABELS.items()})
        self.anchor_none = nn.Linear(joint_size, 1)
        self.anchor_query = nn.Linear(joint_size, hidden_size)
        self.support_query = nn.Linear(joint_size, hidden_size)
        self.topic_projection = nn.Linear(joint_size, hidden_size)
        self.relation_projection = nn.Linear(joint_size, hidden_size)

    def forward(self, encoded: EncodedConversation) -> HeadOutputs:
        shared = self.dropout(torch.cat((encoded.prefix, encoded.current), dim=-1))
        categorical = {name: head(shared) for name, head in self.categorical.items()}
        anchor_scores = torch.einsum("bth,bh->bt", encoded.turns, self.anchor_query(shared))
        anchor_scores = anchor_scores.masked_fill(~encoded.turn_mask, float("-inf"))
        anchor = torch.cat((self.anchor_none(shared), anchor_scores), dim=1)
        supporting = torch.einsum("bth,bh->bt", encoded.turns, self.support_query(shared))
        supporting = supporting.masked_fill(~encoded.turn_mask, -30.0)
        return HeadOutputs(categorical, anchor, supporting, self.topic_projection(shared),
                           self.relation_projection(shared), encoded.turn_mask)
