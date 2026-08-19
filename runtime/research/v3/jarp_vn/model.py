from __future__ import annotations

from torch import nn

from .config import JarpConfig
from .conversation_encoder import HuggingFaceBackbone, SharedConversationEncoder, TinyBackbone
from .heads import HeadOutputs, JointPredictionHeads


class JointActionRelationModel(nn.Module):
    def __init__(self, config: JarpConfig, backbone: nn.Module | None = None):
        super().__init__()
        self.config = config
        backbone = backbone or TinyBackbone(hidden_size=config.hidden_size)
        self.encoder = SharedConversationEncoder(backbone, config.max_turns)
        self.heads = JointPredictionHeads(self.encoder.hidden_size, config.dropout)

    def forward(self, batch: dict) -> HeadOutputs:
        return self.heads(self.encoder(batch))

    @classmethod
    def from_teacher(cls, config: JarpConfig, source: str | None = None,
                     revision: str | None = None) -> "JointActionRelationModel":
        source = source or config.encoder_name
        revision = config.encoder_revision if source == config.encoder_name and revision is None else revision
        backbone = HuggingFaceBackbone(source, revision, config.local_files_only)
        return cls(config, backbone)
