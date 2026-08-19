from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import JarpConfig
from .contrastive import structured_contrastive_loss
from .heads import HeadOutputs
from .schema import LABELS


def conditional_risky_family_ce(action_logits: torch.Tensor, action_targets: torch.Tensor) -> torch.Tensor:
    """Seven-way CE on gold-risky rows using the existing eight-way ActionFamily logits."""
    none_index = LABELS["action_family"].index("NONE")
    risky_mask = action_targets.ne(none_index)
    if not bool(risky_mask.any()):
        return action_logits.sum() * 0.0
    risky_columns = [index for index in range(action_logits.shape[-1]) if index != none_index]
    risky_logits = action_logits[risky_mask][:, risky_columns]
    risky_targets = action_targets[risky_mask] - (action_targets[risky_mask] > none_index).long()
    return F.cross_entropy(risky_logits, risky_targets)


def structural_consistency_loss(raw: HeadOutputs) -> torch.Tensor:
    presence_none = raw.categorical["action_presence"].softmax(-1)[:, LABELS["action_presence"].index("NONE")]
    family_none = raw.categorical["action_family"].softmax(-1)[:, LABELS["action_family"].index("NONE")]
    subtype_none = raw.categorical["action_subtype"].softmax(-1)[:, LABELS["action_subtype"].index("NONE")]
    reported = raw.categorical["reported"].softmax(-1)[:, LABELS["reported"].index("TRUE")]
    current_request = raw.categorical["current_speech_act"].softmax(-1)[:, LABELS["current_speech_act"].index("REQUEST")]
    current_report = raw.categorical["current_speech_act"].softmax(-1)[:, LABELS["current_speech_act"].index("REPORT")]
    identity = (presence_none - family_none).abs().mean() + (presence_none - subtype_none).abs().mean()
    discourse = (reported * current_request * (1.0 - current_report)).mean()
    return identity + discourse


def multitask_loss(raw: HeadOutputs, labels: dict[str, torch.Tensor], config: JarpConfig) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    for name, logits in raw.categorical.items():
        losses[name] = F.cross_entropy(logits, labels[name])
    losses["anchor_turn"] = F.cross_entropy(raw.anchor_turn, labels["anchor_turn"], ignore_index=-100)
    support_loss = F.binary_cross_entropy_with_logits(raw.supporting_turns, labels["supporting_turns"].float(), reduction="none")
    losses["supporting_turns"] = (support_loss * raw.turn_mask.float()).sum() / raw.turn_mask.float().sum().clamp_min(1.0)
    losses["conditional_risky_family"] = conditional_risky_family_ce(raw.categorical["action_family"], labels["action_family"])
    losses["structural_consistency"] = structural_consistency_loss(raw)
    topic, relation = structured_contrastive_loss(raw.topic_embedding, raw.relation_embedding,
                                                   labels["topic_id"], labels["relation_id"], config.contrastive_temperature)
    losses["contrastive_topic"] = topic
    losses["contrastive_relation"] = relation
    primary_contributions = {
        name: config.task_weights[name] * losses[name] * (config.action_family_loss_weight if name == "action_family" else 1.0)
        for name in config.task_weights
    }
    total = sum(primary_contributions.values())
    total = total + config.structural_consistency_weight * losses["structural_consistency"]
    total = total + config.contrastive_topic_weight * topic + config.contrastive_relation_weight * relation
    weighted_conditional = config.conditional_risky_family_loss_weight * losses["conditional_risky_family"]
    total = total + weighted_conditional
    losses["weighted_action_family_contribution"] = primary_contributions["action_family"]
    losses["weighted_conditional_risky_contribution"] = weighted_conditional
    if config.conditional_risky_family_loss_weight > 0.0:
        losses["total_action_related_contribution"] = primary_contributions["action_family"] + weighted_conditional
    losses["total"] = total
    return losses
