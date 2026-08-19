from __future__ import annotations

import torch
from torch.nn import functional as F


def _pairwise_cosine(embeddings: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(embeddings, dim=-1)
    return normalized @ normalized.T


def structured_contrastive_loss(topic_embeddings: torch.Tensor, relation_embeddings: torch.Tensor,
                                topic_ids: torch.Tensor, relation_ids: torch.Tensor,
                                temperature: float = 0.10, margin: float = 0.25) -> tuple[torch.Tensor, torch.Tensor]:
    """Attract shared topics while separating propositionally different relations in a distinct space."""
    same_topic = topic_ids[:, None].eq(topic_ids[None, :])
    same_relation = relation_ids[:, None].eq(relation_ids[None, :])
    eye = torch.eye(len(topic_ids), dtype=torch.bool, device=topic_ids.device)
    topic_pairs = same_topic & ~eye
    relation_positive = same_topic & same_relation & ~eye
    relation_contrast = same_topic & ~same_relation
    topic_similarity = _pairwise_cosine(topic_embeddings)
    relation_similarity = _pairwise_cosine(relation_embeddings)
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    topic_loss = ((1.0 - topic_similarity[topic_pairs]) / temperature).mean() if topic_pairs.any() else topic_similarity.sum() * 0.0
    positive = ((1.0 - relation_similarity[relation_positive]) / temperature).mean() if relation_positive.any() else relation_similarity.sum() * 0.0
    negative = F.relu((relation_similarity[relation_contrast] - margin) / temperature).mean() if relation_contrast.any() else relation_similarity.sum() * 0.0
    return topic_loss, positive + negative
