from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def pool_turns(token_states: torch.Tensor, turn_ids: torch.Tensor, max_turns: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool tokens into observable turn representations; turn id 0 is padding/markers."""
    pooled, valid = [], []
    for turn_position in range(1, max_turns + 1):
        mask = turn_ids.eq(turn_position)
        pooled.append(masked_mean(token_states, mask, dim=1))
        valid.append(mask.any(dim=1))
    return torch.stack(pooled, dim=1), torch.stack(valid, dim=1)
