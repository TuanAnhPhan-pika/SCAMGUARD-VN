from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_TASK_WEIGHTS = {
    "action_presence": 1.0,
    "action_family": 1.0,
    "action_subtype": 1.0,
    "requester": 1.0,
    "target": 1.0,
    "current_speech_act": 1.0,
    "embedded_speech_act": 1.0,
    "polarity": 1.0,
    "quoted": 1.0,
    "reported": 1.0,
    "verification_state": 1.0,
    "anchor_turn": 1.0,
    "supporting_turns": 1.0,
}


@dataclass(frozen=True)
class JarpConfig:
    encoder_name: str = "xlm-roberta-base"
    encoder_revision: str | None = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
    teacher_snapshot_path: str = "models/v3/teachers/xlm_roberta_base/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
    hidden_size: int = 768
    max_turns: int = 12
    max_tokens: int = 384
    dropout: float = 0.1
    task_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TASK_WEIGHTS))
    action_family_loss_weight: float = 1.0
    conditional_risky_family_loss_weight: float = 0.0
    structural_consistency_weight: float = 0.25
    contrastive_relation_weight: float = 0.20
    contrastive_topic_weight: float = 0.10
    contrastive_temperature: float = 0.10
    abstention_threshold: float = 0.55
    local_files_only: bool = True
    confidence_status: str = "UNCALIBRATED_DEVELOPMENT_SCORE"
    random_seed: int = 1502
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    batch_size: int = 12
    epochs: int = 3
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        if set(self.task_weights) != set(DEFAULT_TASK_WEIGHTS):
            raise ValueError("task_weights must explicitly cover all thirteen heads")
        if self.action_family_loss_weight <= 0:
            raise ValueError("action_family_loss_weight must be positive")
        if self.conditional_risky_family_loss_weight < 0:
            raise ValueError("conditional_risky_family_loss_weight must be non-negative")
        if not 0.0 <= self.abstention_threshold <= 1.0:
            raise ValueError("abstention_threshold must be within [0, 1]")
        if self.max_turns < 1 or self.max_tokens < 1:
            raise ValueError("max_turns and max_tokens must be positive")
        if self.learning_rate <= 0 or self.batch_size < 1 or self.epochs < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("training hyperparameters must be positive")
