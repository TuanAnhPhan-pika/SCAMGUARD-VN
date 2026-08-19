from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ACTION_LABELS = (
    "NONE", "TRANSFER_VALUE", "DISCLOSE_SECRET", "INSTALL_SOFTWARE",
    "GRANT_DEVICE_ACCESS", "MOVE_CONVERSATION", "FOLLOW_LINK",
    "SEND_MESSAGE", "PARTICIPATE_TASK", "SELECT_OPTION",
)
BINARY_LABELS = ("NO", "YES")
SEMANTIC_FIELDS = ("requested", "negated", "quoted", "reported")

@dataclass(frozen=True)
class SemanticLabels:
    candidate_action: str
    requested: str
    negated: str
    quoted: str
    reported: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticLabels":
        result = cls(candidate_action=str(value.get("candidate_action", "")),
                     **{f: str(value.get(f, "")).upper() for f in SEMANTIC_FIELDS})
        if result.candidate_action not in ACTION_LABELS:
            raise ValueError(f"invalid candidate_action: {result.candidate_action}")
        for field in SEMANTIC_FIELDS:
            if getattr(result, field) not in BINARY_LABELS:
                raise ValueError(f"{field} must be YES or NO")
        return result

    @property
    def supported(self) -> bool:
        return self.candidate_action != "NONE" and self.requested == "YES" and all(
            getattr(self, f) == "NO" for f in ("negated", "quoted", "reported")
        )
