from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class StrEnum(str, Enum):
    pass


class SpeakerRole(StrEnum):
    USER = "USER"
    OTHER_PARTY = "OTHER_PARTY"
    SYSTEM = "SYSTEM"
    UNKNOWN = "UNKNOWN"


class ActionPresence(StrEnum):
    NONE = "NONE"
    ACTION_PRESENT = "ACTION_PRESENT"


class ActionFamily(StrEnum):
    NONE = "NONE"
    PAYMENT = "PAYMENT"
    OTP_DISCLOSURE = "OTP_DISCLOSURE"
    CREDENTIAL_DISCLOSURE = "CREDENTIAL_DISCLOSURE"
    QR_PAYMENT = "QR_PAYMENT"
    APK_INSTALL = "APK_INSTALL"
    REMOTE_ACCESS = "REMOTE_ACCESS"
    PLATFORM_MIGRATION = "PLATFORM_MIGRATION"


class ActionSubtype(StrEnum):
    NONE = "NONE"
    GENERAL = "GENERAL"
    DEPOSIT = "DEPOSIT"
    ADDITIONAL_PAYMENT = "ADDITIONAL_PAYMENT"


class SpeechAct(StrEnum):
    REQUEST = "REQUEST"
    INSTRUCTION = "INSTRUCTION"
    WARNING = "WARNING"
    REPORT = "REPORT"
    INFORMATION = "INFORMATION"
    UNKNOWN = "UNKNOWN"


class EmbeddedSpeechAct(StrEnum):
    REQUEST = "REQUEST"
    INSTRUCTION = "INSTRUCTION"
    WARNING = "WARNING"
    REPORT = "REPORT"
    INFORMATION = "INFORMATION"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class Polarity(StrEnum):
    AFFIRMATIVE = "AFFIRMATIVE"
    NEGATED = "NEGATED"
    UNKNOWN = "UNKNOWN"


class VerificationState(StrEnum):
    SUPPORTED = "SUPPORTED"
    NEGATED = "NEGATED"
    UNCLEAR = "UNCLEAR"
    CONTRADICTED = "CONTRADICTED"


LABELS = {
    "action_presence": tuple(x.value for x in ActionPresence),
    "action_family": tuple(x.value for x in ActionFamily),
    "action_subtype": tuple(x.value for x in ActionSubtype),
    "requester": tuple(x.value for x in SpeakerRole),
    "target": tuple(x.value for x in SpeakerRole),
    "current_speech_act": tuple(x.value for x in SpeechAct),
    "embedded_speech_act": tuple(x.value for x in EmbeddedSpeechAct),
    "polarity": tuple(x.value for x in Polarity),
    "quoted": ("TRUE", "FALSE"),
    "reported": ("TRUE", "FALSE"),
    "verification_state": tuple(x.value for x in VerificationState),
}


@dataclass(frozen=True)
class Turn:
    turn_index: int
    speaker_role: SpeakerRole
    text: str

    def __post_init__(self) -> None:
        if self.turn_index < 1 or not self.text.strip():
            raise ValueError("turn requires a positive index and non-empty raw text")


@dataclass(frozen=True)
class ConversationInput:
    turns: tuple[Turn, ...]
    current_turn: int

    def __post_init__(self) -> None:
        indexes = [turn.turn_index for turn in self.turns]
        if not indexes or indexes != sorted(set(indexes)):
            raise ValueError("turn indexes must be unique and increasing")
        if indexes[-1] != self.current_turn:
            raise ValueError("observable prefix must end at current_turn; future turns are forbidden")


@dataclass(frozen=True)
class JarpOutput:
    action_present: bool
    action_family: str
    action_subtype: str
    requester: str
    target: str
    current_speech_act: str
    embedded_speech_act: str
    polarity: str
    quoted: bool
    reported: bool
    verification_state: str
    anchor_turn: int | None
    supporting_turns: tuple[int, ...]
    confidence: float | None
    abstained: bool
    evidence_span: str
    safe_action_codes: tuple[str, ...] = ()
    confidence_status: str = "UNCALIBRATED_DEVELOPMENT_SCORE"

    def as_android_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supporting_turns"] = list(self.supporting_turns)
        value["safe_action_codes"] = list(self.safe_action_codes)
        return value
