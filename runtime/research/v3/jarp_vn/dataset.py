from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import (ActionFamily, ActionPresence, ActionSubtype, ConversationInput, EmbeddedSpeechAct,
                     Polarity, SpeakerRole, SpeechAct, Turn, VerificationState)


PROHIBITED_PATH_PARTS = ("frozen_pilot", "controlled_pilot", "pilot_first_run")
PROHIBITED_IDS = ("P03", "P09", "P10")
REQUIRED_FIELDS = {
    "item_id", "provenance", "conversation_prefix", "turn_index", "action_presence", "action_family",
    "action_subtype", "requester", "target", "current_speech_act", "embedded_speech_act", "polarity",
    "quoted", "reported", "verification_state", "anchor_turn", "supporting_turns", "evidence_span",
    "difficulty_tags", "contrast_group_id", "contrast_dimension", "topic_family", "abstention_target",
}


def assert_development_path(path: Path) -> None:
    normalized = path.as_posix().casefold()
    if any(part in normalized for part in PROHIBITED_PATH_PARTS):
        raise ValueError("consumed/frozen pilot access is forbidden in JARP-VN development")


def conversation_from_row(row: dict) -> ConversationInput:
    turns = tuple(Turn(int(turn["turn_index"]), SpeakerRole(turn["speaker_role"]), turn["text"])
                  for turn in row["conversation_prefix"])
    return ConversationInput(turns, int(row["turn_index"]))


def validate_row(row: dict) -> None:
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        raise ValueError(f"missing JARP fields: {sorted(missing)}")
    if row["provenance"] != "AI_ASSISTED_SILVER_TRAIN":
        raise ValueError("silver rows must retain AI_ASSISTED_SILVER_TRAIN provenance")
    if any(token in row["item_id"] for token in PROHIBITED_IDS):
        raise ValueError("historical held-out identifier is forbidden")
    conversation = conversation_from_row(row)
    if row["action_presence"] not in {x.value for x in ActionPresence}:
        raise ValueError("invalid action_presence")
    for field, enum in (("action_family", ActionFamily), ("action_subtype", ActionSubtype),
                        ("requester", SpeakerRole), ("target", SpeakerRole),
                        ("current_speech_act", SpeechAct), ("embedded_speech_act", EmbeddedSpeechAct),
                        ("polarity", Polarity), ("verification_state", VerificationState)):
        if row[field] not in {x.value for x in enum}:
            raise ValueError(f"invalid {field}")
    if row["action_presence"] == "NONE" and (row["action_family"] != "NONE" or row["action_subtype"] != "NONE"):
        raise ValueError("NONE action requires NONE family/subtype")
    observable = {turn.turn_index for turn in conversation.turns}
    if row["anchor_turn"] is not None and row["anchor_turn"] not in observable:
        raise ValueError("anchor_turn is not observable")
    if not set(row["supporting_turns"]).issubset(observable):
        raise ValueError("supporting turn exceeds observable prefix")
    if row["anchor_turn"] is not None and row["anchor_turn"] not in row["supporting_turns"]:
        raise ValueError("anchor must be represented in supporting turns")


def validate_contrast_groups(rows: Iterable[dict], minimum_size: int = 4, maximum_size: int = 8) -> dict[str, int]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        validate_row(row)
        groups.setdefault(row["contrast_group_id"], []).append(row)
    for group_id, members in groups.items():
        if not minimum_size <= len(members) <= maximum_size:
            raise ValueError(f"{group_id} size outside [{minimum_size}, {maximum_size}]")
        if len({member["topic_family"] for member in members}) != 1:
            raise ValueError(f"{group_id} does not preserve one topic family")
        if len({(m["polarity"], m["current_speech_act"], m["requester"], m["target"], m["verification_state"])
                for m in members}) < 4:
            raise ValueError(f"{group_id} lacks relation/discourse diversity")
    return {group_id: len(members) for group_id, members in sorted(groups.items())}


def load_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    assert_development_path(source)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        validate_row(row)
    return rows
