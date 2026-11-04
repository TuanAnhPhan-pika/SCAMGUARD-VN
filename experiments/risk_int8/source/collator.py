from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import torch

from .schema import ConversationInput


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def serialize_conversation(conversation: ConversationInput) -> str:
    parts = []
    for turn in conversation.turns:
        current = "<CURRENT>" if turn.turn_index == conversation.current_turn else ""
        parts.append(f"<T{turn.turn_index}><{turn.speaker_role.value}>{current} {turn.text}")
    return "\n".join(parts)


@dataclass(frozen=True)
class HashedTokenizer:
    """Deterministic smoke tokenizer; it is not the research teacher tokenizer."""

    vocab_size: int = 4096

    def token_id(self, token: str) -> int:
        return int.from_bytes(hashlib.sha256(token.casefold().encode("utf-8")).digest()[:4], "big") % (self.vocab_size - 1) + 1


class JarpCollator:
    def __init__(self, max_tokens: int = 384, max_turns: int = 12, tokenizer: HashedTokenizer | None = None):
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.tokenizer = tokenizer or HashedTokenizer()

    def _turn_chunks(self, conversation: ConversationInput) -> list[tuple[int, list[int]]]:
        chunks = []
        for turn in conversation.turns:
            markers = [f"<T{turn.turn_index}>", f"<{turn.speaker_role.value}>"]
            if turn.turn_index == conversation.current_turn:
                markers.append("<CURRENT>")
            tokens = markers + TOKEN_RE.findall(turn.text)
            chunks.append((turn.turn_index, [self.tokenizer.token_id(token) for token in tokens]))
        return chunks

    @staticmethod
    def _retain(chunks: list[tuple[int, list[int]]], current_turn: int, max_turns: int, token_budget: int) -> list[tuple[int, list[int]]]:
        if not chunks or chunks[-1][0] != current_turn:
            raise ValueError("current turn must be the final observable chunk")
        chunks = chunks[-max_turns:]
        current_index, current_ids = chunks[-1]
        current_ids = current_ids[:token_budget]
        retained = [(current_index, current_ids)]
        remaining = token_budget - len(current_ids)
        for index, ids in reversed(chunks[:-1]):
            if len(ids) > remaining:
                break
            retained.append((index, ids)); remaining -= len(ids)
        return list(reversed(retained))

    def _encode(self, conversation: ConversationInput) -> tuple[list[int], list[int], list[int], list[int]]:
        retained = self._retain(self._turn_chunks(conversation), conversation.current_turn, self.max_turns, self.max_tokens)
        ids, turns, current = [], [], []
        for position, (turn_index, token_ids) in enumerate(retained, start=1):
            ids.extend(token_ids); turns.extend([position] * len(token_ids))
            current.extend([int(turn_index == conversation.current_turn)] * len(token_ids))
        return ids, turns, current, [index for index, _ in retained]

    def __call__(self, conversations: list[ConversationInput]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(item) for item in conversations]
        width = max(len(item[0]) for item in encoded)
        batch = {"input_ids": [], "turn_ids": [], "current_mask": [], "attention_mask": []}
        retained_turn_indexes = []
        for ids, turns, current, retained in encoded:
            pad = width - len(ids)
            batch["input_ids"].append(ids + [0] * pad)
            batch["turn_ids"].append(turns + [0] * pad)
            batch["current_mask"].append(current + [0] * pad)
            batch["attention_mask"].append([1] * len(ids) + [0] * pad)
            retained_turn_indexes.append(retained)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()} | {
            "retained_turn_indexes": retained_turn_indexes,
        }


class TeacherJarpCollator:
    """Collator aligned to the configured HuggingFace teacher tokenizer."""

    def __init__(self, tokenizer, max_tokens: int = 384, max_turns: int = 12):
        self.tokenizer = tokenizer; self.max_tokens = max_tokens; self.max_turns = max_turns
        special = int(tokenizer.num_special_tokens_to_add(pair=False))
        self.core_budget = max_tokens - special
        if self.core_budget < 1:
            raise ValueError("max_tokens leaves no room for current-turn tokens")

    @classmethod
    def from_pretrained(cls, model_name: str, revision: str | None, local_files_only: bool,
                        max_tokens: int = 384, max_turns: int = 12) -> "TeacherJarpCollator":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=local_files_only,
                                                  use_fast=True)
        return cls(tokenizer, max_tokens, max_turns)

    def metadata(self) -> dict:
        return {"name_or_path": self.tokenizer.name_or_path, "class": type(self.tokenizer).__name__,
                "vocab_size": len(self.tokenizer), "model_max_length": self.tokenizer.model_max_length,
                "max_tokens": self.max_tokens, "local_files_only_policy": True}

    def _chunks(self, conversation: ConversationInput) -> list[tuple[int, list[int]]]:
        chunks = []
        for turn in conversation.turns:
            marker = f"<T{turn.turn_index}><{turn.speaker_role.value}>"
            if turn.turn_index == conversation.current_turn:
                marker += "<CURRENT>"
            chunks.append((turn.turn_index, self.tokenizer.encode(f"{marker} {turn.text}", add_special_tokens=False)))
        return chunks

    @staticmethod
    def _special_mask(input_ids: list[int], core_ids: list[int]) -> list[int]:
        """Map the unchanged core sequence inside a tokenizer-wrapped sequence."""
        width = len(core_ids)
        for offset in range(len(input_ids) - width + 1):
            if input_ids[offset:offset + width] == core_ids:
                return [1] * offset + [0] * width + [1] * (len(input_ids) - offset - width)
        raise ValueError("teacher tokenizer did not preserve the core token sequence")

    def __call__(self, conversations: list[ConversationInput]) -> dict:
        examples = []
        for conversation in conversations:
            retained = JarpCollator._retain(self._chunks(conversation), conversation.current_turn,
                                            self.max_turns, self.core_budget)
            flat = [token for _, ids in retained for token in ids]
            core_turns = [position for position, (_, ids) in enumerate(retained, start=1) for _ in ids]
            core_current = [int(index == conversation.current_turn) for index, ids in retained for _ in ids]
            input_ids = self.tokenizer.build_inputs_with_special_tokens(flat)
            special_mask = self._special_mask(input_ids, flat)
            turn_ids, current_mask, cursor = [], [], 0
            for special in special_mask:
                if special:
                    turn_ids.append(0); current_mask.append(0)
                else:
                    turn_ids.append(core_turns[cursor]); current_mask.append(core_current[cursor]); cursor += 1
            examples.append((input_ids, turn_ids, current_mask, [index for index, _ in retained]))
        width = max(len(item[0]) for item in examples); pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("teacher tokenizer must define pad_token_id")
        batch = {"input_ids": [], "turn_ids": [], "current_mask": [], "attention_mask": []}; retained = []
        for ids, turns, current, indexes in examples:
            pad = width - len(ids)
            batch["input_ids"].append(ids + [pad_id] * pad); batch["turn_ids"].append(turns + [0] * pad)
            batch["current_mask"].append(current + [0] * pad); batch["attention_mask"].append([1] * len(ids) + [0] * pad)
            retained.append(indexes)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()} | {
            "retained_turn_indexes": retained,
        }
