from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from risk_retrain_e2e.runtime import EndToEndRiskRuntime, parse_dialogue_text

try:
    from .guidance import generate_reply
except ImportError:  # pragma: no cover
    generate_reply = None


app = FastAPI(title="ScamGuard-VN local API", version="1.0.0")
_runtime: EndToEndRiskRuntime | None = None


class TurnRequest(BaseModel):
    speaker_role: Literal["USER", "OTHER_PARTY", "SYSTEM"]
    text: str = Field(min_length=1, max_length=50_000)


class AnalyzeRequest(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=50_000)
    turns: list[TurnRequest] | None = None
    observable_turn: int | None = None

    @model_validator(mode="after")
    def validate_input(self):
        if bool(self.text) == bool(self.turns):
            raise ValueError("provide exactly one of text or turns")
        return self


def runtime() -> EndToEndRiskRuntime:
    global _runtime
    if _runtime is None:
        _runtime = EndToEndRiskRuntime(device=os.environ.get("SCAMGUARD_DEVICE", "auto"))
    return _runtime


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _runtime is not None, "rules_enabled": False}


@app.post("/v1/analyze-action")
def analyze(request: AnalyzeRequest):
    try:
        turns = parse_dialogue_text(request.text) if request.text else [item.model_dump() for item in request.turns or []]
        model = runtime()
        semantic = model.semantic.analyze_turns(turns, request.observable_turn or len(turns))
        result = {**semantic, **model.analyze_turns(turns)}
        chat = request.text or "\n".join(f"{item['speaker_role']}: {item['text']}" for item in turns)
        result.update({
            "model": "SCAMGUARD-VN XLM-R SEMANTIC + RISK E2E",
            "candidate_text": turns[-1]["text"],
            "input_view": "RAW_PLUS_ACTION_CANDIDATE_TEXT",
            "rules_enabled": False,
        })
        result["guidance"] = generate_reply(chat, result["risk_level"], result) if generate_reply else {
            "status": "disabled", "handling": None, "reply_mode": None, "reply": None
        }
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
