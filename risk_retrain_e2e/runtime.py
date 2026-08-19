from __future__ import annotations
from pathlib import Path
import torch
import re
from semantic_retrain_10class.runtime import SemanticRuntime
from semantic_retrain_10class.train_semantic import conversation,to_device
from .model import EndToEndRiskModel,LABELS
from .train import CHECKPOINT,build

class EndToEndRiskRuntime:
    def __init__(self,device="auto"):
        self.semantic=SemanticRuntime(device=device); self.device=self.semantic.device
        self.model,self.collator=build(); blob=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
        if blob.get("format")!="SCAMGUARD_RISK_E2E_RETRAIN_V1": raise ValueError("unsupported risk checkpoint")
        self.model.load_state_dict(blob["model_state"],strict=True); self.model.to(self.device).eval(); self.epoch=blob["epoch"]
    @torch.inference_mode()
    def analyze_turns(self,turns):
        normalized=[{"turn_index":i,"speaker_role":str(x.get("speaker_role","OTHER_PARTY")),"text":str(x["text"])} for i,x in enumerate(turns,1)]
        row={"turns":normalized,"observable_turn":len(normalized)}; batch=to_device(self.collator([conversation(row)]),self.device)
        probs=self.model(batch).softmax(-1)[0]; pred=int(probs.argmax())
        return {"risk_level":LABELS[pred],"risk_confidence":float(probs[pred]),"risk_scores":{x:float(probs[i]) for i,x in enumerate(LABELS)},"risk_model":"SCAMGUARD RISK E2E RETRAIN V1","risk_checkpoint_epoch":self.epoch}
    def analyze(self,text): return self.analyze_turns(parse_dialogue_text(text))

def parse_dialogue_text(text: str):
    """Parse common Vietnamese chat prefixes; plain text remains one incoming turn."""
    pattern = re.compile(r"^(khách hàng|người dùng|tôi|user|nhân viên hỗ trợ|nhân viên|hệ thống|system)\s*:\s*(.*)$", re.I)
    turns = []
    for line in (item.strip() for item in text.splitlines() if item.strip()):
        match = pattern.match(line)
        if match:
            role = "USER" if match.group(1).lower() in {"khách hàng", "người dùng", "tôi", "user"} else "OTHER_PARTY"
            turns.append({"speaker_role": role, "text": match.group(2)})
        elif turns:
            turns[-1]["text"] += "\n" + line
        else:
            turns.append({"speaker_role": "OTHER_PARTY", "text": line})
    return turns or [{"speaker_role": "OTHER_PARTY", "text": text}]
