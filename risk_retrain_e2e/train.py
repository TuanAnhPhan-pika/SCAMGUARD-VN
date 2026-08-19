from __future__ import annotations
import hashlib,json,math,os,random
from collections import Counter
from pathlib import Path
from typing import Any
import torch
from sklearn.metrics import accuracy_score,classification_report,f1_score
from torch.nn import functional as F
from semantic_retrain_10class.train_semantic import build_model as build_semantic,conversation,to_device
from .model import EndToEndRiskModel,LABELS

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get("SCAMGUARD_DATA_SPLITS",ROOT/"data/splits"))
SEMANTIC_CHECKPOINT=Path(os.environ.get("SCAMGUARD_SEMANTIC_CHECKPOINT",ROOT/"artifacts/checkpoints/checkpoint_best.pt"))
OUTPUT=Path(os.environ.get("SCAMGUARD_RISK_OUTPUT",ROOT/"artifacts/checkpoints")); CHECKPOINT=OUTPUT/"risk_e2e_best.pt"

def load(path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def targets(rows,device): return torch.tensor([LABELS.index(x["labels"]["risk"]) for x in rows],device=device)
def build():
    semantic,collator,_=build_semantic()
    blob=torch.load(SEMANTIC_CHECKPOINT,map_location="cpu",weights_only=False)
    semantic.load_state_dict(blob["model_state"],strict=True)
    model=EndToEndRiskModel(semantic.encoder)
    for p in model.encoder.backbone.model.embeddings.parameters(): p.requires_grad_(False)
    for layer in model.encoder.backbone.model.encoder.layer[:6]:
        for p in layer.parameters(): p.requires_grad_(False)
    return model,collator
@torch.inference_mode()
def evaluate(model,collator,rows,device,batch_size=8):
    model.eval(); truth=[]; pred=[]
    for start in range(0,len(rows),batch_size):
        selected=rows[start:start+batch_size]
        logits=model(to_device(collator([conversation(x) for x in selected]),device))
        truth.extend(targets(selected,device).cpu().tolist()); pred.extend(logits.argmax(-1).cpu().tolist())
    return {"accuracy":accuracy_score(truth,pred),"macro_f1":f1_score(truth,pred,average="macro",zero_division=0),
      "classification_report":classification_report(truth,pred,labels=range(3),target_names=LABELS,zero_division=0,output_dict=True)}
def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    random.seed(11019); torch.manual_seed(11019); torch.cuda.manual_seed_all(11019); device=torch.device("cuda")
    train=load(DATA/"RISK_TRAIN.jsonl"); dev=load(DATA/"RISK_DEV.jsonl")
    model,collator=build(); model.to(device)
    counts=Counter(x["labels"]["risk"] for x in train)
    weights=torch.tensor([len(train)/(3*counts[x]) for x in LABELS],device=device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1.5e-5,weight_decay=0.01)
    batch_size=2; accumulation=8; epochs=3; steps=math.ceil(math.ceil(len(train)/batch_size)/accumulation)*epochs; warmup=max(1,int(.1*steps))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/warmup,max(0.,(steps-s)/max(1,steps-warmup))))
    OUTPUT.mkdir(parents=True,exist_ok=True); best=-1.; history=[]
    for epoch in range(1,epochs+1):
        model.train(); random.Random(11019+epoch).shuffle(train); optimizer.zero_grad(set_to_none=True); total=0.; batches=math.ceil(len(train)/batch_size)
        for bi,start in enumerate(range(0,len(train),batch_size),1):
            selected=train[start:start+batch_size]; batch=to_device(collator([conversation(x) for x in selected]),device)
            with torch.amp.autocast("cuda",dtype=torch.bfloat16): loss=F.cross_entropy(model(batch),targets(selected,device),weight=weights)/accumulation
            loss.backward(); total+=float(loss.detach())*accumulation
            if bi%accumulation==0 or bi==batches:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        metrics=evaluate(model,collator,dev,device); metrics.update({"epoch":epoch,"mean_train_batch_loss":total/batches}); history.append(metrics); print(json.dumps(metrics,ensure_ascii=False),flush=True)
        if metrics["macro_f1"]>best:
            best=metrics["macro_f1"]
            torch.save({"format":"SCAMGUARD_RISK_E2E_RETRAIN_V1","model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},"labels":LABELS,"epoch":epoch,"metrics":metrics,
              "semantic_checkpoint_sha256":hashlib.sha256(SEMANTIC_CHECKPOINT.read_bytes()).hexdigest(),"head_initialization":"FRESH_RANDOM"},CHECKPOINT)
    report={"status":"DEVELOPMENT_TRAINING_COMPLETE","train_rows":len(train),"dev_rows":len(dev),"history":history,"best_macro_f1":best,"test_used":False}
    (OUTPUT/"TRAINING_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__": main()
