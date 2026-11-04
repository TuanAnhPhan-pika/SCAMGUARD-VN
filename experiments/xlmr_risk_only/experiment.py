"""Isolated risk-only training; public artifacts are read-only references."""
from __future__ import annotations
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PUBLIC = ROOT
OUT = HERE / 'outputs'
BASE = Path(os.environ.get('SCAMGUARD_BASE_MODEL', ROOT / 'artifacts/base/xlm-roberta-base'))
DATA = PUBLIC / 'data/splits'
TEST = PUBLIC / 'evaluation/independent_test_200'
SEED = 11019
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, str(PUBLIC / 'runtime'))
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from research.v3.jarp_vn.conversation_encoder import HuggingFaceBackbone, SharedConversationEncoder
from research.v3.jarp_vn.collator import TeacherJarpCollator
from research.v3.jarp_vn.schema import ConversationInput, SpeakerRole, Turn

# This module contains only the risk classifier architecture, no semantic imports.
spec = importlib.util.spec_from_file_location('reference_risk_architecture', PUBLIC / 'risk_retrain_e2e/model.py')
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)
LABELS = arch.LABELS
torch.set_num_threads(4)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(target)

def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def conv(row):
    return ConversationInput(turns=tuple(Turn(turn_index=int(t.get('turn_index', i)),
        speaker_role=SpeakerRole(t.get('speaker_role', 'OTHER_PARTY')), text=t['text'])
        for i,t in enumerate(row['turns'],1)), current_turn=int(row.get('observable_turn',len(row['turns']))))

def batch(collator, selected):
    return {k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in collator([conv(x) for x in selected]).items()}

def targets(selected):
    return torch.tensor([LABELS.index(x['labels']['risk']) for x in selected],device='cuda')

def build():
    # Load ONLY the pinned, original pretrained XLM-R weights.
    encoder = SharedConversationEncoder(HuggingFaceBackbone(str(BASE), local_files_only=True),max_turns=12)
    model = arch.EndToEndRiskModel(encoder)
    for p in model.encoder.backbone.model.embeddings.parameters(): p.requires_grad_(False)
    for layer in model.encoder.backbone.model.encoder.layer[:6]:
        for p in layer.parameters(): p.requires_grad_(False)
    collator = TeacherJarpCollator.from_pretrained(str(BASE),None,True,192,12)
    return model.to('cuda'),collator

def protected_files():
    files = list((PUBLIC/'data/splits').glob('*.jsonl'))
    files += list(TEST.glob('*'))
    files += list((PUBLIC/'artifacts/checkpoints').glob('*.pt'))
    for folder in ('risk_retrain_e2e','semantic_retrain_10class','runtime/research/v3/jarp_vn'):
        files += list((PUBLIC/folder).glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file()}

@torch.inference_mode()
def evaluate(model,collator,selected):
    model.eval(); truth=[]; predictions=[]
    for start in range(0,len(selected),8):
        group=selected[start:start+8]
        pred=model(batch(collator,group)).argmax(-1).cpu().tolist()
        predictions.extend(pred); truth.extend(targets(group).cpu().tolist())
    return {'accuracy':accuracy_score(truth,predictions),
        'macro_f1':f1_score(truth,predictions,average='macro',zero_division=0),
        'classification_report':classification_report(truth,predictions,labels=list(range(3)),target_names=LABELS,output_dict=True,zero_division=0)}

def train():
    if (OUT/'TRAINING_REPORT.json').exists(): raise RuntimeError('Training already complete')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    hashes = protected_files()
    base_hash = sha(BASE/'model.safetensors')
    if base_hash != '6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb':
        raise RuntimeError('Unexpected original XLM-R weights')
    train_rows=rows(DATA/'RISK_TRAIN.jsonl'); dev=rows(DATA/'RISK_DEV.jsonl')
    assert (len(train_rows),len(dev))==(3805,398)
    # Do not use action/semantic labels, even though the source rows contain them.
    train_rows=[{'turns':x['turns'],'observable_turn':x['observable_turn'],'labels':{'risk':x['labels']['risk']}} for x in train_rows]
    dev=[{'turns':x['turns'],'observable_turn':x['observable_turn'],'labels':{'risk':x['labels']['risk']}} for x in dev]
    write('PRETRAIN_MANIFEST.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'initialization':'ORIGINAL_XLM_ROBERTA_BASE_NO_RUN004A_NO_SEMANTIC',
        'base_sha256':base_hash,'protected_files':hashes,'seed':SEED,'train_rows':3805,'dev_rows':398,
        'risk_labels':LABELS,'semantic_labels_used':False,'epochs':3,'batch_size':2,'accumulation':8,
        'learning_rate':1.5e-5,'weight_decay':0.01,'max_tokens':192,'max_turns':12,
        'frozen':'embeddings and lower six transformer blocks','selection':'highest DEV macro F1, earlier epoch on ties',
        'test_policy':'test not used in training or checkpoint selection; comparative re-evaluation of existing 200',
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__})
    random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model,collator=build()
    counts=Counter(x['labels']['risk'] for x in train_rows)
    weights=torch.tensor([len(train_rows)/(3*counts[x]) for x in LABELS],device='cuda')
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1.5e-5,weight_decay=.01)
    batches=math.ceil(len(train_rows)/2); steps=math.ceil(batches/8)*3; warmup=max(1,int(.1*steps))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/warmup,max(0.,(steps-s)/max(1,steps-warmup))))
    best=-1.;history=[]; started=time.monotonic()
    print('TRAINING_START '+json.dumps({'batches_per_epoch':batches,'gpu':torch.cuda.get_device_name()}),flush=True)
    for epoch in range(1,4):
        model.train();random.Random(SEED+epoch).shuffle(train_rows);optimizer.zero_grad(set_to_none=True);total=0.
        for bi,start in enumerate(range(0,len(train_rows),2),1):
            selected=train_rows[start:start+2]
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                loss=F.cross_entropy(model(batch(collator,selected)),targets(selected),weight=weights)/8
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            loss.backward();total+=float(loss.detach())*8
            if bi%8==0 or bi==batches:
                torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            if bi%100==0 or bi==batches:
                progress={'epoch':epoch,'batch':bi,'batches':batches,'loss':total/bi,'elapsed_seconds':round(time.monotonic()-started,1)}
                write('PROGRESS.json',progress); print(json.dumps(progress),flush=True)
        metrics=evaluate(model,collator,dev);metrics.update(epoch=epoch,mean_train_batch_loss=total/batches);history.append(metrics)
        if metrics['macro_f1']>best:
            best=metrics['macro_f1']
            torch.save({'format':'XLMR_RISK_ONLY_BASELINE_V1','model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'epoch':epoch,'metrics':metrics,'labels':LABELS,'base_sha256':base_hash,'semantic_pretraining':False},OUT/'baseline_best.pt')
        write('TRAINING_HISTORY.json',history);print('DEV '+json.dumps(metrics),flush=True)
    assert protected_files()==hashes,'Public reference modified'
    write('TRAINING_REPORT.json',{'status':'COMPLETE','history':history,'best_macro_f1':best,'test_used':False,
        'train_rows':len(train_rows),'dev_rows':len(dev),'public_unchanged':True,'elapsed_seconds':time.monotonic()-started})

USER_SPEAKERS = {'khách hàng', 'ứng viên', 'người thuê'}
def parse_dialogue_text(text: str) -> list[dict[str, object]]:
    turns = []
    for line in text.splitlines():
        match = re.match(r"^\s*([^:\n]{2,40}):\s*(.+?)\s*$", line)
        if not match:
            continue
        speaker, content = match.group(1).strip().casefold(), match.group(2).strip()
        if speaker.startswith(("kết luận", "kiểm chứng", "gold label", "điểm gây nhiễu", "tổng hợp")):
            continue
        role = "USER" if speaker in USER_SPEAKERS else "OTHER_PARTY"
        turns.append({"turn_index": len(turns) + 1, "speaker_role": role, "text": content})
    if not turns:
        turns = [{"turn_index": 1, "speaker_role": "OTHER_PARTY", "text": text.strip()}]
    return turns


def original_test_parser():
    return parse_dialogue_text

def infer():
    if (OUT/'baseline_predictions_frozen.jsonl').exists(): raise RuntimeError('Predictions already frozen')
    assert (OUT/'TRAINING_REPORT.json').exists() or (ROOT/'reports/training/xlmr_risk_only_training_report.json').exists()
    model,collator=build()
    blob=torch.load(OUT/'baseline_best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(blob['model_state'],strict=True);del blob;gc.collect();model.eval()
    parse=original_test_parser(); predictions=[]
    test=rows(TEST/'FINAL_MODEL_INPUT.jsonl')
    with torch.inference_mode():
        for x in test:
            turns=x.get('turns') or parse(x['text'])
            normalized=[{'turn_index':i,'speaker_role':t.get('speaker_role','OTHER_PARTY'),'text':t['text']} for i,t in enumerate(turns,1)]
            scores=model(batch(collator,[{'turns':normalized}])).softmax(-1)[0].cpu().tolist()
            predictions.append({'case_id':x['case_id'],'prediction':LABELS[max(range(3),key=lambda i:scores[i])],
                'risk_scores':dict(zip(LABELS,scores))})
    path=OUT/'baseline_predictions_frozen.jsonl'
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in predictions),encoding='utf-8')
    write('PREDICTION_FREEZE.json',{'cases':len(predictions),'predictions_sha256':sha(path),
        'checkpoint_sha256':sha(OUT/'baseline_best.pt'),'input_sha256':sha(TEST/'FINAL_MODEL_INPUT.jsonl'),
        'parser':'original risk_v05 parser used to produce the public frozen 200 predictions',
        'timestamp_utc':datetime.now(timezone.utc).isoformat()})
    print('PREDICTIONS_FROZEN',flush=True)

def compare():
    freeze=json.loads((OUT/'PREDICTION_FREEZE.json').read_text(encoding='utf-8'))
    assert sha(OUT/'baseline_predictions_frozen.jsonl')==freeze['predictions_sha256']
    gold=rows(TEST/'INTERNAL_GOLD.jsonl');ids=[x['case_id'] for x in gold];byid={x['case_id']:x for x in gold}
    results={};errors={}
    for name,path in [('baseline',OUT/'baseline_predictions_frozen.jsonl'),('public',TEST/'predictions_frozen.jsonl')]:
        predictions=rows(path);pmap={x['case_id']:x for x in predictions}
        assert len(predictions)==len(pmap)==200 and set(pmap)==set(ids)
        truth=[byid[i]['gold_risk'] for i in ids]; pred=[pmap[i]['prediction'] for i in ids]
        report={'correct':sum(a==b for a,b in zip(truth,pred)),'cases':200,
            'accuracy':accuracy_score(truth,pred),'macro_f1_all_3_labels':f1_score(truth,pred,labels=list(LABELS),average='macro',zero_division=0),
            'classification_report':classification_report(truth,pred,labels=list(LABELS),output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(truth,pred,labels=list(LABELS)).tolist(),'label_order':LABELS,
            'prediction_counts':dict(Counter(pred)),
            'scam_to_no_evidence':sum(a=='HIGH_EVIDENCE' and b=='NO_EVIDENCE' for a,b in zip(truth,pred)),
            'scam_to_review':sum(a=='HIGH_EVIDENCE' and b=='REVIEW' for a,b in zip(truth,pred)),
            'negative_to_high_evidence':sum(a=='NO_EVIDENCE' and b=='HIGH_EVIDENCE' for a,b in zip(truth,pred)),
            'negative_to_review':sum(a=='NO_EVIDENCE' and b=='REVIEW' for a,b in zip(truth,pred))}
        report['subsets']={}
        for subset in sorted({x['subset'] for x in gold}):
            selected=[i for i in ids if byid[i]['subset']==subset]
            report['subsets'][subset]={'cases':len(selected),'correct':sum(byid[i]['gold_risk']==pmap[i]['prediction'] for i in selected)}
        results[name]=report
        errors[name]=[{'case_id':i,'gold':byid[i]['gold_risk'],'prediction':pmap[i]['prediction']} for i in ids if byid[i]['gold_risk']!=pmap[i]['prediction']]
    old=json.loads((TEST/'FINAL_REPORT.json').read_text(encoding='utf-8'))
    assert abs(results['public']['accuracy']-old['strict_accuracy'])<1e-10
    before=json.loads((OUT/'PRETRAIN_MANIFEST.json').read_text(encoding='utf-8'))['protected_files']
    assert protected_files()==before
    report={'status':'COMPLETE','design':'single matched risk-stage seed; no semantic or RUN004A initialization for baseline',
        'evaluation_type':'COMPARATIVE_REEVALUATION_NOT_NEW_BLIND_TEST','results':results,'errors':errors,'public_unchanged':True,
        'limitations':['One seed, no training-variance estimate','Public system also inherits RUN004A action pretraining; difference is the whole auxiliary-pretraining pipeline, not semantic alone','Gold contains no REVIEW','Existing test results were previously inspected','Original evaluation parser intentionally used for both sides; current public API helper differs']}
    write('COMPARISON_REPORT.json',report)
    lines=['# XLM-R risk-only versus public checkpoint','',
        'Same 3,805 risk TRAIN / 398 DEV; seed 11019; three epochs. Baseline initialized only from original XLM-R. No public files changed.',
        '', '| Metric | XLM-R risk-only | Public pretrained pipeline |','|---|---:|---:|']
    for title,key in [('Correct / 200','correct'),('Accuracy','accuracy'),('Macro-F1, 3 labels','macro_f1_all_3_labels'),
        ('Scam predicted NO_EVIDENCE','scam_to_no_evidence'),('Scam predicted REVIEW','scam_to_review'),
        ('Negative predicted HIGH_EVIDENCE','negative_to_high_evidence'),('Negative predicted REVIEW','negative_to_review')]:
        a,b=results['baseline'][key],results['public'][key]
        if isinstance(a,float):a,b=f'{a:.4%}',f'{b:.4%}'
        lines.append(f'| {title} | {a} | {b} |')
    lines+=['','Comparative re-evaluation of the existing 200 cases, not a new blind test. One training seed. Public model inherits both action and semantic pretraining. Gold has no REVIEW. No tuning on these test results.']
    (OUT/'KET_QUA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    {'train':train,'infer':infer,'compare':compare}[sys.argv[1]]()
