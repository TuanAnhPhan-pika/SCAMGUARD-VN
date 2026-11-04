"""Checks why current published preprocessing differs from original frozen predictions."""
from pathlib import Path
import json,re
import risk_only_runtime as runtime
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
test=ROOT/'evaluation/independent_test_200'
def read(path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def historical_parser(text):
    turns=[]
    for line in text.splitlines():
        m=re.match(r'^\s*([^:\n]{2,40}):\s*(.+?)\s*$',line)
        if not m: continue
        speaker,content=m.group(1).strip().casefold(),m.group(2).strip()
        if speaker.startswith(('kết luận','kiểm chứng','gold label','điểm gây nhiễu','tổng hợp')):continue
        turns.append({'speaker_role':'USER' if speaker in {'khách hàng','ứng viên','người thuê'} else 'OTHER_PARTY','text':content})
    return turns or [{'speaker_role':'OTHER_PARTY','text':text.strip()}]

model=runtime.RiskOnlyRuntime(HERE/'outputs/risk_only_fp32.onnx')
inputs=read(test/'FINAL_MODEL_INPUT.jsonl')
frozen={x['case_id']:x for x in read(test/'predictions_frozen.jsonl')}
current={x['case_id']:x for x in read(HERE/'outputs/risk_only_onnx_fp32_predictions.jsonl')}
old_differences=[];current_differences=[];max_score_error=0.;historical_predictions=[]
for row in inputs:
    cid=row['case_id'];ts=row.get('turns') or runtime.parse_text(row['text'])
    result=model.analyze_turns(ts)
    if result['risk_level']!=current[cid]['prediction']:current_differences.append(cid)
    max_score_error=max(max_score_error,max(abs(result['risk_scores'][k]-current[cid]['risk_scores'][k]) for k in runtime.LABELS))
    old_ts=row.get('turns') or historical_parser(row['text'])
    old=model.analyze_turns(old_ts)
    historical_predictions.append({'case_id':cid,**old})
    if old['risk_level']!=frozen[cid]['prediction']:old_differences.append(cid)
report={'purpose':'Preprocessing audit only; not choosing preprocessing by test accuracy. Current published parser is retained for all compression comparisons.',
    'native_tokenizer_pipeline_changed_labels_vs_reference_onnx':current_differences,
    'native_tokenizer_max_probability_error':max_score_error,
    'historical_parser_changed_labels_vs_frozen':old_differences,
    'current_vs_frozen_changed_cases':[c for c in current if current[c]['prediction']!=frozen[c]['prediction']]}
(HERE/'outputs/PREPROCESSING_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(HERE/'outputs/historical_parser_audit_predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in historical_predictions),encoding='utf-8')
assert not current_differences and max_score_error<1e-5,'Native tokenizer parity failed'
print(json.dumps(report,ensure_ascii=False,indent=2))
