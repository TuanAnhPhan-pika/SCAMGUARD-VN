"""Verify archived research results without model inference, retraining, or writes."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LABELS=['NO_EVIDENCE','REVIEW','HIGH_EVIDENCE']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def near(a,b):assert abs(a-b)<1e-9,(a,b)

def compute(path,gold):
    predictions=read(path); byid={x['case_id']:x for x in predictions}
    assert len(predictions)==len(byid)==200 and set(byid)==set(gold),str(path)
    matrix=[[0]*3 for _ in LABELS];operational=0
    for cid,g in gold.items():
        prediction=byid[cid]['prediction']
        assert prediction in LABELS
        scores=byid[cid].get('risk_scores',{})
        if scores:
            assert set(scores)==set(LABELS)
            assert abs(sum(scores.values())-1)<1e-5
            assert all(0<=v<=1 for v in scores.values())
            assert max(scores,key=scores.get)==prediction
        matrix[LABELS.index(g['gold_risk'])][LABELS.index(prediction)]+=1
        allowed={'POSITIVE':{'REVIEW','HIGH_EVIDENCE'},'HARD_NEGATIVE':{'NO_EVIDENCE','REVIEW'},'EASY_NEGATIVE':{'NO_EVIDENCE'}}[g['subset']]
        operational+=prediction in allowed
    accuracy=sum(matrix[i][i] for i in range(3))/200
    f1=sum(2*matrix[i][i]/max(1,sum(matrix[i])+sum(row[i] for row in matrix)) for i in [0,2])/2
    return {'accuracy':accuracy,'macro_f1':f1,'operational':operational/200,'matrix':matrix}

def main():
    manifest=ROOT/'evaluation/RESEARCH_CHECKSUMS.json'
    if manifest.exists():
        for name,expected in json.loads(manifest.read_text(encoding='utf-8')).items():
            assert sha(ROOT/name)==expected,f'Hash mismatch: {name}'
    gold={x['case_id']:x for x in read(ROOT/'evaluation/independent_test_200/INTERNAL_GOLD.jsonl')}
    base=ROOT/'evaluation/semantic_pretraining_comparison'
    report=json.loads((base/'EXACT_PUBLIC_SCORING.json').read_text(encoding='utf-8'))
    freeze=json.loads((base/'PREDICTION_FREEZE.json').read_text(encoding='utf-8'))
    assert sha(base/'baseline_predictions_frozen.jsonl')==freeze['predictions_sha256']
    for name,path in [('baseline',base/'baseline_predictions_frozen.jsonl'),('public',ROOT/'evaluation/independent_test_200/predictions_frozen.jsonl')]:
        actual=compute(path,gold);expected=report['results'][name]
        near(actual['accuracy'],expected['strict_accuracy']);near(actual['macro_f1'],expected['macro_f1_supported_labels'])
        near(actual['operational'],expected['operational_accuracy'])
        assert actual['matrix']==expected['confusion_matrix']['values']
        print(f'{name}: accuracy={actual["accuracy"]:.4f}, macro-F1={actual["macro_f1"]:.6f}, operational={actual["operational"]:.4f}')
    quant=ROOT/'evaluation/risk_int8'
    for folder,prediction,report_file,section in [
        ('round1','risk_only_onnx_fp32_predictions.jsonl','REPORT.json','risk_only_onnx_fp32'),
        ('round1','risk_only_onnx_int8_predictions.jsonl','REPORT.json','risk_only_onnx_int8'),
        ('round2_research','selected_test_predictions.jsonl','REPORT.json','test_selected'),
        ('round3_quality','selected_test_predictions.jsonl','REPORT.json','test')]:
        actual=compute(quant/folder/prediction,gold)
        stored=json.loads((quant/folder/report_file).read_text(encoding='utf-8'))
        expected=stored['metrics'][section] if folder=='round1' else stored[section]
        near(actual['accuracy'],expected.get('strict_accuracy',expected.get('accuracy')))
        near(actual['macro_f1'],expected.get('macro_f1_gold_supported_labels',expected.get('macro_f1')))
        assert actual['matrix']==expected['confusion_matrix']
        if folder!='round1':
            assert sha(quant/folder/prediction)==stored.get('selected_prediction_sha256',sha(quant/folder/prediction))
        print(f'{folder}/{prediction}: accuracy={actual["accuracy"]:.4f}, macro-F1={actual["macro_f1"]:.6f}')
    audit=json.loads((quant/'round1/PREPROCESSING_AUDIT.json').read_text(encoding='utf-8'))
    assert audit['historical_parser_changed_labels_vs_frozen']==[]
    assert len(audit['current_vs_frozen_changed_cases'])==7
    original={x['case_id']:x['prediction'] for x in read(ROOT/'evaluation/independent_test_200/predictions_frozen.jsonl')}
    historical=read(quant/'round1/historical_parser_audit_predictions.jsonl')
    assert len(historical)==200 and all(x['risk_level']==original[x['case_id']] for x in historical)
    print('OK: archived hashes, all 200-case metrics, and parser audit. No models or files modified.')

if __name__=='__main__':main()
