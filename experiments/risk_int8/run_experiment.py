"""Isolated, read-only-source experiment. All generated files stay beside this script."""
from pathlib import Path
import os, sys, json, time, hashlib, gc, re, statistics, platform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / 'deps'))
os.environ['HF_HOME'] = str(HERE / 'cache' / 'huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TEMP'] = os.environ['TMP'] = str(HERE / 'tmp')
OUT = HERE / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
(HERE/'tmp').mkdir(exist_ok=True)

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from source.collator import TeacherJarpCollator
from source.schema import ConversationInput, Turn, SpeakerRole
from source.conversation_encoder import SharedConversationEncoder
from source.turn_encoder import masked_mean
from source.risk_model import EndToEndRiskModel, LABELS

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.manual_seed(11019)

def log(message):
    print(message, flush=True)

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def parse_dialogue_text(text):
    # Exact parser used by the published risk runtime, copied without importing training.
    pattern = re.compile(r'^(khách hàng|người dùng|tôi|user|nhân viên hỗ trợ|nhân viên|hệ thống|system)\s*:\s*(.*)$', re.I)
    turns = []
    for line in (item.strip() for item in text.splitlines() if item.strip()):
        match = pattern.match(line)
        if match:
            role = 'USER' if match.group(1).lower() in {'khách hàng','người dùng','tôi','user'} else 'OTHER_PARTY'
            turns.append({'speaker_role': role, 'text': match.group(2)})
        elif turns: turns[-1]['text'] += '\n' + line
        else: turns.append({'speaker_role':'OTHER_PARTY', 'text':line})
    return turns or [{'speaker_role':'OTHER_PARTY','text':text}]

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(str(HERE / 'tokenizer'), local_files_only=True)
        config._attn_implementation = 'eager'
        self.model = AutoModel.from_config(config)
        self.hidden_size = config.hidden_size
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

class RiskOnly(nn.Module):
    """Keeps the fine-tuned risk encoder; removes separate semantic inference and unused turn pooling."""
    def __init__(self, reference):
        super().__init__()
        self.backbone = reference.encoder.backbone
        self.dropout = reference.dropout
        self.classifier = reference.classifier
    def forward(self, input_ids, attention_mask, current_mask):
        states = self.backbone(input_ids, attention_mask)
        attention = attention_mask.bool()
        prefix = masked_mean(states, attention, dim=1)
        current = masked_mean(states, current_mask.bool() & attention, dim=1)
        return self.classifier(self.dropout(torch.cat((prefix,current),dim=-1)))

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=-1, keepdims=True)

def metrics(pred, gold, ids):
    matrix = np.zeros((3,3),dtype=int)
    for case, label in zip(ids,pred): matrix[LABELS.index(gold[case]['gold_risk']),int(label)] += 1
    supported = [i for i in range(3) if matrix[i].sum()]
    f1=[]
    for i in supported:
        tp=matrix[i,i]; denominator=matrix[i].sum()+matrix[:,i].sum()
        f1.append(float(2*tp/denominator) if denominator else 0.)
    subsets={}
    for subset in sorted({x['subset'] for x in gold.values()}):
        chosen=[(gold[c]['gold_risk'],LABELS[int(p)]) for c,p in zip(ids,pred) if gold[c]['subset']==subset]
        subsets[subset]={'cases':len(chosen),'strict_correct':sum(a==b for a,b in chosen),
            'predicted_high':sum(b=='HIGH_EVIDENCE' for _,b in chosen),
            'predicted_no_evidence':sum(b=='NO_EVIDENCE' for _,b in chosen)}
    return {'strict_accuracy':float(np.trace(matrix)/matrix.sum()),'macro_f1_gold_supported_labels':float(np.mean(f1)),
        'confusion_matrix':matrix.tolist(),'label_order':LABELS,'subsets':subsets,
        'positive_missed_to_no_evidence':int(matrix[2,0]),'negative_false_red':int(matrix[0,2]),
        'gold_review_support':int(matrix[1].sum())}

def main():
    checkpoint=ROOT/'artifacts/checkpoints/risk_e2e_best.pt'
    test=ROOT/'evaluation/independent_test_200'
    expected='824b58d32366d746adc32b4350c73cad49054f538d365f6661da4036535c8328'
    log('Verifying source checkpoint (read only)...')
    assert sha(checkpoint)==expected
    inputs=rows(test/'FINAL_MODEL_INPUT.jsonl')
    gold={x['case_id']:x for x in rows(test/'INTERNAL_GOLD.jsonl')}
    frozen={x['case_id']:x for x in rows(test/'predictions_frozen.jsonl')}
    collator=TeacherJarpCollator.from_pretrained(str(HERE/'tokenizer'),None,True,max_tokens=192,max_turns=12)
    batches=[]
    for row in inputs:
        ts=row.get('turns') or parse_dialogue_text(row['text'])
        conv=ConversationInput(turns=tuple(Turn(i,SpeakerRole(x.get('speaker_role','OTHER_PARTY')),x['text']) for i,x in enumerate(ts,1)),current_turn=len(ts))
        batches.append(collator([conv]))
    ids=[x['case_id'] for x in inputs]
    log('Building published risk architecture; strict loading ALL risk weights. No semantic checkpoint loaded.')
    reference=EndToEndRiskModel(SharedConversationEncoder(Backbone(),max_turns=12)).eval()
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    reference.load_state_dict(blob['model_state'],strict=True)
    standalone=RiskOnly(reference).eval()
    del blob
    names=['input_ids','attention_mask','current_mask']
    arrays=[{n:b[n].numpy() for n in names} for b in batches]
    predictions={}; timings={}; logits={}
    def evaluate(name, fn):
        log('Evaluating '+name+' on locked comparative benchmark (200 cases)...')
        for i in range(3): fn(i)
        values=[]; elapsed=[]
        for i in range(len(inputs)):
            start=time.perf_counter(); values.append(fn(i));elapsed.append((time.perf_counter()-start)*1000)
            if (i+1)%25==0: log(f'{name}: {i+1}/{len(inputs)}')
        values=np.concatenate(values);logits[name]=values
        pred=values.argmax(-1);predictions[name]=pred
        timings[name]={'median_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95)),
            'mean_ms':float(np.mean(elapsed)),'cpu_threads':4,'batch_size':1,'warmup_calls':3,
            'includes_tokenization':False,'includes_separate_semantic_inference':False}
        with (OUT/(name+'_predictions.jsonl')).open('w',encoding='utf-8') as f:
            for cid,label,scores in zip(ids,pred,softmax(values)):
                f.write(json.dumps({'case_id':cid,'prediction':LABELS[int(label)],'risk_scores':dict(zip(LABELS,map(float,scores)))},ensure_ascii=False)+'\n')
        return metrics(pred,gold,ids)
    with torch.inference_mode():
        results={'published_risk_reference':evaluate('published_risk_reference',lambda i:reference(batches[i]).numpy()),
            'risk_only_torch_fp32':evaluate('risk_only_torch_fp32',lambda i:standalone(*(batches[i][n] for n in names)).numpy())}
        assert np.array_equal(predictions['published_risk_reference'],predictions['risk_only_torch_fp32'])
        fp32_path=OUT/'risk_only_fp32.onnx'
        if not fp32_path.exists():
            log('Exporting standalone FP32 ONNX with dynamic batch/sequence lengths...')
            torch.onnx.export(standalone,tuple(batches[0][n] for n in names),str(fp32_path),
                input_names=names,output_names=['risk_logits'],opset_version=17,dynamo=False,
                dynamic_axes={**{n:{0:'batch',1:'sequence'} for n in names},'risk_logits':{0:'batch'}},
                do_constant_folding=True)
    del standalone,reference
    gc.collect()
    import onnx, onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8_path=OUT/'risk_only_int8.onnx'
    if not int8_path.exists():
        log('Quantizing constant-weight MatMul/Gemm and embedding Gather to INT8...')
        quantize_dynamic(str(fp32_path),str(int8_path),weight_type=QuantType.QInt8,per_channel=False,
            op_types_to_quantize=['MatMul','Gemm','Gather'],extra_options={'MatMulConstBOnly':True})
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for name,path in [('risk_only_onnx_fp32',fp32_path),('risk_only_onnx_int8',int8_path)]:
        start=time.perf_counter();session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        load_seconds=time.perf_counter()-start
        results[name]=evaluate(name,lambda i:session.run(None,arrays[i])[0])
        timings[name]['session_load_seconds']=load_seconds
        del session;gc.collect()
    base=predictions['published_risk_reference']
    frozen_labels=np.array([LABELS.index(frozen[c]['prediction']) for c in ids])
    comparisons={name:{'changed_labels_vs_reference':int(np.sum(pred!=base)),
        'max_abs_logit_difference_vs_reference':float(np.max(np.abs(logits[name]-logits['published_risk_reference'])))} for name,pred in predictions.items()}
    int8_graph=onnx.load(str(int8_path))
    types={}
    for tensor in int8_graph.graph.initializer:
        label=onnx.TensorProto.DataType.Name(tensor.data_type);types[label]=types.get(label,0)+1
    del int8_graph;gc.collect()
    manifest={'checkpoint_sha256':expected,'input_sha256':sha(test/'FINAL_MODEL_INPUT.jsonl'),
        'gold_sha256':sha(test/'INTERNAL_GOLD.jsonl'),'frozen_prediction_sha256':sha(test/'predictions_frozen.jsonl'),
        'model_files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in [fp32_path,int8_path]},
        'source_files':{p.name:sha(p) for p in (HERE/'source').glob('*.py')},
        'versions':{'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__,'python':platform.python_version()},
        'quantization':{'method':'dynamic INT8','weight_type':'QInt8','per_channel':False,'operators':['MatMul','Gemm','Gather'],'MatMulConstBOnly':True},
        'int8_initializer_types':types}
    report={'protocol':'LOCKED_COMPARATIVE_REEVALUATION; not a new independent test; no training, threshold tuning or test calibration',
        'scope':'Risk-only retains the encoder learned through semantic training. Removes separate semantic checkpoint/outputs, not prior semantic knowledge.',
        'reference_vs_published_frozen_changed_labels':int(np.sum(base!=frozen_labels)),
        'metrics':results,'timings':timings,'comparisons':comparisons,
        'size_reduction_fp32_to_int8':float(1-int8_path.stat().st_size/fp32_path.stat().st_size),
        'android_tested':False,'memory_measured':False,'manifest':manifest}
    audit_path=OUT/'PREPROCESSING_AUDIT.json'
    if audit_path.exists():
        report['preprocessing_audit']=json.loads(audit_path.read_text(encoding='utf-8'))
        report['preprocessing_explanation']='Historical frozen run used the generic speaker-prefix parser. Current published parser is fixed across ALL compression comparisons; audit runs historical parser separately, not to select by test accuracy.'
    report['manifest']['attention_implementation']='eager; original risk checkpoint loaded strict'
    if results['risk_only_onnx_int8']['positive_missed_to_no_evidence']>results['published_risk_reference']['positive_missed_to_no_evidence'] or results['risk_only_onnx_int8']['negative_false_red']>results['published_risk_reference']['negative_false_red']:
        report['deployment_recommendation']='EXPERIMENT_ONLY: INT8 increases safety-relevant errors on this benchmark; do not replace FP32 automatically.'
    for path in (HERE/'tokenizer').glob('*'):
        manifest.setdefault('tokenizer_files',{})[path.name]={'bytes':path.stat().st_size,'sha256':sha(path)}
    save('REPORT.json',report)
    save('MANIFEST.json',manifest)
    lines=['# Kết quả thử nghiệm risk-only INT8','','Đây là đánh giá so sánh trên bộ 200 đã công bố, không phải independent test mới. Không train hoặc chỉnh ngưỡng.','',
        '| Phiên bản | Accuracy | Macro-F1 hai nhãn gold | Bỏ sót positive sang xanh | Negative báo đỏ | Đổi nhãn so với risk gốc | Trung vị CPU (ms) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in results.items():
        lines.append(f"| {name} | {m['strict_accuracy']:.2%} | {m['macro_f1_gold_supported_labels']:.2%} | {m['positive_missed_to_no_evidence']} | {m['negative_false_red']} | {comparisons[name]['changed_labels_vs_reference']} | {timings[name]['median_ms']:.1f} |")
    lines += ['',f"ONNX FP32: {fp32_path.stat().st_size/1e6:.1f} MB; ONNX INT8: {int8_path.stat().st_size/1e6:.1f} MB; giảm {report['size_reduction_fp32_to_int8']:.2%}.",
        '',f"Risk reference khác prediction đóng băng: {report['reference_vs_published_frozen_changed_labels']} / 200 mẫu.",
        '', 'Đo CPU laptop, 4 threads, batch 1, 3 warmup; không gồm tokenizer, semantic model riêng hoặc guidance. Không phải tốc độ trên điện thoại. Chưa đo RAM hoặc chạy Android.',
        '', 'Bản risk-only vẫn giữ encoder đã học semantic; bỏ semantic inference không tương đương train risk không có semantic pretraining. Gold không có REVIEW nên chưa đánh giá đầy đủ ba lớp.']
    if audit_path.exists():
        lines += ['', 'Đã có audit parser tại PREPROCESSING_AUDIT.json. So sánh model nén luôn dùng parser hiện tại; số công bố cũ dùng parser lịch sử. Không dùng thay đổi parser để tối ưu theo test.']
    lines += ['',report.get('deployment_recommendation','Chưa kiểm chứng trên Android.')]
    (OUT/'KET_QUA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    log(json.dumps({'accuracy':{k:v['strict_accuracy'] for k,v in results.items()},'comparisons':comparisons,'size_reduction':report['size_reduction_fp32_to_int8']},indent=2))

if __name__=='__main__': main()
