"""Round 2: fixed recipes, choose on DEV before reading test; no training or threshold tuning."""
from pathlib import Path
import sys,os,json,time,hashlib,gc,datetime
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE/'deps'))
(HERE/'tmp').mkdir(exist_ok=True)
os.environ['TEMP']=os.environ['TMP']=str(HERE/'tmp')
import numpy as np
import onnx
from onnx import numpy_helper,helper,TensorProto
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic,QuantType
from risk_only_runtime import RiskOnlyRuntime,LABELS,parse_text

R=HERE/'round2_research';R.mkdir(exist_ok=True)
FP=HERE/'outputs/risk_only_fp32.onnx'
RECIPES=[
 {'name':'per_channel_global_embedding','row_embedding':False,'keep_embedding':False,'keep_last':0},
 {'name':'per_channel_keep_embedding_fp32','row_embedding':False,'keep_embedding':True,'keep_last':0},
 {'name':'row_embedding_per_channel','row_embedding':True,'keep_embedding':False,'keep_last':0},
 {'name':'row_embedding_per_channel_last2_fp32','row_embedding':True,'keep_embedding':False,'keep_last':2},
]
def log(x):print(x,flush=True)
def save(name,x): (R/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def read(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
def session(path):
 o=ort.SessionOptions();o.intra_op_num_threads=4;o.inter_op_num_threads=1
 return ort.InferenceSession(str(path),sess_options=o,providers=['CPUExecutionProvider'])
def stats(values,targets,base=None):
 pred=values.argmax(-1);mat=np.zeros((3,3),dtype=int)
 for a,b in zip(targets,pred):mat[a,b]+=1
 support=[i for i in range(3) if mat[i].sum()]
 f1=[2*mat[i,i]/max(1,mat[i].sum()+mat[:,i].sum()) for i in support]
 out={'accuracy':float(np.trace(mat)/mat.sum()),'macro_f1':float(np.mean(f1)),
 'confusion_matrix':mat.tolist(),'gold_class_support':mat.sum(1).tolist(),
 'high_to_no_evidence':int(mat[2,0]),'no_evidence_to_high':int(mat[0,2])}
 if base is not None:
  out.update({'changed_labels_vs_fp32':int(np.sum(pred!=base.argmax(-1))),
   'logit_mse_vs_fp32':float(np.mean((values-base)**2)),
   'max_abs_logit_error':float(np.max(np.abs(values-base)))})
 return out
def evaluate(name,path,arrays,targets,base=None):
 log('DEV evaluate '+name)
 s=session(path)
 for x in arrays[:3]:s.run(None,x)
 values=[];ms=[]
 for i,x in enumerate(arrays):
  t=time.perf_counter();values.append(s.run(None,x)[0]);ms.append((time.perf_counter()-t)*1000)
  if (i+1)%100==0:log(f'{name}: {i+1}/{len(arrays)}')
 values=np.concatenate(values);result=stats(values,targets,base)
 result.update({'median_ms':float(np.median(ms)),'p95_ms':float(np.percentile(ms,95)),
 'model_bytes':path.stat().st_size})
 np.savez_compressed(R/(name+'_dev_logits.npz'),logits=values)
 del s;gc.collect();return result,values

def make_row_embedding():
 path=R/'row_embedding_fp32_rest.onnx'
 if path.exists():return path
 log('Building per-token-row INT8 word embeddings; transformer stays FP32 in intermediate.')
 model=onnx.load(str(FP));weight_name='backbone.model.embeddings.word_embeddings.weight'
 weight=next(x for x in model.graph.initializer if x.name==weight_name)
 array=numpy_helper.to_array(weight)
 scale=np.maximum(np.max(np.abs(array),axis=1,keepdims=True)/127,np.float32(1e-12)).astype(np.float32)
 q=np.clip(np.rint(array/scale),-127,127).astype(np.int8)
 model.graph.initializer.remove(weight)
 model.graph.initializer.extend([numpy_helper.from_array(q,weight_name+'_row_int8'),
   numpy_helper.from_array(scale,weight_name+'_row_scale')])
 nodes=[]
 for node in model.graph.node:
  if node.op_type=='Gather' and node.input[0]==weight_name:
   original=node.output[0];indices=node.input[1]
   nodes.extend([
    helper.make_node('Gather',[weight_name+'_row_int8',indices],[original+'_int8'],name=node.name+'_RowInt8',axis=0),
    helper.make_node('Cast',[original+'_int8'],[original+'_float'],name=node.name+'_Cast',to=TensorProto.FLOAT),
    helper.make_node('Gather',[weight_name+'_row_scale',indices],[original+'_scale'],name=node.name+'_RowScale',axis=0),
    helper.make_node('Mul',[original+'_float',original+'_scale'],[original],name=node.name+'_Dequantize')])
  else:nodes.append(node)
 del model.graph.node[:];model.graph.node.extend(nodes)
 onnx.checker.check_model(model)
 onnx.save(model,str(path));del model,weight,array,scale,q,nodes;gc.collect()
 return path

def build(recipe):
 path=R/(recipe['name']+'.onnx')
 if path.exists():return path
 source=make_row_embedding() if recipe['row_embedding'] else FP
 model=onnx.load(str(source),load_external_data=False)
 exclusions=[n.name for n in model.graph.node if recipe['keep_last'] and
  any(f'/encoder/layer.{i}/' in n.name for i in range(12-recipe['keep_last'],12))]
 del model;gc.collect()
 operators=['MatMul','Gemm']
 if not recipe['row_embedding'] and not recipe['keep_embedding']:operators+=['Gather']
 log('Quantizing '+recipe['name'])
 quantize_dynamic(str(source),str(path),weight_type=QuantType.QInt8,per_channel=True,
  op_types_to_quantize=operators,nodes_to_exclude=exclusions,
  extra_options={'MatMulConstBOnly':True})
 return path

def main():
 protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'recipes':RECIPES,
  'selection':'DEV only: candidates <=450 MB decimal, highest three-class macro-F1; ties fewer HIGH->NO and NO->HIGH errors; then fewer changed labels; then smaller size.',
  'not_used_for_selection':'Independent test 200, its errors or gold; no thresholds or retraining.',
  'source_fp32_sha256':sha(FP),'size_budget_bytes':450000000}
 if not (R/'PROTOCOL_BEFORE_DEV.json').exists():save('PROTOCOL_BEFORE_DEV.json',protocol)
 rows=read(HERE.parents[1]/'data/splits/RISK_DEV.jsonl')
 encoder=RiskOnlyRuntime.__new__(RiskOnlyRuntime)
 from tokenizers import Tokenizer
 encoder.tokenizer=Tokenizer.from_file(str(HERE/'tokenizer/tokenizer.json'))
 arrays=[encoder.encode(x['turns']) for x in rows]
 targets=np.array([LABELS.index(x['labels']['risk']) for x in rows])
 results={}
 results['fp32'],base=evaluate('fp32',FP,arrays,targets)
 results['previous_int8'],_=evaluate('previous_int8',HERE/'outputs/risk_only_int8.onnx',arrays,targets,base)
 for recipe in RECIPES:
  path=build(recipe);results[recipe['name']],_=evaluate(recipe['name'],path,arrays,targets,base)
  save('DEV_RESULTS.json',results);log(json.dumps({recipe['name']:results[recipe['name']]}))
 eligible=[x['name'] for x in RECIPES if results[x['name']]['model_bytes']<=450000000]
 selected=sorted(eligible,key=lambda n:(-results[n]['macro_f1'],results[n]['high_to_no_evidence']+results[n]['no_evidence_to_high'],results[n]['changed_labels_vs_fp32'],results[n]['model_bytes']))[0]
 selected_path=R/(selected+'.onnx')
 freeze={'selected':selected,'sha256':sha(selected_path),'selection_completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'dev':results[selected],'reason':'Fixed rule in PROTOCOL_BEFORE_DEV.json; no test inference yet in this round.'}
 save('SELECTED_BEFORE_TEST.json',freeze)
 # Only now open test gold, after choosing and hashing the single selected candidate.
 test=HERE.parents[1]/'evaluation/independent_test_200'
 inputs=read(test/'FINAL_MODEL_INPUT.jsonl');gold={x['case_id']:x['gold_risk'] for x in read(test/'INTERNAL_GOLD.jsonl')}
 test_arrays=[encoder.encode(x.get('turns') or parse_text(x['text'])) for x in inputs]
 truth=np.array([LABELS.index(gold[x['case_id']]) for x in inputs])
 s=session(FP);fp_logits=np.concatenate([s.run(None,x)[0] for x in test_arrays]);del s;gc.collect()
 s=session(selected_path);ms=[];values=[]
 for x in test_arrays[:3]:s.run(None,x)
 for x in test_arrays:
  t=time.perf_counter();values.append(s.run(None,x)[0]);ms.append((time.perf_counter()-t)*1000)
 values=np.concatenate(values);del s;gc.collect()
 test_result=stats(values,truth,fp_logits)
 test_result.update({'median_ms':float(np.median(ms)),'p95_ms':float(np.percentile(ms,95))})
 with (R/'selected_test_predictions.jsonl').open('w',encoding='utf-8') as f:
  for row,value in zip(inputs,values):
   prob=np.exp(value-value.max());prob/=prob.sum()
   f.write(json.dumps({'case_id':row['case_id'],'prediction':LABELS[int(value.argmax())],
    'risk_scores':dict(zip(LABELS,map(float,prob)))},ensure_ascii=False)+'\n')
 report={'scope':'DEV-selected round 2; test is comparative re-evaluation, not new independent evidence.',
  'selected':freeze,'dev_results':results,'test_fp32':stats(fp_logits,truth),
  'test_selected':test_result,'test_gold_has_no_review':True,'android_tested':False,
  'test_input_sha256':sha(test/'FINAL_MODEL_INPUT.jsonl'),
  'selected_prediction_sha256':sha(R/'selected_test_predictions.jsonl')}
 save('REPORT.json',report)
 lines=['# Cải thiện risk INT8 — vòng 2','','Chọn cấu hình trên DEV 398 mẫu trước khi chạy lại bộ 200; không train hoặc chỉnh ngưỡng.','',
 '| DEV cấu hình | Macro-F1 ba lớp | Accuracy | MB | Trung vị CPU ms | Đổi nhãn so với FP32 |',
 '|---|---:|---:|---:|---:|---:|']
 for name,result in results.items():
  lines.append(f"| {name} | {result['macro_f1']:.2%} | {result['accuracy']:.2%} | {result['model_bytes']/1e6:.1f} | {result['median_ms']:.1f} | {result.get('changed_labels_vs_fp32',0)} |")
 lines+=['',f'Được chọn trước test: **{selected}**.','',
 '| Test 200 | Accuracy | Macro-F1 hai nhãn | Positive lọt xanh | Negative báo đỏ |',
 '|---|---:|---:|---:|---:|']
 for name,result in [('FP32',report['test_fp32']),('INT8 chọn bằng DEV',test_result)]:
  lines.append(f"| {name} | {result['accuracy']:.2%} | {result['macro_f1']:.2%} | {result['high_to_no_evidence']} | {result['no_evidence_to_high']} |")
 lines+=['','Không chọn lại cấu hình dựa trên kết quả test này. DEV tổng hợp còn hạn chế, gold test không có REVIEW. Tốc độ CPU laptop không đại diện Android. Giữ encoder risk học semantic; không chạy semantic model riêng.','',
 'Đây là ứng dụng nguyên tắc per-channel/per-row và mixed precision, không phải tái hiện đầy đủ OS+, SmoothQuant hoặc Variable Layerwise Quantization.']
 (R/'KET_QUA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 log(json.dumps({'selected':selected,'test':test_result,'dev_f1':results[selected]['macro_f1']},indent=2))

if __name__=='__main__':main()
