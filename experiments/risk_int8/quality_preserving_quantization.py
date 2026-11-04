"""Round 3: weight-only or activation smoothing, chosen by unlabeled DEV teacher fidelity."""
from pathlib import Path
import sys,os,json,gc,time,datetime,random
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(HERE/'deps'))
os.environ['TEMP']=os.environ['TMP']=str(HERE/'tmp')
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from onnxruntime.quantization import quantize_dynamic,QuantType
import improve_quantization as q
from risk_only_runtime import RiskOnlyRuntime,LABELS,parse_text
from tokenizers import Tokenizer

R=HERE/'round3_quality';R.mkdir(exist_ok=True)
q.R=R
FP=q.FP
def save(name,x):q.save(name,x)
def word_row_model(source,target):
    old_fp=q.FP;q.FP=source
    # The helper's fixed intermediate name is safe in a unique per-recipe folder.
    old_r=q.R;q.R=target.parent;q.R.mkdir(exist_ok=True)
    path=q.make_row_embedding()
    q.R=old_r;q.FP=old_fp
    return path

def weight_only(source,target):
    if target.exists():return target
    q.log('INT8 weight-only: per-output-channel linear weights, FP32 activations.')
    model=onnx.load(str(source));initial={x.name:x for x in model.graph.initializer};replaced={};nodes=[]
    for node in model.graph.node:
        if node.op_type not in ['MatMul','Gemm'] or node.input[1] not in initial:
            nodes.append(node);continue
        name=node.input[1]
        if name not in replaced:
            tensor=initial[name];w=numpy_helper.to_array(tensor)
            if w.ndim!=2 or w.dtype!=np.float32:nodes.append(node);continue
            trans_b=next((a.i for a in node.attribute if a.name=='transB'),0)
            axis=0 if node.op_type=='Gemm' and trans_b else 1
            scale=np.maximum(np.max(np.abs(w),axis=1-axis)/127,np.float32(1e-12)).astype(np.float32)
            shape=[1,1];shape[axis]=len(scale)
            quant=np.clip(np.rint(w/scale.reshape(shape)),-127,127).astype(np.int8)
            model.graph.initializer.remove(tensor)
            model.graph.initializer.extend([numpy_helper.from_array(quant,name+'_w8'),
                numpy_helper.from_array(scale,name+'_scale'),numpy_helper.from_array(np.zeros(len(scale),dtype=np.int8),name+'_zero')])
            nodes.append(helper.make_node('DequantizeLinear',[name+'_w8',name+'_scale',name+'_zero'],[name],axis=axis,name=node.name+'_WeightOnlyDQ'))
            replaced[name]=True
        nodes.append(node)
    del model.graph.node[:];model.graph.node.extend(nodes)
    onnx.checker.check_model(model);onnx.save(model,str(target));del model,nodes,initial;gc.collect()
    return target

def smooth_models(encoder):
    stats_path=R/'train_activation_max.npz'
    graph=onnx.load(str(FP),load_external_data=False);initial={t.name:t for t in graph.graph.initializer}
    projections=[n for n in graph.graph.node if n.op_type=='MatMul' and n.input[1] in initial]
    dimensions={n.input[0]:initial[n.input[1]].dims[0] for n in projections}
    if not stats_path.exists():
        q.log('Collecting activation maxima from 64 TRAIN rows, no test/calibration gold.')
        augmented=R/'calibration_augmented.onnx'
        for name,width in dimensions.items():graph.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,[None,None,width]))
        onnx.save(graph,str(augmented));del graph;gc.collect()
        session=q.session(augmented);names=[x.name for x in session.get_outputs()][1:]
        train=q.read(HERE.parents[1]/'data/splits/RISK_TRAIN.jsonl')
        chosen=random.Random(20261103).sample(range(len(train)),64)
        save('TRAIN_CALIBRATION_IDS.json',{'seed':20261103,'item_ids':[train[i]['item_id'] for i in chosen]})
        maxima={name:np.zeros(width,dtype=np.float32) for name,width in dimensions.items()}
        for pos,i in enumerate(chosen):
            output=session.run(names,encoder.encode(train[i]['turns']))
            for name,x in zip(names,output):maxima[name]=np.maximum(maxima[name],np.max(np.abs(x),axis=tuple(range(x.ndim-1))))
            if (pos+1)%16==0:q.log(f'TRAIN calibration {pos+1}/64')
        np.savez_compressed(stats_path,**maxima);del session,output;gc.collect()
    else:del graph;gc.collect()
    maxima=np.load(stats_path)
    paths=[]
    for alpha in [.5,.7]:
        name=f'smooth_alpha_{str(alpha).replace(".","_")}'
        folder=R/name;folder.mkdir(exist_ok=True);smooth=folder/'smoothed_fp32.onnx'
        if not smooth.exists():
            model=onnx.load(str(FP));initial={t.name:t for t in model.graph.initializer};nodes=[]
            for node in model.graph.node:
                if node.op_type=='MatMul' and node.input[1] in initial:
                    weight=initial[node.input[1]];w=numpy_helper.to_array(weight)
                    am=np.maximum(maxima[node.input[0]],1e-5);wm=np.maximum(np.max(np.abs(w),axis=1),1e-5)
                    scale=np.clip(am**alpha/wm**(1-alpha),1e-5,1e5).astype(np.float32)
                    scale_name=node.name+'_smooth_scale';input_name=node.input[0]+'_smooth_'+str(len(nodes))
                    nodes.append(helper.make_node('Div',[node.input[0],scale_name],[input_name],name=node.name+'_SmoothDiv'))
                    node.input[0]=input_name
                    model.graph.initializer.remove(weight)
                    model.graph.initializer.extend([numpy_helper.from_array(w*scale[:,None],weight.name),numpy_helper.from_array(scale,scale_name)])
                nodes.append(node)
            del model.graph.node[:];model.graph.node.extend(nodes)
            onnx.checker.check_model(model);onnx.save(model,str(smooth));del model,nodes,initial;gc.collect()
        # Check smoothing equivalence on TRAIN before quantization; no fitting to test.
        if not (folder/'EQUIVALENCE.json').exists():
            train=q.read(HERE.parents[1]/'data/splits/RISK_TRAIN.jsonl')[:4]
            arrays=[encoder.encode(x['turns']) for x in train]
            s=q.session(FP);base=np.concatenate([s.run(None,x)[0] for x in arrays]);del s;gc.collect()
            s=q.session(smooth);changed=np.concatenate([s.run(None,x)[0] for x in arrays]);del s;gc.collect()
            difference=float(np.max(np.abs(base-changed)));assert difference<.001,difference
            (folder/'EQUIVALENCE.json').write_text(json.dumps({'max_abs_logit_error_fp32':difference,'rows':4}),encoding='utf-8')
        row=word_row_model(smooth,folder/'row_embedding_fp32_rest.onnx')
        target=folder/'risk_int8.onnx'
        if not target.exists():
            q.log('Quantizing smoothed '+name)
            quantize_dynamic(str(row),str(target),weight_type=QuantType.QInt8,per_channel=True,
                op_types_to_quantize=['MatMul','Gemm'],extra_options={'MatMulConstBOnly':True})
        paths.append((name,target))
    return paths

def main():
    protocol={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'context':'Adaptive development after round2 test failure; no claim of fresh independent evaluation.',
        'candidates':['row-wise embedding + per-channel INT8 weights with FP32 activations','SmoothQuant-inspired alpha=0.5','SmoothQuant-inspired alpha=0.7'],
        'selection':'No gold labels used for ranking: among <=450MB and median latency <=1.25x FP32, minimize DEV logit MSE to FP32, tie fewer changed labels. Fallback min MSE among <=450MB.',
        'calibration':'64 seeded TRAIN inputs for smoothing; no test input or gold used for activation statistics; no training or threshold changes.'}
    if not (R/'PROTOCOL_BEFORE_DEV.json').exists():save('PROTOCOL_BEFORE_DEV.json',protocol)
    encoder=RiskOnlyRuntime.__new__(RiskOnlyRuntime);encoder.tokenizer=Tokenizer.from_file(str(HERE/'tokenizer/tokenizer.json'))
    dev=q.read(HERE.parents[1]/'data/splits/RISK_DEV.jsonl');arrays=[encoder.encode(x['turns']) for x in dev]
    targets=np.array([LABELS.index(x['labels']['risk']) for x in dev])
    results={};results['fp32'],base=q.evaluate('fp32',FP,arrays,targets)
    row=word_row_model(FP,R/'row_embedding_fp32_rest.onnx')
    paths=[('weight_only_int8',weight_only(row,R/'risk_weight_only_int8.onnx'))]+smooth_models(encoder)
    for name,path in paths:
        results[name],_=q.evaluate(name,path,arrays,targets,base)
        save('DEV_RESULTS.json',results)
    eligible=[n for n,p in paths if results[n]['model_bytes']<=450000000 and results[n]['median_ms']<=results['fp32']['median_ms']*1.25]
    if not eligible:eligible=[n for n,p in paths if results[n]['model_bytes']<=450000000]
    chosen=min(eligible,key=lambda n:(results[n]['logit_mse_vs_fp32'],results[n]['changed_labels_vs_fp32']))
    path=dict(paths)[chosen]
    freeze={'name':chosen,'path':str(path.relative_to(HERE)),'sha256':q.sha(path),
        'dev':results[chosen],'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save('SELECTED_BEFORE_TEST.json',freeze)
    test=HERE.parents[1]/'evaluation/independent_test_200';inputs=q.read(test/'FINAL_MODEL_INPUT.jsonl')
    gold={x['case_id']:x['gold_risk'] for x in q.read(test/'INTERNAL_GOLD.jsonl')}
    arrays=[encoder.encode(x.get('turns') or parse_text(x['text'])) for x in inputs]
    targets=np.array([LABELS.index(gold[x['case_id']]) for x in inputs])
    session=q.session(FP);reference=np.concatenate([session.run(None,x)[0] for x in arrays]);del session;gc.collect()
    session=q.session(path);values=[];ms=[]
    for x in arrays[:3]:session.run(None,x)
    for x in arrays:
        t=time.perf_counter();values.append(session.run(None,x)[0]);ms.append((time.perf_counter()-t)*1000)
    values=np.concatenate(values);test_result=q.stats(values,targets,reference)
    test_result.update({'median_ms':float(np.median(ms)),'p95_ms':float(np.percentile(ms,95))})
    with (R/'selected_test_predictions.jsonl').open('w',encoding='utf-8') as f:
        for row,value in zip(inputs,values):
            prob=np.exp(value-value.max());prob/=prob.sum()
            f.write(json.dumps({'case_id':row['case_id'],'prediction':LABELS[int(value.argmax())],'risk_scores':dict(zip(LABELS,map(float,prob)))},ensure_ascii=False)+'\n')
    save('REPORT.json',{'protocol':protocol,'selected':freeze,'dev_results':results,'test':test_result,
        'test_fp32':q.stats(reference,targets),'android_tested':False,'not_independent_test':True})
    q.log(json.dumps({'selected':chosen,'dev':results[chosen],'test':test_result},indent=2))

if __name__=='__main__':main()
