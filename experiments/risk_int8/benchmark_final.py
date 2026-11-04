from pathlib import Path
import json,time,gc
import risk_only_runtime as rt
import numpy as np
from tokenizers import Tokenizer
HERE=Path(__file__).resolve().parent
R=HERE/'round3_quality'
selection=json.loads((R/'SELECTED_BEFORE_TEST.json').read_text(encoding='utf-8'))
paths={'fp32':HERE/'outputs/risk_only_fp32.onnx','selected_int8':HERE/selection['path']}
inputs=[json.loads(x) for x in (HERE.parents[1]/'evaluation/independent_test_200/FINAL_MODEL_INPUT.jsonl').read_text(encoding='utf-8').splitlines()]
encoder=rt.RiskOnlyRuntime.__new__(rt.RiskOnlyRuntime)
encoder.tokenizer=Tokenizer.from_file(str(HERE/'tokenizer/tokenizer.json'))
arrays=[encoder.encode(x.get('turns') or rt.parse_text(x['text'])) for x in inputs]
times={k:[] for k in paths};loads={k:[] for k in paths}
for order in [('fp32','selected_int8'),('selected_int8','fp32')]:
 for name in order:
  start=time.perf_counter();model=rt.RiskOnlyRuntime(paths[name]);loads[name].append(time.perf_counter()-start)
  for x in arrays[:5]:model.session.run(None,x)
  for x in arrays:
   start=time.perf_counter();model.session.run(None,x);times[name].append((time.perf_counter()-start)*1000)
  del model;gc.collect()
  print(name,'finished',len(times[name]),'timed calls',flush=True)
report={'protocol':'Fixed selected model; no model selection here. Two repetitions; reversed model order in repeat 2; 5 warmups per session; same 200 pretokenized inputs; CPU 4 threads, batch 1; no tokenization/guidance.',
 'android_tested':False,'models':{k:{'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95)),'mean_ms':float(np.mean(v)),'calls':len(v),'session_load_seconds':loads[k]} for k,v in times.items()}}
report['median_speedup']=report['models']['fp32']['median_ms']/report['models']['selected_int8']['median_ms']
(R/'FINAL_LATENCY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
