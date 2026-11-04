"""Standalone CPU inference: ONNX + tokenizer, no torch or semantic checkpoint."""
from pathlib import Path
import sys, re, json
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE/'deps'))
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

LABELS=('NO_EVIDENCE','REVIEW','HIGH_EVIDENCE')

def parse_text(text):
    pattern=re.compile(r'^(khách hàng|người dùng|tôi|user|nhân viên hỗ trợ|nhân viên|hệ thống|system)\s*:\s*(.*)$',re.I)
    turns=[]
    for line in (x.strip() for x in text.splitlines() if x.strip()):
        m=pattern.match(line)
        if m:
            role='USER' if m.group(1).lower() in {'khách hàng','người dùng','tôi','user'} else 'OTHER_PARTY'
            turns.append({'speaker_role':role,'text':m.group(2)})
        elif turns: turns[-1]['text']+='\n'+line
        else: turns.append({'speaker_role':'OTHER_PARTY','text':line})
    return turns or [{'speaker_role':'OTHER_PARTY','text':text}]

class RiskOnlyRuntime:
    def __init__(self, model_path=None):
        self.tokenizer=Tokenizer.from_file(str(HERE/'tokenizer/tokenizer.json'))
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model_path or HERE.parents[1]/'artifacts/onnx/risk_int8.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    def encode(self,turns):
        current=len(turns);chunks=[]
        for i,turn in enumerate(turns,1):
            marker=f"<T{i}><{turn.get('speaker_role','OTHER_PARTY')}>"+('<CURRENT>' if i==current else '')
            chunks.append((i,self.tokenizer.encode(marker+' '+turn['text'],add_special_tokens=False).ids))
        chunks=chunks[-12:];index,ids=chunks[-1];ids=ids[:190]
        kept=[(index,ids)];remaining=190-len(ids)
        for index,ids in reversed(chunks[:-1]):
            if len(ids)>remaining: break
            kept.append((index,ids));remaining-=len(ids)
        kept.reverse()
        flat=[v for _,ids in kept for v in ids]
        mask=[int(index==current) for index,ids in kept for _ in ids]
        # XLM-R single-sequence special tokens: <s>=0, </s>=2. No batch padding at batch 1.
        ids=[0]+flat+[2]
        return {'input_ids':np.array([ids],dtype=np.int64),
            'attention_mask':np.ones((1,len(ids)),dtype=np.int64),
            'current_mask':np.array([[0]+mask+[0]],dtype=np.int64)}
    def analyze_turns(self,turns):
        logits=self.session.run(None,self.encode(turns))[0][0]
        probs=np.exp(logits-logits.max());probs/=probs.sum();index=int(probs.argmax())
        return {'risk_level':LABELS[index],'risk_scores':dict(zip(LABELS,map(float,probs)))}
    def analyze(self,text): return self.analyze_turns(parse_text(text))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--text',required=True)
    parser.add_argument('--model',type=Path,help='Explicit ONNX path; default is the separately released experimental INT8 model; does not replace model-v1.')
    args=parser.parse_args()
    print(json.dumps(RiskOnlyRuntime(args.model).analyze(args.text),ensure_ascii=False,indent=2))
