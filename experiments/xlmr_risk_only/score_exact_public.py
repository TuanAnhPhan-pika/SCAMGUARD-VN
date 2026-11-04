"""Execute the public metric computation unchanged on each frozen prediction set."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
PUBLIC = HERE.parents[1]
TEST = PUBLIC / 'evaluation/independent_test_200'
evaluator = TEST / 'evaluate.py'
verifier = PUBLIC / 'scripts/verify_release.py'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def assigned(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id == name for t in node.targets)

spec = importlib.util.spec_from_file_location('public_evaluator', evaluator)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # Import only; does not run its main or write public files.
main = next(n for n in ast.parse(evaluator.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name == 'main')
start = next(i for i,n in enumerate(main.body) if assigned(n,'predictions'))
end = next(i for i,n in enumerate(main.body) if assigned(n,'report'))
metric_code = compile(ast.Module(body=main.body[start:end+1],type_ignores=[]),str(evaluator),'exec')

# Use the exact macro-F1 expression from the public verification script as well.
vmain = next(n for n in ast.parse(verifier.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name == 'main')
f1expr = None
for n in ast.walk(vmain):
    if isinstance(n,ast.Dict):
        for k,v in zip(n.keys,n.values):
            if isinstance(k,ast.Constant) and k.value == 'macro_f1_supported_labels':
                f1expr = compile(ast.Expression(body=v),str(verifier),'eval')
assert f1expr is not None

paths = {'baseline': OUT/'baseline_predictions_frozen.jsonl', 'public': TEST/'predictions_frozen.jsonl'}
protected = {str(p):sha(p) for p in [evaluator,verifier,TEST/'INTERNAL_GOLD.jsonl',*paths.values()]}
results = {}
for name,pred_path in paths.items():
    ns = dict(vars(module))
    ns.update(OUT=TEST,pred_path=pred_path)
    exec(metric_code,ns)
    r = ns['report']
    ns['supported_labels'] = sorted(set(ns['truth']))
    # Keep metrics, not the evaluator's historical claims about a new blind test.
    results[name] = {k:r[k] for k in ['cases','gold_distribution','prediction_distribution','strict_accuracy',
        'strict_macro_f1_all_3_labels','classification_report_strict','confusion_matrix','operational_policy',
        'operational_accuracy','subsets','strict_errors','predictions_sha256','internal_gold_sha256']}
    results[name]['macro_f1_supported_labels'] = eval(f1expr,ns)
    results[name]['positive_warning_recall'] = r['subsets']['POSITIVE']['operational_accuracy']

original = json.loads((TEST/'FINAL_REPORT.json').read_text(encoding='utf-8'))
for key in ['strict_accuracy','strict_macro_f1_all_3_labels','operational_accuracy','confusion_matrix','subsets']:
    assert results['public'][key] == original[key],key
assert all(sha(Path(p))==h for p,h in protected.items())
report = {'status':'EXACT_PUBLIC_METRIC_CODE_APPLIED_TO_BOTH_MODELS',
    'evaluation_type':'POST_HOC_COMPARATIVE_RESCORING_OF_FROZEN_PREDICTIONS',
    'evaluator_sha256':sha(evaluator),'verifier_sha256':sha(verifier),
    'prediction_or_model_changes':False,'reference_files_unchanged':True,'results':results}
(OUT/'EXACT_PUBLIC_SCORING.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines = ['# So sánh theo đúng cách chấm public','',
    'Chạy nguyên khối tính metric trong `github_publish/evaluation/independent_test_200/evaluate.py` cho cả hai prediction đã khóa. Macro-F1 hai nhãn dùng nguyên biểu thức từ `github_publish/scripts/verify_release.py`. Không train lại, không đổi prediction, model hay ngưỡng. Kết quả public khớp lại báo cáo gốc cả ma trận nhầm lẫn và từng nhóm.', '',
    '| Chỉ số cùng cách báo cáo README public | XLM-R chỉ học risk | Public |', '|---|---:|---:|']
for label,key in [('Strict accuracy','strict_accuracy'),('Macro-F1 trên hai nhãn có support','macro_f1_supported_labels'),
    ('Operational accuracy','operational_accuracy'),('Positive được cảnh báo vàng/đỏ (recall cảnh báo)','positive_warning_recall')]:
    a,b = (results[n][key] for n in ('baseline','public'))
    lines.append(f'| {label} | {a:.2%} | {b:.2%} |')
lines += ['', 'Quy tắc public:',
    '- Strict: dự đoán phải trùng gold. REVIEW không được đổi thành HIGH_EVIDENCE. Macro-F1 hai nhãn chỉ bỏ REVIEW khỏi phép lấy trung bình vì gold không có lớp này.',
    '- Operational: POSITIVE chấp nhận REVIEW/HIGH_EVIDENCE; HARD_NEGATIVE chấp nhận NO_EVIDENCE/REVIEW; EASY_NEGATIVE chỉ chấp nhận NO_EVIDENCE.',
    '- Recall cảnh báo: số POSITIVE được dự đoán REVIEW/HIGH_EVIDENCE chia 70.', '',
    'Baseline đúng strict 114/200, đúng operational 137/200, cảnh báo 64/70 positive. Public tương ứng 153/200, 173/200, 63/70.', '',
    'Bảng này là bảng đối chiếu chính theo cách báo cáo public. Bảng gộp REVIEW thành ALERT trong phân tích bổ sung trước đó không phải chính sách operational public.', '',
    'Giới hạn: một seed, đánh giá so sánh lại trên test đã xem; public kế thừa cả action và semantic pretraining. Không gọi đây là test blind mới hoặc kết luận tác dụng riêng của semantic.']
(OUT/'SO_SANH_DUNG_CACH_CHAM_PUBLIC.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Verified exact public scoring for both models; reports saved; public files unchanged.')
