"""Download only the pinned, original XLM-R base for the risk-only baseline."""
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
target = root / 'artifacts/base/xlm-roberta-base'
snapshot_download('FacebookAI/xlm-roberta-base',
    revision='e73636d4f797dec63c3081bb6ed5c7b0bb3f2089',local_dir=str(target),
    allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json',
        'special_tokens_map.json','sentencepiece.bpe.model'])
print(f'Original pretrained model downloaded to {target}')
