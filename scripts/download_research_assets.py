"""Download separately versioned research assets, verifying byte count and SHA256."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=['baseline','int8'])
    args=parser.parse_args()
    entries=json.loads((ROOT/'artifacts/research_assets.json').read_text(encoding='utf-8'))
    for key in (['baseline'] if args.kind=='baseline' else ['int8','tokenizer']):
        entry=entries[key]; path=ROOT/entry['destination']
        if path.exists():
            if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:
                raise SystemExit(f'Existing file differs; refusing to replace: {path}')
            print(f'Already verified: {path}');continue
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(path.suffix+'.download')
        try:
            with urlopen(entry['url'],timeout=120) as response,temp.open('wb') as f:
                while block:=response.read(4*1024*1024):f.write(block)
            if temp.stat().st_size!=entry['bytes'] or digest(temp)!=entry['sha256']:
                raise RuntimeError(f'Integrity check failed: {key}')
            os.replace(temp,path)
        finally:
            temp.unlink(missing_ok=True)
        print(f'Downloaded and verified: {path}')

if __name__=='__main__':main()
