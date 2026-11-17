"""Prepare genuine open-source Python source corpora for PDLm's byte-LM data loader.

Train: CPython stdlib Lib/**/*.py, validation: requests/src/requests/**/*.py.
The source repositories are separate; the model sees real GitHub source text,
not generated facts. Task is next-byte language modeling, NOT semantic
multi-hop code execution. This distinction is intentional.
"""
import json, os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TRAIN=Path(os.environ["CODE_TRAIN_REPO"])
VAL=Path(os.environ["CODE_VAL_REPO"])
assert TRAIN.is_dir() and VAL.is_dir()
def read_files(root, pattern):
    pairs=[]
    for p in sorted(root.glob(pattern)):
        if p.is_file() and p.suffix==".py":
            b=p.read_bytes()
            if 40<=len(b)<=1000000:
                pairs.append((str(p.relative_to(root)),b))
    assert pairs, root
    return pairs

train=read_files(TRAIN,"Lib/**/*.py")
val=read_files(VAL,"**/*.py")
# Restrict validation to actual package files, not training, tests or vendored code.
val=[(n,b) for n,b in val if "/requests/" in "/"+n.replace("\\","/")+"/" or n.startswith("src/requests/")]
assert val, "requests package code not found"
assert len(train)>100 and len(val)>3
def join(pairs):
    return b"\n\n\n".join(b for _,b in pairs)
train_bytes=join(train)
val_bytes=join(val)
assert len(train_bytes)>2000000 and len(val_bytes)>10000
out=ROOT/"data"
out.mkdir(exist_ok=True)
np.save(out/"fineweb_train_000001_bytes.npy",np.frombuffer(train_bytes,dtype=np.uint8).copy())
np.save(out/"fineweb_val_000000_bytes.npy",np.frombuffer(val_bytes,dtype=np.uint8).copy())
manifest={
 "is_fineweb":False,"unit":"UTF-8 bytes","train_source":"CPython Lib Python files",
 "val_source":"psf/requests source Python files",
 "train_git_sha":os.environ.get("CODE_TRAIN_SHA"),
 "val_git_sha":os.environ.get("CODE_VAL_SHA"),
 "train_files":len(train),"val_files":len(val),
 "train_bytes":len(train_bytes),"val_bytes":len(val_bytes),
 "train_path_count":len(set(n for n,_ in train)),
 "val_path_count":len(set(n for n,_ in val)),
 "task":"next-byte prediction, not variable resolution"
}
(ROOT/"audit"/"code_corpus_manifest.json").write_text(json.dumps(manifest,indent=2))
print("CODE_CORPUS_MANIFEST="+json.dumps(manifest),flush=True)
