"""Executable-Python semantic assignment-tracing benchmark.

Program text is legitimate Python, verified with exec. Model input is a
structured two-token-per-assignment representation of that program (syntax
markers for scope/call/return), NOT pretrained Python source, not RULER.
Tests include RHS-value snapshot (x=y is a copy), reassignment, lexical
shadowing via a function, and held-out chain length. No target leakage:
only one final query token is supervised per example. Validation never used
for gradient updates and has a separate RNG seed.

The optional 'recency' arm adds a tiny bias only among near-maximum pointer
logits; this is an ablation, not a proof of exact lexical tie handling.
"""
import argparse, importlib.util, json, os, random, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

p=argparse.ArgumentParser()
p.add_argument("--arm",choices=["rope","nope","onehop","bln","bln-recent"],required=True)
p.add_argument("--seed",type=int,default=0)
p.add_argument("--steps",type=int,default=1500)
p.add_argument("--batch",type=int,default=32)
p.add_argument("--width",type=int,default=64)
p.add_argument("--layers",type=int,default=4)
p.add_argument("--lr",type=float,default=0.003)
p.add_argument("--eval_every",type=int,default=150)
p.add_argument("--eval_per_kind",type=int,default=96)
p.add_argument("--out",default="semantic-result.json")
a=p.parse_args()
torch.manual_seed(a.seed)
torch.set_num_threads(2)
rng=np.random.default_rng(875+a.seed)
val_rng=np.random.default_rng(984711)

TOKENS=["PAD","BEGIN","END","RETURN","LOCAL","CALL","QUERY"]+list("abcdefgh")+[str(k) for k in range(20,36)]
ENC={name:i for i,name in enumerate(TOKENS)}
V=len(TOKENS)
KINDS=["chain3","chain5","reassign","shadow","mixed","chain7"]

def make_program(gen,kind):
    symbols=[str(s) for s in gen.permutation(list("abcdefgh"))]
    a0,b0,c0,d0,e0,f0,g0,h0=symbols
    vals=[int(z) for z in gen.choice(np.arange(20,36),size=4,replace=False)]
    v0,v1,v2,v3=vals
    if kind.startswith("chain"):
        n=int(kind[-1])
        names=symbols[:n+1]
        lines=[f"{names[0]} = {v0}"]
        for j in range(1,n+1):
            lines.append(f"{names[j]} = {names[j-1]}")
            if j in (1,3,5) and n>=5:
                lines.append(f"{names[j-1]} = {vals[j % 3 + 1]}")
        query=names[-1]
        expected=v0
    elif kind=="reassign":
        lines=[f"{a0} = {v0}",f"{b0} = {a0}",
               f"{a0} = {v1}",f"{c0} = {b0}",
               f"{a0} = {v2}",f"{d0} = {c0}",
               f"{e0} = {d0}"]
        query=e0;expected=v0
    elif kind=="mixed":
        lines=[f"{a0} = {v0}",f"{b0} = {a0}",
               f"{a0} = {v1}",f"{c0} = {a0}",
               f"{d0} = {b0}",f"{e0} = {c0}",
               f"{a0} = {v2}"]
        if int(gen.integers(0,2)):
            query=d0;expected=v0
        else:
            query=e0;expected=v1
    elif kind=="shadow":
        lines=[f"{a0} = {v0}",f"{b0} = {a0}",
               "def fn():",
               f"    {a0} = {v1}",
               f"    {c0} = {a0}",
               f"    {a0} = {v2}",
               f"    return {c0}",
               f"{d0} = fn()",f"{e0} = {b0}"]
        if int(gen.integers(0,2)):
            query=d0;expected=v1
        else:
            query=e0;expected=v0
    else:
        raise ValueError(kind)
    source="\n".join(lines)+"\n"
    env={"__builtins__":{}}
    exec(compile(source,"<audited_generated_program>","exec"),env)
    observed=env[query]
    assert observed==expected,(source,query,expected,observed)
    return lines,query,expected,source

def encode_program(lines,query,max_len=80):
    result=[]
    for line in lines:
        in_func=line.startswith("    ")
        t=line.strip()
        if t.startswith("def fn():"):
            result.append("BEGIN")
        elif t.startswith("return "):
            _,var=t.split()
            result.extend(["RETURN",var,"END"])
        else:
            left,right=[s.strip() for s in t.split("=",1)]
            if in_func:result.append("LOCAL")
            if right=="fn()":
                result.extend([left,"CALL"])
            else:
                assert right in ENC
                result.extend([left,right])
    assert all(z in ENC for z in result),result
    assert len(result)+2<=max_len,(len(result),max_len)
    result+=["PAD"]*(max_len-len(result)-2)+["QUERY",query]
    return [ENC[z] for z in result]

def generate_batch(gen,n,kinds):
    x=[];y=[];kt=[]
    for _ in range(n):
        kind=str(gen.choice(kinds))
        lines,query,value,_=make_program(gen,kind)
        x.append(encode_program(lines,query))
        y.append(ENC[str(value)])
        kt.append(kind)
    return torch.tensor(x,dtype=torch.long),torch.tensor(y,dtype=torch.long),kt

# Verify all categories with an independent Python interpreter before training.
for kind in KINDS:
    for _ in range(100):
        lines,q,v,_=make_program(val_rng,kind)
        assert TOKENS[ENC[str(v)]]==str(v)
        assert len(encode_program(lines,q))==80
# Fresh fixed validation RNG; no dependence on train draws.
val_rng=np.random.default_rng(712931)
val={}
for kind in KINDS:
    vx,vy,_=generate_batch(val_rng,a.eval_per_kind,[kind])
    val[kind]=(vx,vy)

sys.argv=["pdg_semantic_generated.py","--dev","cpu","--threads","2",
          "--task","mqar","--L","80","--V","256","--pairs","8",
          "--width",str(a.width),"--layers",str(a.layers),"--heads","4",
          "--seed",str(a.seed),"--init","0.5","--sink","0","--alibi","0",
          "--nope",str(int(a.arm=="nope")),
          "--tie_recent",str(0.12 if a.arm=="bln-recent" else 0.0),
          "--tie_tol","0.04"]
if a.arm in ("bln","bln-recent"):
    sys.argv+=["--eqh","2","--eqt","3","--ptr","11","--pnb","3"]
elif a.arm=="onehop":
    sys.argv+=["--eqh","1","--eqt","3","--ptr","1","--pnb","3"]
path=Path(__file__).resolve().parents[1]/"pdg_semantic_generated.py"
spec=importlib.util.spec_from_file_location("pdg_semantic_model",path)
model_mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_mod)
torch.manual_seed(a.seed)
model=model_mod.Model(V,a.width,a.layers,4).to("cpu")
optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01,betas=(0.9,0.95))
def evaluate():
    model.eval();details={}
    with torch.no_grad():
        for name,(vx,vy) in val.items():
            good=0;tot=0;ls=0.
            for offset in range(0,len(vy),32):
                xx=vx[offset:offset+32]
                yy=vy[offset:offset+32]
                out=model(xx)[:,-1,:]
                good+=int((out.argmax(-1)==yy).sum())
                tot+=len(yy)
                ls+=float(F.cross_entropy(out.float(),yy,reduction="sum"))
            details[name]=dict(acc=good/tot,loss=ls/tot)
    model.train()
    return details

start=time.time();curve=[]
for step in range(a.steps):
    sc=min((step+1)/max(1,a.steps//50),1.0,(a.steps-step)/max(1,a.steps//10))
    for group in optimizer.param_groups:group["lr"]=a.lr*sc
    x,y,_=generate_batch(rng,a.batch,["chain3","chain5","reassign","shadow","mixed"])
    pred=model(x)[:,-1,:]
    loss=F.cross_entropy(pred.float(),y)
    loss.backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    if (step+1)%a.eval_every==0 or step+1==a.steps:
        d=evaluate()
        macro=sum(v["acc"] for v in d.values())/len(d)
        row=dict(step=step+1,macro=macro,train_loss=float(loss.detach()),by_kind=d)
        curve.append(row)
        print("SEMANTIC_PROGRESS",a.arm,a.seed,step+1,round(macro,4),
              "REASSIGN",round(d["reassign"]["acc"],4),"SHADOW",round(d["shadow"]["acc"],4),
              "CHAIN7",round(d["chain7"]["acc"],4),flush=True)
out=dict(arm=a.arm,seed=a.seed,steps=a.steps,batch=a.batch,optimizer="AdamW",
         lr=a.lr,params=sum(p.numel() for p in model.parameters()),
         protocol="Real Python exec verified synthetic programs; normalized key/value tokens; single final query supervision",
         scope="assignment semantics not full production source code",
         train_types=["chain3","chain5","reassign","shadow","mixed"],
         val_types=KINDS,heldout=["chain7"],chance=1/16,
         recency="near-max attention score tie bonus" if a.arm=="bln-recent" else "none",
         train_seconds=time.time()-start,final=curve[-1],curve=curve)
Path(a.out).write_text(json.dumps(out,indent=2))
print("SEMANTIC_FINAL="+json.dumps({k:out[k] for k in ["arm","seed","train_seconds","params","final"]}),flush=True)
