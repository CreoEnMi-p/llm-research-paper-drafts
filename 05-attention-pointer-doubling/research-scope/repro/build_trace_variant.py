"""Instrument strict CPU V2 without changing the registered trainer.

Print hashes of inputs, model weights, and prediction scores at steps 0, 1, 2,
10, 100, and 1000, per independent hosted runner. This is only to diagnose
nondeterminism; the V2 test is already consumed.
"""
from pathlib import Path
from runpy import run_path
here=Path(__file__).resolve().parent
run_path(str(here/"build_strict_cpu_variant.py"))
s=(here/"train_composition_strict_generated.py").read_text()
def patch(a,b):
    global s
    assert s.count(a)==1,("must find once",a[:150],s.count(a))
    s=s.replace(a,b,1)
# Leave all model computation untouched; insert only detached fingerprints.
patch("models=Net();optimizer=torch.optim.AdamW(models.parameters(),lr=a.lr,weight_decay=.01)",
"""models=Net()
def sha_tensor(t):
    import hashlib
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:20]
def sha_model():
    import hashlib
    h=hashlib.sha256()
    for p in models.parameters():h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:20]
print('TRACE_INIT='+json.dumps(dict(seed=a.seed,model=a.model,params=sha_model(),
  torch_rng=sha_tensor(torch.get_rng_state()),python_seed=a.seed)),flush=True)
optimizer=torch.optim.AdamW(models.parameters(),lr=a.lr,weight_decay=.01)""")
patch("""    logits=models(x,d)
    loss=F.cross_entropy(logits,y)""",
"""    logits=models(x,d)
    if step in (0,1,2,9,99,999):
        print('TRACE_PRE='+json.dumps(dict(step=step+1,model=a.model,seed=a.seed,
          input=sha_tensor(x),labels=sha_tensor(y),logits=sha_tensor(logits),
          params=sha_model(),torch_rng=sha_tensor(torch.get_rng_state()))),flush=True)
    loss=F.cross_entropy(logits,y)""")
patch("""    optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(models.parameters(),1);optimizer.step()
    if (step+1)%a.eval_every""",
"""    optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(models.parameters(),1);optimizer.step()
    if step in (0,1,2,9,99,999):
        print('TRACE_POST='+json.dumps(dict(step=step+1,model=a.model,seed=a.seed,
          params=sha_model(),torch_rng=sha_tensor(torch.get_rng_state()))),flush=True)
    if (step+1)%a.eval_every""")
out=here/"train_composition_trace_generated.py"
out.write_text(s)
print("TRACE_GENERATED",len(s))
