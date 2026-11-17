"""Isolate L5/L6 strong RoPE k1->k7 curriculum from original Sanford k7 runner."""
from pathlib import Path
p=Path(__file__).with_name("sanford_terminal7.py")
src=p.read_text()
def patch(old,new):
    global src
    n=src.count(old)
    assert n==1,(n,old[:90])
    src=src.replace(old,new,1)
patch('p.add_argument("--out",type=str,default="sanford-final.json")',
      'p.add_argument("--warmup-onehop",type=int,default=400)\np.add_argument("--out",type=str,default="sanford-final.json")')
patch('val_task=InductionHopsFullSequenceTask(**task_args,rng=np.random.RandomState(918276))',
      'val_task=InductionHopsFullSequenceTask(**task_args,rng=np.random.RandomState(918276))\ntrain_task_k1=InductionHopsFullSequenceTask(seq_len=100,char_tokens=4,min_hops=1,max_hops=1,rng=np.random.RandomState(1813+a.seed))')
patch('def valid_batch(task,B):','def valid_batch(task,B,hops=7):')
patch('x,y=task.get_batch(max(32,B),hops=7)','x,y=task.get_batch(max(32,B),hops=hops)')
patch('xx,yy,_=valid_batch(train_task,a.batch)',
      'hops=1 if step<a.warmup_onehop else 7\n    xx,yy,_=valid_batch(train_task_k1 if hops==1 else train_task,a.batch,hops=hops)')
patch('row=dict(step=step+1,acc=acc,val_loss=valloss,train_loss=float(loss.detach()))',
      'row=dict(step=step+1,acc=acc,val_loss=valloss,train_loss=float(loss.detach()),train_hops=hops)')
patch('res=dict(arm=a.arm,seed=a.seed,steps=a.steps,batch=a.batch,layers=a.layers,',
      'res=dict(arm=a.arm,seed=a.seed,steps=a.steps,batch=a.batch,layers=a.layers,warmup_onehop=a.warmup_onehop,')
out=Path(__file__).with_name("sanford_terminal_curriculum_generated.py")
out.write_text(src)
print("generated",out)
