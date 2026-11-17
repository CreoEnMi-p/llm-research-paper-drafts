"""Adapt pinned Sanford terminal-only training to k=1/7 with recency variants.

Build under audit/ so original pdg.py, 20k Sanford run, and prior metrics
are unaffected. One generated model per CI job:
  - none: standard BLN
  - rank: near-top-logit recent rank (not exact ID)
  - exact: hard nearest-embedding decoder + exact-key-only tied recency
All use same Sanford get_batch and final position loss.
"""
from pathlib import Path
src=Path(__file__).with_name("sanford_terminal7.py").read_text()
def patch(old,new):
    global src
    n=src.count(old)
    assert n==1,(old[:110],n)
    src=src.replace(old,new,1)
patch('p.add_argument("--arm",choices=["rope","nope","bln","onehop","loop-nope"],required=True)',
      'p.add_argument("--arm",choices=["rope","nope","bln","onehop","loop-nope"],required=True)\n'
      'p.add_argument("--k",type=int,choices=[1,7],required=True)\n'
      'p.add_argument("--prior",choices=["none","rank","exact"],default="none")')
patch('sys.argv=["pdg_reviewer_generated.py","--dev","cpu","--threads","2",',
      'sys.argv=["pdg_semantic_generated.py","--dev","cpu","--threads","2",')
patch('          "--seed",str(a.seed),"--sink","0","--alibi","0",',
      '          "--seed",str(a.seed),"--sink","0","--alibi","0",\n'
      '          "--tie_recent",str(3.0 if a.prior!="none" else 0.0),"--tie_tol","0.04",')
patch('path=Path(__file__).resolve().parents[1]/"pdg_reviewer_generated.py"',
      'path=Path(__file__).resolve().parents[1]/"pdg_semantic_generated.py"')
patch('task_args=dict(seq_len=100,char_tokens=4,min_hops=7,max_hops=7)',
      'task_args=dict(seq_len=100,char_tokens=4,min_hops=a.k,max_hops=a.k)')
patch('x,y=task.get_batch(max(32,B),hops=7)',
      'x,y=task.get_batch(max(32,B),hops=a.k)')
patch('task="pinned official Sanford generator k=7, FINAL VALID token ONLY",',
      'task=f"pinned official Sanford generator k={a.k}, FINAL VALID token ONLY",\n'
      '         prior=a.prior,')
patch('     "only_supervised_position":-1', '     "only_supervised_position":-1') if False else None
out=Path(__file__).with_name("sanford_terminal_recency_generated.py")
out.write_text(src)
print("wrote",out)
