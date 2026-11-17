"""Construct an adversarial held-out assignment task variant from the audited
Python-exec reference, without editing baseline results or model code.
"""
from pathlib import Path
base=Path(__file__).with_name("semantic_assignments.py")
s=base.read_text()
def patch(old,new):
    global s
    assert s.count(old)==1,(old[:90],s.count(old))
    s=s.replace(old,new,1)
patch('KINDS=["chain3","chain5","reassign","shadow","mixed","chain7"]',
      'KINDS=["chain3","chain5","reassign","shadow","mixed","latest4","latest8","shadow2","chain7"]')
patch('    elif kind=="reassign":',
'''    elif kind in ("latest4","latest8"):
        allvals=[int(q) for q in gen.choice(np.arange(20,36),size=8,replace=False)]
        n=4 if kind=="latest4" else 8
        lines=[]
        for i in range(n):
            lines.append(f"{a0} = {allvals[i]}")
            if i % 2 == 0 and i < n-1:
                aux=[b0,c0,d0,e0][i//2]
                lines.append(f"{aux} = {a0}")
        query=a0
        expected=allvals[n-1]
    elif kind=="shadow2":
        lines=[f"{a0} = {v0}",f"{b0} = {a0}",
               "def fn():",
               f"    {a0} = {v1}",
               f"    {c0} = {a0}",
               f"    {a0} = {v2}",
               f"    {d0} = {c0}",
               f"    return {d0}",
               f"{e0} = fn()",
               f"{a0} = {v3}",
               f"{f0} = {b0}"]
        if int(gen.integers(0,2)):
            query=e0;expected=v1
        else:
            query=f0;expected=v0
    elif kind=="reassign":''')
patch('["chain3","chain5","reassign","shadow","mixed"]','["chain3","chain5","reassign","shadow","mixed","latest4"]')
patch('str(0.12 if a.arm=="bln-recent" else 0.0)',
      'str(3.0 if a.arm=="bln-recent" else 0.0)')
out=Path(__file__).resolve().parents[1]/"semantic_adversarial_generated.py"
out.write_text(s)
print("wrote",out)
