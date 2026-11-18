"""Purely independent matched-length and lexical-scope probe definitions.

Reads build_semantic_adversarial.py's generated script, so existing original
experiments remain exactly reproducible. All new programs are executed with
Python to verify their last queried value before creating training labels.
New OOD categories:
- unique8: same length as 8 writes but NO repeated definition
- latest8_interleaved: 4 writes of target interleaved with 4 distinct distractors
- latest8_plain: 8 target writes without extra alias copies
- scope_global_post: direct RHS copy of global after function's inner x definition
- scope_global_query: query global name after same-name local definition
"""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("build_semantic_adversarial.py")))
path=Path(__file__).with_name("semantic_adversarial_generated.py")
s=path.read_text()
def patch(old,new):
    global s
    assert s.count(old)==1,(old,s.count(old))
    s=s.replace(old,new,1)

patch('KINDS=["chain3","chain5","reassign","shadow","mixed","latest4","latest8","shadow2","chain7"]',
      'KINDS=["chain3","chain5","reassign","shadow","mixed","latest4","latest8","latest8_plain","latest8_interleaved","unique8","scope_global_post","scope_global_query","shadow2","chain7"]')
patch('    elif kind=="shadow2":', '''    elif kind=="unique8":
        # Eight different variable names, same number of statements as latest8_plain.
        allvals=[int(q) for q in gen.choice(np.arange(20,36),size=8,replace=False)]
        lines=[f"{name} = {val}" for name,val in zip(symbols,allvals)]
        query=symbols[-1];expected=allvals[-1]
    elif kind=="latest8_plain":
        # Eight repeated writes, no aliases or other distractors.
        allvals=[int(q) for q in gen.choice(np.arange(20,36),size=8,replace=False)]
        lines=[f"{a0} = {val}" for val in allvals]
        query=a0;expected=allvals[-1]
    elif kind=="latest8_interleaved":
        # 4 target writes, 4 other distinct-definition writes (8 total).
        allvals=[int(q) for q in gen.choice(np.arange(20,36),size=8,replace=False)]
        lines=[]
        for j in range(4):
            lines.append(f"{a0} = {allvals[2*j]}")
            lines.append(f"{symbols[j+1]} = {allvals[2*j+1]}")
        query=a0;expected=allvals[6]
    elif kind in ("scope_global_post","scope_global_query"):
        # Deliberate counterexample to globally newest same-name definition:
        # global x must NOT be replaced by the local x from the function body.
        lines=[f"{a0} = {v0}",
               "def fn():",
               f"    {a0} = {v1}",
               f"    return {a0}",
               f"{b0} = fn()"]
        if kind=="scope_global_post":
            lines.append(f"{c0} = {a0}")
            query=c0;expected=v0
        else:
            query=a0;expected=v0
    elif kind=="shadow2":''')
# Paths before import are unchanged; evaluates all new categories before training.
path.write_text(s)
print("wrote matched-length and lexical-scope OOD variants")
