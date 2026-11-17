"""Deterministic CPython binding counterexamples; no neural model is involved.

These are not training samples. They distinguish:
- lexical scopes from indentation depths or nearest textual same-name writes;
- dynamic value versions from static lexical binding IDs;
- closures' runtime environment instances from source occurrence positions;
- Python's static-local rule (UnboundLocalError).
All code is author-generated, terminating, and uses a restricted print-only
builtins dictionary. Never exec user-provided untrusted source here.
"""
from __future__ import annotations

def execute(source: str):
    out=[]
    builtins={"print":lambda x:out.append(x)}
    exec(compile(source,"<scope_oracle>","exec"),{"__builtins__":builtins})
    return out

CASES={
    "module_survives_local_shadow":("""
x=20
def f():
    x=21
    return x
f()
print(x)
""",[20]),
    "global_declaration_changes_binding":("""
x=20
def f():
    global x
    x=21
    return x
f()
print(x)
""",[21]),
    "assignment_copies_current_value":("""
x=20
y=x
x=21
print(y)
""",[20]),
    "closure_reads_latest_global":("""
x=20
def f():
    return x
x=21
print(f())
""",[21]),
    "nonlocal_mutates_outer":("""
def outer():
    x=20
    def inner():
        nonlocal x
        x=21
    inner()
    return x
print(outer())
""",[21]),
    "inner_local_does_not_mutate_outer":("""
def outer():
    x=20
    def inner():
        x=21
        return x
    inner()
    return x
print(outer())
""",[20]),
    "same_source_two_closure_instances":("""
def make(x):
    def inner():
        return x
    return inner
a=make(20)
b=make(21)
print(a())
print(b())
""",[20,21]),
    "sibling_scopes_same_depth":("""
def f():
    x=20
    return x
def g():
    x=21
    return x
print(f())
print(g())
""",[20,21]),
    "source_position_not_execution_time":("""
def make():
    def inner():
        return x
    return inner
x=20
a=make()
x=21
print(a())
""",[21]),
}

UNBOUND_LOCAL="""
x=20
def f():
    y=x
    x=21
    return y
f()
"""

def main():
    for name,(code,expected) in CASES.items():
        actual=execute(code)
        assert actual==expected,(name,actual,expected)
    try:
        execute(UNBOUND_LOCAL)
    except UnboundLocalError: pass
    else: raise AssertionError("static lexical local declaration was not honored")
    print("VERIFIED_SCOPE_CASES="+str(len(CASES)+1))
    print("GLOBAL_PAIR=(20,21)")
    print("CLOSURE_MULTI_FRAME=(20,21)")
    print("STATIC_LOCAL_EXCEPTION=UnboundLocalError")

if __name__=="__main__":main()
