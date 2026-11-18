"""Independent restricted-Python symbolic ceiling; NO exec/eval on source.

Uses source AST and a lexically scoped environment; assignments copy values,
closures capture frames, global/nonlocal update the right binding, and locals
are statically declared even before the first assignment. Supports the
benchmark's restricted grammar only. This is an explicit *symbolic oracle
comparator*, not a heuristic, neural model, or general Python interpreter.
"""
import ast
from dataclasses import dataclass,field
UNBOUND=object()

@dataclass
class Env:
    parent: "Env | None"=None
    values: dict=field(default_factory=dict)
    locals: set=field(default_factory=set)
    globals: set=field(default_factory=set)
    nonlocals: set=field(default_factory=set)
    def module(self):
        e=self
        while e.parent:e=e.parent
        return e
    def get(self,n):
        if n in self.globals:return self.module().get(n)
        if n in self.nonlocals:
            e=self.parent
            while e and e.parent:
                if n in e.locals:return e.get(n)
                e=e.parent
            raise NameError(n)
        if n in self.locals:
            v=self.values.get(n,UNBOUND)
            if v is UNBOUND:raise UnboundLocalError(n)
            return v
        if n in self.values:return self.values[n]
        if self.parent:return self.parent.get(n)
        raise NameError(n)
    def put(self,n,v):
        if n in self.globals:self.module().values[n]=v;return
        if n in self.nonlocals:
            e=self.parent
            while e and e.parent:
                if n in e.locals:e.put(n,v);return
                e=e.parent
            raise NameError(n)
        self.values[n]=v

def declarations(fn):
    names={x.arg for x in fn.args.posonlyargs+fn.args.args+fn.args.kwonlyargs}
    glob=set();nonloc=set()
    def visit(t):
        if isinstance(t,ast.FunctionDef):names.add(t.name);return
        if isinstance(t,ast.Global):glob.update(t.names);return
        if isinstance(t,ast.Nonlocal):nonloc.update(t.names);return
        if isinstance(t,ast.Name) and isinstance(t.ctx,ast.Store):names.add(t.id)
        for x in ast.iter_child_nodes(t):visit(x)
    for child in fn.body:visit(child)
    return names-glob-nonloc,glob,nonloc

@dataclass
class Function:
    node: ast.FunctionDef
    captured: Env
    local: set
    glob: set
    nonloc: set

class Returned(Exception):
    def __init__(self,value):self.value=value

class Script:
    def __init__(self):self.printed=[]
    def expr(self,x,e):
        if isinstance(x,ast.Constant):return x.value
        if isinstance(x,ast.Name):return e.get(x.id)
        if isinstance(x,ast.Call):
            if isinstance(x.func,ast.Name) and x.func.id=="print":
                assert len(x.args)==1
                self.printed.append(self.expr(x.args[0],e));return None
            f=self.expr(x.func,e)
            assert isinstance(f,Function)
            args=[self.expr(q,e) for q in x.args]
            assert len(args)==len(f.node.args.args) and not x.keywords
            local=Env(parent=f.captured,locals=f.local,globals=f.glob,nonlocals=f.nonloc)
            for q,v in zip(f.node.args.args,args):local.put(q.arg,v)
            try:self.block(f.node.body,local)
            except Returned as r:return r.value
            return None
        if isinstance(x,ast.IfExp):
            return self.expr(x.body if self.expr(x.test,e) else x.orelse,e)
        if isinstance(x,ast.Compare) and len(x.ops)==len(x.comparators)==1:
            a,b=self.expr(x.left,e),self.expr(x.comparators[0],e)
            if isinstance(x.ops[0],ast.Eq):return a==b
            if isinstance(x.ops[0],ast.NotEq):return a!=b
        raise NotImplementedError(ast.dump(x))
    def block(self,body,e):
        for x in body:
            if isinstance(x,ast.Assign):
                assert len(x.targets)==1 and isinstance(x.targets[0],ast.Name)
                e.put(x.targets[0].id,self.expr(x.value,e))
            elif isinstance(x,ast.FunctionDef):
                l,g,n=declarations(x)
                e.put(x.name,Function(x,e,l,g,n))
            elif isinstance(x,ast.Expr):self.expr(x.value,e)
            elif isinstance(x,ast.Return):
                raise Returned(self.expr(x.value,e) if x.value else None)
            elif isinstance(x,(ast.Global,ast.Nonlocal,ast.Pass)):pass
            else:raise NotImplementedError(ast.dump(x))
    def run(self,source):
        self.block(ast.parse(source).body,Env())
        assert len(self.printed)==1 and type(self.printed[0]) is int
        return self.printed[0]

def predict(source):return Script().run(source)

if __name__=="__main__":
    import runpy
    cases=runpy.run_path(__file__.replace("ast_reference.py","test_scope_semantics.py"))
    for key,(source,expected) in cases["CASES"].items():
        # Unlike the single-print benchmark, these examples may have two prints.
        tree=ast.parse(source)
        engine=Script();engine.block(tree.body,Env())
        assert engine.printed==expected,(key,engine.printed,expected)
    try:predict(cases["UNBOUND_LOCAL"])
    except UnboundLocalError:pass
    else:raise AssertionError("forgot static-local Python rule")
    print("AST_SYMBOLIC_ORACLE_PASSES "+str(len(cases["CASES"])+1)+" independent CPython counterexamples")
