"""Correctness audit of gated recurrence, not an accuracy benchmark.
Run: python -m unittest -v test_gate_exact.py
"""
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import mix
from gate_exact import exact_skip

def args(**kw):
    x = dict(lps=0.0, lpsf=1, lpsR=0, lpsc=0, lpsC=3, lpsQ=0,
             lpsE=0, lpsD=0, lpsM=0.0, lpsI=0, pairs=0, orq=-1.0)
    x.update(kw)
    return SimpleNamespace(**x)

def inputs(seed, code, T=9):
    torch.manual_seed(seed)
    B,H,D=2,2,4
    q=F.normalize(torch.randn(B,H,T,D,dtype=torch.float64),dim=-1)
    k=F.normalize(torch.randn(B,H,T,D,dtype=torch.float64),dim=-1)
    v=torch.randn(B,H,T,D,dtype=torch.float64)
    beta=torch.rand(B,H,T,dtype=torch.float64)*.5+.2
    la=-torch.rand(B,H,T,dtype=torch.float64)*.2-.01
    s=SimpleNamespace(training=True,
      erase=torch.rand(B,H,T,dtype=torch.float64)*.7+.1 if code=='e' else None,
      S={'on':torch.ones(1,H,D,dtype=torch.float64),
         'lpg':torch.zeros(1,H,dtype=torch.float64)})
    return s,torch.zeros(B,T,H*D,dtype=torch.float64),q,k,v,beta,la

class TestExactGate(unittest.TestCase):
    def test_no_gate_matches_existing_chunked(self):
        for code in ('g','d','D','e'):
            for seed in (0,1):
                s,h,q,k,v,beta,la=inputs(seed,code)
                for C in (1,3,9):
                    a=args(lpsC=C)
                    x=mix.chunked(s,a,0,h,code,q,k,v,beta,la)
                    y=exact_skip(s,a,0,h,code,q,k,v,beta,la)
                    self.assertTrue(torch.allclose(x,y,atol=1e-10,rtol=1e-10),
                       (code,seed,C,(x-y).abs().max().item()))

    def test_intra_gate_chunk_invariance(self):
        for code in ('g','d','D','e'):
            s,h,q,k,v,beta,la=inputs(2,code)
            ys=[mix.chunked(s,args(lps=.6,lpsC=C,lpsI=1),0,h,code,q,k,v,beta,la)
                for C in (1,3,9)]
            for y in ys[1:]:
                self.assertTrue(torch.equal(ys[0],y))

    def test_margin_stop_gradient_forward_equivalence(self):
        for code in ('g','d'):
            s,h,q,k,v,beta,la=inputs(3,code)
            x=exact_skip(s,args(lps=.6,lpsD=0,lpsM=.2),0,h,code,q,k,v,beta,la)
            y=exact_skip(s,args(lps=.6,lpsD=1,lpsM=.2),0,h,code,q,k,v,beta,la)
            self.assertTrue(torch.equal(x,y))

    def test_original_c1_gate_matches_self_consistent_recurrence(self):
        # C=1 is the independent exact-gate oracle already inside the
        # original mix.chunked implementation (no intra-chunk history).
        for code in ('g','d','D','e'):
            for seed in (0,1):
                s,h,q,k,v,beta,la=inputs(seed,code)
                a=args(lps=.7,lpsC=1)
                old=mix.chunked(s,a,0,h,code,q,k,v,beta,la)
                new=exact_skip(s,a,0,h,code,q,k,v,beta,la)
                self.assertTrue(torch.allclose(old,new,atol=1e-10,rtol=1e-10),
                    (code,seed,(old-new).abs().max().item()))

    def test_mqar2_phase_stats_are_distinct(self):
        # The second batch of facts is at [64,80) for L128, P16;
        # it must not be conflated with query tokens.
        s,h,q,k,v,beta,la=inputs(6,"g",T=128)
        s.training=False
        a=args(lps=.5,pairs=16,task="mqar2")
        exact_skip(s,a,0,h,"g",q,k,v,beta,la)
        st=s.gst[0]
        for name in ("factA_v","queryA_v","factB_v","queryAB_v"):
            self.assertIn(name,st)
            self.assertTrue(0<=st[name]<=1)
        self.assertNotIn("pv",st)  # standard MQAR stats are invalid here.

    def test_old_chunkstart_gate_is_not_exact(self):
        for code in ('g','d'):
            s,h,q,k,v,beta,la=inputs(4,code)
            a=args(lps=.7,lpsC=9)
            x=mix.chunked(s,a,0,h,code,q,k,v,beta,la)
            y=exact_skip(s,a,0,h,code,q,k,v,beta,la)
            self.assertGreater((x-y).abs().max().item(),1e-5)

    def test_exact_gate_backward_finite(self):
        for code in ('g','d'):
            s,h,q,k,v,beta,la=inputs(5,code)
            q=q.clone().requires_grad_(True)
            v=v.clone().requires_grad_(True)
            beta=beta.clone().requires_grad_(True)
            y=mix.chunked(s,args(lps=.8,lpsC=3,lpsI=1,lpsD=1,lpsM=.2),
                          0,h,code,q,k,v,beta,la)
            y.square().mean().backward()
            for name,t in (('q',q),('v',v),('beta',beta)):
                self.assertIsNotNone(t.grad,(code,name))
                self.assertTrue(torch.isfinite(t.grad).all(),(code,name))

if __name__=='__main__':
    unittest.main()
