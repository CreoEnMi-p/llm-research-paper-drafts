"""Independent task-role mask and local-gate tests; oracle is not a model."""
import unittest
from types import SimpleNamespace
import torch
import torch.nn as nn
from audit_update_task import layout
import mix

class LocalAndRoleControlTests(unittest.TestCase):
    def test_role_mask_skips_only_query_answers(self):
        a=SimpleNamespace(task="mqar_rewrite",L=96,pairs=12)
        W,Q1,Q2,bd=layout(a.L,a.pairs)
        f=mix.phase_write_multiplier(a,96,torch.device("cpu"),torch.float64)
        self.assertEqual(int((f==0).sum()),Q1+Q2)
        skip=set(range(bd[1]+1,bd[2],2))|set(range(bd[3]+1,bd[4],2))
        self.assertEqual(set(torch.where(f==0)[0].tolist()),skip)
        # New required write VALUES and first-stage facts remain enabled.
        for pos in range(bd[2]+1,bd[3],2):
            self.assertEqual(f[pos].item(),1.)
        for pos in range(1,bd[1],2):
            self.assertEqual(f[pos].item(),1.)
        a.task="mqar_alias"
        self.assertTrue(torch.equal(f,mix.phase_write_multiplier(a,96,torch.device("cpu"),torch.float64)))

    def test_prefix_oracle_semantics_identical(self):
        a=SimpleNamespace(task="mqar_rewrite",L=96,pairs=12)
        entire=mix.phase_write_multiplier(a,96,torch.device("cpu"),torch.float64)
        for prefix in (1,10,24,25,26,54,55,66,67,70,95):
            part=mix.phase_write_multiplier(a,prefix,torch.device("cpu"),torch.float64)
            self.assertTrue(torch.equal(part,entire[:prefix]),prefix)

    def test_local_gate_is_learned_from_current_hidden_state(self):
        a=SimpleNamespace(sc=0,lps=0.,lnv=0.,mix="g",localS=1)
        S={}
        mix.init(S,a,1,2,32)
        self.assertEqual(tuple(S["localw"].shape),(1,2,32))
        self.assertTrue(torch.allclose(S["localb"],torch.full((1,2),2.2)))
        h=torch.randn(4,12,32)
        mult=torch.sigmoid(torch.einsum("btd,hd->bht",h,S["localw"][0])+S["localb"][0][None,:,None])
        self.assertGreater(mult.min().item(),0.8)
        mult.mean().backward()
        self.assertIsNotNone(S["localw"].grad)
        self.assertGreater(S["localw"].grad.abs().sum().item(),0.)

if __name__=="__main__":
    unittest.main()
