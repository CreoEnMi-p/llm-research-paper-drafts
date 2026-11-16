"""Deterministic and semantic property tests for the two update tasks."""
import unittest
from types import SimpleNamespace
import numpy as np
from audit_update_task import make_update, layout


class UpdateDatasetTests(unittest.TestCase):
    def run_checks(self, task, T=96, N=12, seed=122):
        a=SimpleNamespace(task=task,pairs=N,tv=256)
        rng=np.random.default_rng(seed)
        B=32
        seq=rng.integers(1,a.tv,size=(B,T+1))
        mask=np.zeros((B,T),bool)
        seq,mask=make_update(seq,mask,rng,a)
        W,Q1,Q2,bd=layout(T,N)
        self.assertEqual(int(mask.sum()),B*(Q1+Q2))
        self.assertEqual(Q1,15)
        self.assertEqual(Q2,15)
        self.assertEqual(bd[-1],T)
        keys0, vals0 = seq[:,:2*N:2],seq[:,1:2*N:2]
        keys2, vals2 = seq[:,bd[2]:bd[3]:2],seq[:,bd[2]+1:bd[3]:2]
        qa_k, qa_v = seq[:,bd[1]:bd[2]:2],seq[:,bd[1]+1:bd[2]:2]
        qb_k, qb_v = seq[:,bd[3]:bd[4]:2],seq[:,bd[3]+1:bd[4]:2]
        for b in range(B):
            original=dict(zip(keys0[b],vals0[b]))
            self.assertEqual(len(original),N)
            second=dict(zip(keys2[b],vals2[b]))
            self.assertEqual(len(second),W)
            for k,v in zip(qa_k[b],qa_v[b]):
                self.assertEqual(original[k],v)
            for k,v in zip(qb_k[b],qb_v[b]):
                self.assertEqual(second.get(k,original.get(k)),v)
            changed=sum(k in second for k in qb_k[b])
            self.assertEqual(changed,Q2//2)
            if task=="mqar_rewrite":
                for k,v in second.items():
                    self.assertIn(k,original)
                    self.assertNotEqual(v,original[k])
            else:
                for k,v in second.items():
                    self.assertNotIn(k,original)
                    self.assertIn(v,original.values())
        rng2=np.random.default_rng(seed)
        seq2=rng2.integers(1,a.tv,size=(B,T+1))
        seq2,mask2=make_update(seq2,np.zeros((B,T),bool),rng2,a)
        self.assertTrue(np.array_equal(seq,seq2))
        self.assertTrue(np.array_equal(mask,mask2))

    def test_rewrite(self):
        self.run_checks("mqar_rewrite")

    def test_alias(self):
        self.run_checks("mqar_alias")


if __name__=="__main__":
    unittest.main()
