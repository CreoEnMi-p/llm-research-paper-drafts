"""No training unless 12-key harder random stream passes exact-data replay."""
import unittest
from types import SimpleNamespace
import numpy as np
from audit_stream_task import make_stream,oracle_groups


class MoreKeysPreflight(unittest.TestCase):
    def test_harder_variable_stream_and_subgroups(self):
        a=SimpleNamespace(L=144,pairs=8,stream_writes=10,stream_new=4,
                          stream_alias=.5,tv=256)
        B=96;rng=np.random.default_rng(4141)
        x=np.zeros((B,a.L+1),np.int64)
        m=np.zeros((B,a.L),bool)
        x,m=make_stream(x,m,rng,a)
        g,stale,counts=oracle_groups(x,m,a)
        self.assertEqual(counts["queries"],B*30)
        self.assertEqual(counts["new"],B*4)
        self.assertEqual(counts["overwrites"],B*6)
        self.assertGreater(counts["alias"],B//2)
        self.assertTrue(all(g[z].sum()>=100 for z in
                        ("old_untouched","after_overwrite","new_key")))
        self.assertTrue(np.array_equal(g["all"],m))
        self.assertTrue((stale[g["after_overwrite"]]>=a.tv//2).all())
        self.assertEqual(np.unique(x[:,3*a.pairs:a.L:3],axis=0).shape[0],B)

if __name__=="__main__":
    unittest.main()
