"""All generator correctness checks must pass before training even one model."""
import unittest
from types import SimpleNamespace
import numpy as np
from audit_stream_task import make_stream, oracle_groups, specs
from audit_stream_eval import passes_learning_gate


def args():
    return SimpleNamespace(L=96,pairs=4,stream_writes=6,stream_new=2,
                           stream_alias=.5,tv=256)


def generate(B=128,seed=331):
    a=args()
    rng=np.random.default_rng(seed)
    seq=rng.integers(1,a.tv,size=(B,a.L+1))
    mask=np.zeros((B,a.L),bool)
    return make_stream(seq,mask,rng,a)


class RandomStreamTests(unittest.TestCase):
    def test_determinism_and_causal_semantics(self):
        a=args()
        seq,mask=generate()
        seq2,mask2=generate()
        self.assertTrue(np.array_equal(seq,seq2))
        self.assertTrue(np.array_equal(mask,mask2))
        groups,stale,counts=oracle_groups(seq,mask,a)
        self.assertEqual(counts["new"],128*a.stream_new)
        self.assertEqual(counts["overwrites"],128*(a.stream_writes-a.stream_new))
        self.assertEqual(counts["queries"],128*(a.L//3-a.pairs-a.stream_writes))
        self.assertTrue(30<counts["alias"]<counts["new"])
        self.assertGreater(int(groups["after_overwrite"].sum()),100)
        self.assertGreater(int(groups["new_key"].sum()),100)
        self.assertGreater(int(groups["old_untouched"].sum()),100)
        self.assertTrue(np.array_equal(groups["all"],mask))
        self.assertTrue(np.all(seq[:,-1]==0))
        self.assertTrue(np.all(stale[groups["after_overwrite"]]>=a.tv//2))
        self.assertTrue(np.all(stale[~groups["after_overwrite"]]==-1))

    def test_write_query_orders_are_not_fixed(self):
        seq,_=generate()
        # Exclude mandatory initial writes: each row has its own shuffled
        # write/query record order and no common second write phase.
        tokens=(seq[:,3*args().pairs:args().L:3])
        self.assertGreater(len({tuple(row) for row in tokens}),20)
        self.assertTrue((tokens==args().tv-1).any(axis=1).all())

    def test_query_answers_cannot_be_shuffled_without_detection(self):
        a=args()
        seq,mask=generate(B=16)
        bad=seq.copy()
        pos=np.flatnonzero(mask[0])[0]+1
        bad[0,pos]=(bad[0,pos]+7-a.tv//2)%(a.tv//2-2)+a.tv//2
        with self.assertRaises(AssertionError):
            oracle_groups(bad,mask,a)

    def test_invalid_shapes_and_mixed_vocab(self):
        a=args()
        a.L=97
        with self.assertRaises(ValueError):
            specs(a)
        a=args();a.stream_new=9
        with self.assertRaises(ValueError):
            specs(a)

    def test_learning_gate_requires_every_group(self):
        base={"scores":{
          "all":{"n":1000,"acc":.92},
          "old_untouched":{"n":250,"acc":.90},
          "after_overwrite":{"n":300,"acc":.84},
          "new_key":{"n":200,"acc":.80}}}
        self.assertTrue(passes_learning_gate(base))
        base["scores"]["new_key"]["acc"]=.40
        self.assertFalse(passes_learning_gate(base))

if __name__=="__main__":
    unittest.main()
