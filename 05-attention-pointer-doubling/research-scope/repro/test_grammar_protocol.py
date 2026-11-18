"""Integrity regression checks for the locked compositional grammar experiment."""
import hashlib,json,random,ast
from pathlib import Path
from grammar_compositions import sample,paired,make_dataset

def main():
    for combo in ('AB','CD','AC'):
        rng=random.Random(7891)
        for i in range(200):
            p=sample(rng,combo)
            assert p.composition==combo
            assert p.unique_literal_candidates==p.literal_occurrences==4
            assert p.truth in p.source_literals
            assert 'print(' in p.source and len(p.ids)<110
            ast.parse(p.source)
        rng=random.Random(21157)
        for i in range(100):
            left,right=paired(rng,combo)
            assert left.source.rsplit('print(',1)[0]==right.source.rsplit('print(',1)[0]
            assert left.truth!=right.truth
            assert left.pair_id==right.pair_id
    manifest=json.loads(Path('frozen_protocol_v2/PROTOCOL_MANIFEST.json').read_text())
    data=Path('frozen_protocol_v2/FROZEN_AC_TEST_256pairs.jsonl').read_bytes()
    assert hashlib.sha256(data).hexdigest()==manifest['frozen_test_sha256']
    assert manifest['sealed_evaluation_composition']=='AC'
    rows=[json.loads(line) for line in data.splitlines()]
    assert len(rows)==256
    assert len({row['pair_id'] for row in rows})==256
    assert all(row['left']['label']!=row['right']['label'] for row in rows)
    assert all(len(row['left']['literal_candidates'])==4 for row in rows)
    assert {row['left']['production'] for row in rows}=={'alias_crosses_local_shadow','alias_and_parameter_shadow','alias_nested_shadow','sibling_scope_alias'}
    print('GRAMMAR_INTEGRITY_PASSED 3*200 singles + 3*100 strict pairs + V2 SHA256 + 256 frozen pairs')

if __name__=='__main__':main()
