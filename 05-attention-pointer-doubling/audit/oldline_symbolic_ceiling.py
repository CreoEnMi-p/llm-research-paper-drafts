"""Paper 05 exact programmatic ceiling and NON-equivalent chance priors.

Consumes the actual original pdg.py hopx generator, but ignores the target
value slot at prediction time. Derives the one-hop map solely from context
pairs and resolves first/all questions by repeated table lookup. This is a
scripted oracle upper bound, NOT a learnable neural baseline.

For N unique final answers and teacher-forced all-query context, a model that
can recognize the N candidate final values without following the query chain
can eliminate previously observed query answers. Its per-index expected
success is 1/(N-j), zero-index j. This is not vocabulary-uniform 1/V.
"""
import importlib.util,json,math,sys
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]/"pdg.py"
configs=[(7,8,2000),(15,8,2000),(63,4,1024),(63,8,2048),(127,4,1024)]
rows=[]
for hops,n,V in configs:
    sys.argv=["pdg.py","--task","hopx","--dev","cpu","--threads","2",
              "--hops",str(hops),"--pairs",str(n),"--tv",str(V),
              "--V",str(V),"--rev","1","--L","0",
              "--fill","0","--dup","0","--syn","0"]
    spec=importlib.util.spec_from_file_location(f"pdg_script_{hops}_{n}",p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    data,mask=m.make(512,np.random.default_rng(49821+hops+n))
    off=hops*n*2
    assert mask[:,off:off+2*n:2].all()
    assert not mask[:,:off].any()
    all_correct=0
    first_correct=0
    random_hits=np.zeros(n,dtype=np.int64)
    rng=np.random.default_rng(987+V)
    perindex_hits=np.zeros(n,dtype=np.int64)
    for row in data:
        # Build directed map from actual reverse-ordered context blocks.
        pairs=row[:off].reshape(-1,2)
        mapping={int(x):int(y) for x,y in pairs}
        assert len(mapping)==hops*n, "stage ranges must be unique"
        ends={int(x) for x,y in pairs if int(x)//((V-1)//(hops+1))>=hops-1}
        # Use all terminal tokens as candidate set instead, via map keys'
        # value whose id is not itself a map key.
        candidates={int(v) for v in mapping.values() if int(v) not in mapping}
        assert len(candidates)==n,(hops,n,candidates)
        prior=set()
        for j in range(n):
            key=int(row[off+2*j]);gold=int(row[off+2*j+1])
            cur=key
            for _ in range(hops):
                cur=mapping[cur]
            assert cur==gold,(hops,n,j,cur,gold)
            perindex_hits[j]+=1
            if j==0:first_correct+=1
            all_correct+=1
            # A deliberately *non-chain-solving* elimination policy: assumes
            # the terminal candidate set is available but cannot map queries.
            valid=sorted(candidates-prior)
            assert gold in valid and len(valid)==n-j
            guess=valid[int(rng.integers(0,len(valid)))]
            random_hits[j]+=(guess==gold)
            prior.add(gold)
    row=dict(k=hops,n_pairs=n,version_vocab=V,rows=len(data),
         script_exact_first_query_accuracy=first_correct/len(data),
         script_exact_all_query_accuracy=all_correct/(len(data)*n),
         script_per_query_accuracy=(perindex_hits/len(data)).tolist(),
         conditional_candidate_set_first_chance=1/n,
         conditional_candidate_elimination_by_query=[1/(n-j) for j in range(n)],
         conditional_candidate_elimination_mean=sum(1/i for i in range(1,n+1))/n,
         uniform_vocabulary_chance=1/V,
         empirical_no_chain_elimination_by_query=(random_hits/len(data)).tolist())
    rows.append(row)
print("SCRIPT_CEILING="+json.dumps(rows),flush=True)
Path("audit/oldline_symbolic_ceiling_results.json").write_text(json.dumps({
    "generator":"Unmodified pdg.py original hopx reverse-order, no fillers/repeated keys/synonyms; 512 rows per condition",
    "model":"script-only dictionary with repeated one-hop traversal, deterministic full 100 percent; not a learned baseline",
    "distinguish_chances":"1/V vocabulary uniform; 1/N first query with known candidates; H_N/N mean teacher-forced elimination with known candidates. All three are distinct.", "rows":rows},indent=2))
