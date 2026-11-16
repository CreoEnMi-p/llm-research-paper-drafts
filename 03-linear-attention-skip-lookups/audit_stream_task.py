"""Randomly interleaved key-value reads, new writes, overwrites.

Minimal learnability-gated dataset; not an ICML benchmark.

A stream consists of triplets:
  WRITE key value  (store or overwrite key -> value)
  READ  key answer (predict the answer using only prefix ending at key)

The first N records initialize the memory. The remaining record types are
randomly permuted *per sequence*: no fixed second write or query phase.
Necessary external writes remain in input; query answers are scored only at
their key positions. Values must be in a disjoint vocabulary range.

For each random stream: fixed exactly NW late writes with NNEW new keys,
NW-NNEW overwrites, and the remaining records queries. Key identity, values,
write order, and read positions vary with every example. Half of new-key
writes reuse a value already in memory to test same-value aliases.
"""
import numpy as np

def specs(a):
    T=int(a.L); N=int(a.pairs); NW=int(a.stream_writes); NN=int(a.stream_new)
    V=int(a.tv)
    if T%3!=0 or N<2 or NW<=NN or NN<1 or N+NW>=T//3-5:
        raise ValueError("Need T divisible by 3, N>=2, 1<=new<writes, >=6 queries")
    if V<40 or N+NN>=V//2-1:
        raise ValueError("Need disjoint marker, key and value vocabulary.")
    return T,N,NW,NN,V

def make_stream(seq,mask,rng,a):
    T,N,NW,NN,V=specs(a)
    B=seq.shape[0]
    if seq.shape!=(B,T+1) or mask.shape!=(B,T):
        raise ValueError("Data buffers have wrong shape")
    W=V-2; Q=V-1; low=V//2; high=V-2
    seq[:,T]=0
    mask[:]=False
    R=T//3; n_other=R-N
    for b in range(B):
        keys=rng.choice(np.arange(1,V//2),size=N+NN,replace=False)
        mem={}; lastkind={}; prev_value={}
        choices=["query"]*(n_other-NW)+["write"]*NW
        rng.shuffle(choices)
        write_subtypes=["new"]*NN+["overwrite"]*(NW-NN)
        rng.shuffle(write_subtypes)
        wi=0; introduced=0
        for j in range(R):
            pos=3*j
            if j<N:
                ev="new_initial"
            elif choices[j-N]=="query":
                ev="query"
            else:
                ev=write_subtypes[wi]
                wi+=1
            if ev=="query":
                current=list(mem)
                # Favor modified/new keys in half the queries; still sample
                # all existing identities without using future targets.
                changed=[k for k in current if lastkind[k] in ("overwrite","new")]
                if changed and rng.random()<.45:
                    k=int(rng.choice(changed))
                else:
                    k=int(rng.choice(current))
                v=mem[k]
                seq[b,pos:pos+3]=(Q,k,v)
                mask[b,pos+1]=True
            else:
                if ev=="new_initial":
                    k=int(keys[j])
                elif ev=="new":
                    k=int(keys[N+introduced])
                    introduced+=1
                else:
                    k=int(rng.choice(list(mem)))
                if ev=="new" and rng.random()<float(a.stream_alias):
                    v=int(rng.choice(list(mem.values())))
                else:
                    v=int(rng.integers(low,high))
                if ev=="overwrite":
                    old=mem[k]
                    if v==old:
                        v=low+(v-low+1)%(high-low)
                    prev_value[k]=old
                mem[k]=v
                lastkind[k]=("new" if ev=="new" else ("initial" if ev=="new_initial" else "overwrite"))
                seq[b,pos:pos+3]=(W,k,v)
        assert wi==NW and introduced==NN
    return seq,mask


def oracle_groups(seqs,masks,a):
    """Independently replay visible facts and verify all labels causally."""
    T,N,NW,NN,V=specs(a)
    W,Q=V-2,V-1
    low,high=V//2,V-2
    B=len(seqs)
    groups={s:np.zeros((B,T),bool) for s in
            ("all","old_untouched","after_overwrite","new_key","other")}
    stale=np.full((B,T),-1,np.int64)
    events={"new":0,"overwrites":0,"alias":0,"queries":0,
            "max_active_keys":0,"valid":True}
    for b in range(B):
        mem={};lastkind={};prev_value={}; all_old_values=set()
        initial_keys=set()
        for j in range(T//3):
            pos=3*j; typ,k,v=map(int,seqs[b,pos:pos+3])
            assert 1<=k<V//2 and low<=v<high
            if typ==W:
                if j<N:
                    assert k not in initial_keys and k not in mem
                    initial_keys.add(k)
                if k in mem:
                    assert v!=mem[k],("no-op overwrite",b,j)
                    prev_value[k]=mem[k]
                    events["overwrites"]+=1
                    lastkind[k]="overwrite"
                else:
                    if j>=N:
                        events["new"]+=1
                        if v in mem.values():
                            events["alias"]+=1
                        lastkind[k]="new"
                    else:
                        lastkind[k]="initial"
                mem[k]=v
            elif typ==Q:
                assert k in mem,("query missing key",b,j)
                assert mem[k]==v,("incorrect causal label",b,j,k,v,mem[k])
                assert bool(masks[b,pos+1]),("wrong score position",b,j)
                groups["all"][b,pos+1]=True
                name={"initial":"old_untouched","overwrite":"after_overwrite",
                      "new":"new_key"}[lastkind[k]]
                groups[name][b,pos+1]=True
                events["queries"]+=1
                if lastkind[k]=="overwrite":
                    stale[b,pos+1]=prev_value[k]
            else:
                raise AssertionError(("unrecognized stream marker",typ))
            events["max_active_keys"]=max(events["max_active_keys"],len(mem))
        assert len(initial_keys)==N
        assert np.array_equal(masks[b],groups["all"][b]),("mask mismatch",b)
        # Exactly NN new late keys and NW-NN overwrites per row.
    assert events["new"]==B*NN,events
    assert events["overwrites"]==B*(NW-NN),events
    assert sum(int(v.sum()) for k,v in groups.items() if k!="all")==events["queries"]
    return groups,stale,events
