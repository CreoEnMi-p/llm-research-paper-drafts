"""Grid driver for the pointer-doubling study (copied from idea-attn-identity-heads). Reads lines 'tag | script | args' from stdin (or a file argument),
runs each one at a time in the foreground with nice -n 20, and prints one summary line per run.
Arms are expanded with {ARM} macros, e.g. 'G1-h2 | rbg.py | --task hopx --hops 2 {ID}'.
Usage: python3 grid.py < block.txt
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REC = "--init 0.5 --stop 0.99 --L 0 --nev 256"   # recall-bed defaults (init 0.5: plain init 0.25 never learns recall)
ARMS = {
    "F": "--sink 1 --alibi 1 --lr 0.04",                                  # tuned plain attention: ALiBi + sink
    "KVS": "--sink 1 --alibi 1 --kvs 1 --lr 0.04",                         # KV shifting 2411.19574
    "PRI": "--sink 1 --alibi 1 --dconv 3 --lr 0.04",                       # Primer conv 2109.08668
    "NG": "--sink 1 --alibi 1 --ngh 2 --lr 0.04",                          # n-gram heads 2401.12973 (order 1)
    "NG2": "--sink 1 --alibi 1 --ngh 2 --ngo 2 --lr 0.04",                 # n-gram heads order 2
    "ID": "--sink 1 --alibi 1 --eqh 2 --eqt 3 --eqna 1 --lr 0.02",         # identity heads
    "PM": "--sink 1 --alibi 1 --eqh 2 --eqt 3 --eqna 1 --eqn2 0.001 --lr 0.02",   # pair-match identity heads
    "PTR": "--sink 1 --alibi 1 --eqh 2 --eqt 3 --eqna 1 --ptr 3 --lr 0.02",       # pointer identity heads
    "PNB": "--sink 1 --alibi 1 --eqh 2 --eqt 3 --eqna 1 --ptr 3 --pnb 3 --lr 0.02",   # one-hop pointer + no-backtrack (identity-heads paper)
    "DBL": "--sink 0 --alibi 0 --eqh 2 --eqt 3 --ptr 4 --lr 0.02",                 # pointer doubling P <- A @ P, head-0 value = pointer
    "DBLA": "--sink 1 --alibi 1 --eqh 2 --eqt 3 --eqna 1 --ptr 4 --lr 0.02",       # same with sink + ALiBi on the other heads
    "DNB": "--sink 0 --alibi 0 --eqh 2 --eqt 3 --ptr 4 --pnb 3 --lr 0.02",        # doubling + doubled no-backtrack
    "DB5": "--sink 0 --alibi 0 --eqh 2 --eqt 3 --ptr 5 --lr 0.02",                 # doubling without the pointer value (residual readout)
    "DSL": "--sink 0 --alibi 0 --eqh 2 --eqt 3 --ptr 9 --pself 0.33 --lr 0.02",   # doubling + self-loop
    "OSL": "--sink 0 --alibi 0 --eqh 2 --eqt 3 --ptr 10 --pself 0.33 --lr 0.02",  # one-hop pointer + self-loop (control)
    "BL": "--sink 0 --alibi 0 --eqh 2 --eqt 3 --ptr 11 --lr 0.02",                # binary lifting: doubling table + gated jump pointer
    "BLN": "--sink 0 --alibi 0 --eqh 2 --eqt 3 --ptr 11 --pnb 3 --lr 0.02",       # METHOD: binary lifting (doubling table + gated jump pointer) + no-backtrack
    "PRD": "--sink 0 --alibi 0 --eqh 2 --eqt 3 --ptr 6 --lr 0.02",       # predecessor doubling
    "PRG": "--sink 0 --alibi 0 --eqh 2 --eqt 3 --ptr 8 --lr 0.02",       # gated predecessor doubling
    # LM bed (atg.py): same lr 0.04 for all, 1 special head per layer as in IDEA.md
    "LF": "--sink 1 --alibi 1",
    "LPRI": "--sink 1 --alibi 1 --dconv 3",
    "LNG2": "--sink 1 --alibi 1 --ngh 1 --ngo 2",
    "LID": "--sink 1 --alibi 1 --eqh 1 --eqt 3",
    "LIDNA": "--sink 1 --alibi 1 --eqh 1 --eqt 3 --eqna 1",
    "LPM": "--sink 1 --alibi 1 --eqh 1 --eqt 3 --eqn2 0.001",
    "LPMNA": "--sink 1 --alibi 1 --eqh 1 --eqt 3 --eqn2 0.001 --eqna 1",
    "LPTRNA": "--sink 1 --alibi 1 --eqh 1 --eqt 3 --ptr 3 --eqna 1",
    "LKVS": "--sink 1 --alibi 1 --kvs 1",
    "LPTR": "--sink 1 --alibi 1 --eqh 1 --eqt 3 --ptr 3",
    "LBLN": "--sink 1 --alibi 1 --eqh 2 --eqt 3 --ptr 11 --pnb 3 --eqna 1",
    "LBLNPM": "--sink 1 --alibi 1 --eqh 3 --eqt 3 --ptr 11 --pnb 3 --eqna 1 --eqn2 0.001",
    "LDBL": "--sink 1 --alibi 1 --eqh 1 --eqt 3 --eqn2 0.001 --ptr 5",
    "LDSL": "--sink 1 --alibi 1 --eqh 1 --eqt 3 --eqn2 0.001 --ptr 9 --pself 0.33",
}
LM = "--width 64 --layers 4 --L 256 --batch 16 --steps 150"   # S5


def expand(s):
    for k, v in sorted(ARMS.items(), key=lambda kv: -len(kv[0])): s = s.replace("{" + k + "}", v)
    return s.replace("{REC}", REC).replace("{LM}", LM)


def last_result(tag):
    with open(os.path.join(HERE, "results.jsonl")) as f: lines = f.readlines()
    for l in reversed(lines[-50:]):
        r = json.loads(l)
        if r.get("tag") == tag: return r
    return None


def main():
    src = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    jobs = [l.strip() for l in src if l.strip() and not l.strip().startswith("#")]
    for j in jobs:
        tag, script, args = [x.strip() for x in j.split("|")]
        cmd = ["nice", "-n", "20", "python3", os.path.join(HERE, script), "--tag", tag] + expand(args).split()
        t0 = time.time()
        for attempt in range(4):   # the lag guard kills the biggest Python process when all sessions together pass 10 GB; rerun a killed run
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode >= 0: break
            print(f"{tag}: killed (signal {-p.returncode}), retry {attempt + 1} in 30 s", flush=True); time.sleep(30)
        if p.returncode != 0: print(f"{tag}: FAILED {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ''}", flush=True); continue
        import resource; gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e9   # max RSS of any finished child so far
        r = last_result(tag); extra = [l for l in p.stdout.splitlines() if l.startswith("  ")]
        if r and "acc" in r and r["acc"] is not None :
            print(f"{tag:32s} acc {r['acc']:.3f}  s90 {str(r['steps90']):>4s}  ran {r.get('steps_run', '-'):>4}  {time.time() - t0:5.1f}s  {gb:.1f}GB", flush=True)
        elif r: print(f"{tag:32s} val {r['val']:.4f}" + (f"  off {r['val_off']:.4f}" if 'val_off' in r else "") + (f"  rep {r['val_rep']:.4f}" if 'val_rep' in r else "") + f"  {time.time() - t0:5.1f}s  {gb:.1f}GB", flush=True)
        for l in extra: print("   " + l.strip(), flush=True)


if __name__ == "__main__":
    main()
