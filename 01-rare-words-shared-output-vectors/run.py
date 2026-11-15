"""Run a block of grid.py configs one at a time in the foreground: python3 run.py blocks/b1.txt
Each non-empty line: <tag> <grid.py flags...>. Lines starting with # are skipped. Tags already in grid.jsonl are skipped."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
done = set()
if os.path.exists(os.path.join(HERE, "grid.jsonl")):
    done = {json.loads(l)["tag"] for l in open(os.path.join(HERE, "grid.jsonl"))}
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#"): continue
    tag, *flags = line.split()
    if tag in done: continue
    subprocess.run(["nice", "-n", "20", "python3", os.path.join(HERE, "grid.py"), "--tag", tag, *flags])
