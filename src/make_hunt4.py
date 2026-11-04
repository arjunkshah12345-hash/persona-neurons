"""Build hunt4.py (run 4: random-neuron control, system-prompt baseline, dense persona-vector baseline).

Takes hunt2.py's setup up to its stage switch and appends hunt4_tail.py. Random neurons are drawn here, so the
draw is recorded in the script: for each of run 2's 960 screened (neuron, dose) configs, a uniformly random neuron
from the same layer, never one of the 320 attributed candidates, steered at the same signed dose.

usage: python3 -I src/make_hunt4.py <out.py>    (run from the repo root)
"""
import json
import random
import sys

src = open("src/hunt2.py").read()
head = src[: src.index('if STAGE == "screen":')]
head = head.replace('STAGE = "screen"', 'STAGE = "controls"')
scr = json.load(open("results/v2/screen.json"))
cands = {(o["L"], o["n"]) for c in scr["candidates"].values() for o in c}
rng = random.Random(4)
pick, random_cfgs = {}, []
for cfg in scr["screen"]:
    if cfg["concept"] == "control":
        continue
    L, n, d = cfg["neurons"][0]
    if (L, n) not in pick:
        while True:
            m = rng.randrange(9216)
            if (L, m) not in cands and m not in {v[1] for k, v in pick.items() if k[0] == L}:
                break
        pick[(L, n)] = (L, m)
    random_cfgs.append({"concept": "random", "neurons": [[L, pick[(L, n)][1], d]], "matched": [L, n, d]})
consts = (f"RANDOM = {json.dumps(random_cfgs)}\n"
          "DENSE_LAYERS, DENSE_DOSES = [4, 8, 12], [0.5, 1.0, 2.0]\n\n")
open(sys.argv[1], "w").write(head + consts + open("src/hunt4_tail.py").read())
print(len(random_cfgs), "random configs,", len(pick), "random neurons")
