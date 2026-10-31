"""Find screen-stage neurons the old self-judge dropped that Sonnet rates highly.

usage: python3 -I analyze_screen.py <results.json> <rejudge-screen.json> <rejudge-summary.json> <out.json>

Each screen config (one neuron at one dose, 4 prompts) is scored on all 12 concepts, so a
neuron screened for "sarcastic" can surface as the best "anxious" neuron. Identical
(layer, neuron, dose) configs screened under several concepts are merged.
"""
import json
import sys
from collections import defaultdict

res, rjs, summ, dst = sys.argv[1:5]
r, j, s = json.load(open(res)), json.load(open(rjs)), json.load(open(summ))
C = list(j["concepts"])

by_cfg = defaultdict(list)
for it in j["items"]:
    if it["src"] == "screen" and "s" in it:
        L, n, d = r["screen"][it["idx"]]["neurons"][0]
        by_cfg[(L, n, d)].append((it, r["screen"][it["idx"]]["concept"]))

finals_neurons = defaultdict(set)  # (L, n) -> concepts it was promoted for
for f in r["finals"]:
    for L, n, _ in f["neurons"]:
        finals_neurons[(L, n)].add(f["concept"])

rows = []
for (L, n, d), pairs in by_cfg.items():
    its = [p[0]["s"] for p in pairs]
    k = len(its)
    row = {"spec": f"L{L} N{n} {d:+.2f}", "L": L, "n": n, "dose": d, "k": k,
           "screened_for": sorted({p[1] for p in pairs}),
           "promoted_for": sorted(finals_neurons.get((L, n), [])),
           "fluent": round(sum(x["fluent"] for x in its) / k / 10, 3),
           "on_topic": round(sum(x["on_topic"] for x in its) / k / 10, 3)}
    for c in C:
        row[c] = {"concept": round(sum(x[c] for x in its) / k / 10, 3),
                  "style": round(sum(x[c] * x["fluent"] for x in its) / k / 100, 3),
                  "strict": round(sum(x[c] * min(x["fluent"], x["on_topic"]) for x in its) / k / 100, 3)}
    rows.append(row)

out = {}
print(f"{len(rows)} unique screen configs\n")
for c in C:
    fin = s["winners"][c]["style"]
    top = sorted(rows, key=lambda x: -x[c]["style"])[:5]
    out[c] = {"finals_best": {"spec": fin["spec"], "style": fin["style"], "concept": fin["concept"]},
              "screen_top": [{k: v for k, v in x.items() if k not in C} | x[c] for x in top]}
    print(f"{c}  (finals best {fin['spec']} style {fin['style']:.2f} concept {fin['concept']:.2f})")
    for x in top:
        missed = "" if x["promoted_for"] else "  << never promoted"
        other = f" [screened for {','.join(x['screened_for'])}]" if c not in x["screened_for"] else ""
        print(f"   {x['spec']:16} style {x[c]['style']:.2f} conc {x[c]['concept']:.2f} strict {x[c]['strict']:.2f} "
              f"flu {x['fluent']:.2f} on {x['on_topic']:.2f} n={x['k']}{other}{missed}")
json.dump({"rows": rows, "by_concept": out}, open(dst, "w"), indent=1)
