"""Rank hunt3 top-k screen configs by judge scores and pick the finals.

usage: python3 -I analyze_v3.py <results.json> <judge.json> <finals.json>

Prints, for each persona, the best style score at each k (best over pool and total dose), then picks finals:
the top 2 configs by style, the top config by keep, and the smallest k reaching 80% of the best style.
  style = mean(concept * fluent)                 sounds like X and stays readable
  keep  = mean(concept * min(fluent, on_topic))  ...and still answers the question
"""
import json
import sys
from collections import defaultdict

res, rj, dst = sys.argv[1:4]
r, j = json.load(open(res)), json.load(open(rj))
C = [c for c in j["concepts"] if c in r["concepts"]]
groups = defaultdict(list)
for it in j["items"]:
    if "s" in it and it["src"] == "screen":
        groups[it["idx"]].append(it["s"])


def agg(ss, c):
    k = len(ss)
    return {"style": sum(s[c] * s["fluent"] for s in ss) / k / 100,
            "keep": sum(s[c] * min(s["fluent"], s["on_topic"]) for s in ss) / k / 100,
            "concept": sum(s[c] for s in ss) / k / 10, "fluent": sum(s["fluent"] for s in ss) / k / 10,
            "on_topic": sum(s["on_topic"] for s in ss) / k / 10}


rows = []
for i, cfg in enumerate(r["screen"]):
    if groups[i]:
        rows.append({"i": i, **cfg, **{"by": {c: agg(groups[i], c) for c in C}}})

print("controls:")
for x in rows:
    if x["concept"] == "control":
        top = max(C, key=lambda c: x["by"][c]["style"])
        print(f"   {x['neurons']}  top {top} style {x['by'][top]['style']:.2f} on {x['by'][top]['on_topic']:.2f}")

KS = sorted({x["k"] for x in rows if "k" in x})
print("\nbest style by k (max over pool and total):")
print(f"{'':12}" + "".join(f"{'k=' + str(k):>8}" for k in KS))
for c in C:
    own = [x for x in rows if x["concept"] == c]
    print(f"{c:12}" + "".join(f"{max([x['by'][c]['style'] for x in own if x['k'] == k] or [0]):8.2f}" for k in KS))

finals, seen = [], set()
for c in C:
    own = sorted([x for x in rows if x["concept"] == c], key=lambda x: -x["by"][c]["style"])
    if not own:
        continue
    best = own[0]["by"][c]["style"]
    picks = own[:2] + [max(own, key=lambda x: x["by"][c]["keep"])]
    small = [x for x in sorted(own, key=lambda x: (x["k"], -x["by"][c]["style"])) if x["by"][c]["style"] >= 0.8 * best]
    picks += small[:1]
    print(f"\n{c}")
    for x in own[:4]:
        a = x["by"][c]
        print(f"   {x['pool']:5} k={x['k']:<2} T={x['total']:<3} style {a['style']:.2f} keep {a['keep']:.2f} "
              f"conc {a['concept']:.2f} flu {a['fluent']:.2f} on {a['on_topic']:.2f}")
    for x in picks:
        if x["i"] in seen or x["by"][c]["style"] < 0.05:
            continue
        seen.add(x["i"])
        finals.append({"concept": c, "kind": f"{x['pool']} k={x['k']} T={x['total']}", "pool": x["pool"],
                       "k": x["k"], "total": x["total"], "neurons": x["neurons"]})
json.dump(finals, open(dst, "w"))
print(f"\n{len(finals)} finals configs -> {dst}")
