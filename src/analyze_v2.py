"""Rank hunt2 screen configs by Sonnet scores and pick the finals.

usage: python3 -I analyze_v2.py <results.json> <rejudge.json> <finals.json>

Each config's samples are scored on every concept, so a neuron screened for one concept can win another.
  style = mean(concept * fluent)                 sounds like X and stays readable
  keep  = mean(concept * min(fluent, on_topic))  ...and still answers the question
Finals per concept: top 3 distinct neurons by style at 0.75x / 1x / 1.5x their best dose, plus the top-2 pair.
"""
import json
import sys
from collections import defaultdict

res, rj, dst = sys.argv[1:4]
r, j = json.load(open(res)), json.load(open(rj))
C = [c for c in j["concepts"] if c in r["concepts"]]

groups = defaultdict(list)
for it in j["items"]:
    if "s" in it:
        groups[(it["src"], it["idx"])].append(it["s"])


def agg(ss):
    k = len(ss)
    row = {"k": k, "fluent": sum(s["fluent"] for s in ss) / k / 10, "on_topic": sum(s["on_topic"] for s in ss) / k / 10}
    for c in C:
        row[c] = {"concept": sum(s[c] for s in ss) / k / 10,
                  "style": sum(s[c] * s["fluent"] for s in ss) / k / 100,
                  "keep": sum(s[c] * min(s["fluent"], s["on_topic"]) for s in ss) / k / 100}
    return row


def spec(neurons):
    return ", ".join(f"L{L} N{n} {d:+.2f}" for L, n, d in neurons)


base = agg(sum((groups[("baseline", i)] for i in range(len(r["baseline_samples"]))), []))
sec = "finals" if r.get("stage") == "finals" else "screen"
rows = []
for i, cfg in enumerate(r[sec]):
    if groups[(sec, i)]:
        rows.append({"spec": spec(cfg["neurons"]), "neurons": cfg["neurons"], "searched_for": cfg["concept"],
                     "attr": cfg.get("attr"), **agg(groups[(sec, i)])})

print(f"{sec}: {len(rows)} configs | baseline fluent {base['fluent']:.2f} on_topic {base['on_topic']:.2f} "
      f"max concept {max(base[c]['concept'] for c in C):.2f}\n")
print("controls / calibration:")
for x in rows:
    if x["searched_for"] == "control":
        top = max(C, key=lambda c: x[c]["style"])
        print(f"   {x['spec']:18} top {top:11} conc {x[top]['concept']:.2f} style {x[top]['style']:.2f} "
              f"flu {x['fluent']:.2f} on {x['on_topic']:.2f}")

finals, seen = [], set()
for c in C:
    rank = sorted(rows, key=lambda x: -x[c]["style"])
    print(f"\n{c}")
    for x in rank[:6]:
        tag = "" if x["searched_for"] == c else f" [searched for {x['searched_for']}]"
        print(f"   {x['spec']:30} style {x[c]['style']:.2f} keep {x[c]['keep']:.2f} conc {x[c]['concept']:.2f} "
              f"flu {x['fluent']:.2f} on {x['on_topic']:.2f}{tag}")
    picks, used = [], set()
    for x in rank:
        if len(picks) == 3 or x[c]["style"] < 0.05:
            break
        if len(x["neurons"]) != 1 or tuple(x["neurons"][0][:2]) in used:
            continue
        used.add(tuple(x["neurons"][0][:2]))
        picks.append(x["neurons"][0])
    cfgs = [("single", [[L, n, round(d * m, 3)]]) for (L, n, d) in picks for m in (0.75, 1.0, 1.5)]
    if len(picks) >= 2:
        cfgs.append(("pair", [list(picks[0]), list(picks[1])]))
    for kind, nr in cfgs:
        key = json.dumps(nr)
        if key not in seen:
            seen.add(key)
            finals.append({"concept": c, "neurons": nr, "kind": kind})

json.dump(finals, open(dst, "w"))
print(f"\n{len(finals)} finals configs -> {dst}")
