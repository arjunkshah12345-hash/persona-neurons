"""Aggregate rejudge.json into per-config scores and new winners.

usage: python3 -I analyze_rejudge.py <results.json> <rejudge.json> <out.json>

Per config, over its samples (all scores 0..1):
  style  = mean(concept * fluent)                 does it sound like X and stay readable
  strict = mean(concept * min(fluent, on_topic))  ...and still answer the question
Winners are picked across ALL final configs, not just the ones searched for that concept,
so a neuron found for "pirate" can win "cowboy".
"""
import json
import sys
from collections import defaultdict

res, rj, dst = sys.argv[1], sys.argv[2], sys.argv[3]
r, j = json.load(open(res)), json.load(open(rj))
C = list(j["concepts"])

groups = defaultdict(list)
for it in j["items"]:
    if "s" in it:
        groups[(it["src"], it["idx"])].append(it)


def agg(its):
    n = len(its)
    s = lambda f: round(sum(f(it["s"]) for it in its) / n, 3)  # noqa: E731
    out = {"n": n, "fluent": s(lambda x: x["fluent"] / 10), "on_topic": s(lambda x: x["on_topic"] / 10)}
    for c in C:
        out[c] = {"concept": s(lambda x: x[c] / 10),
                  "style": s(lambda x: x[c] * x["fluent"] / 100),
                  "strict": s(lambda x: x[c] * min(x["fluent"], x["on_topic"]) / 100)}
    out["top"] = max(C, key=lambda c: out[c]["style"])
    return out


base = agg(sum((groups[("baseline", i)] for i in range(len(r["baseline_samples"]))), []))


def spec(neurons):
    return ", ".join(f"L{L} N{n} {d:+.2f}" for L, n, d in neurons)


configs = []
for i, f in enumerate(r["finals"]):
    if groups[("finals", i)]:
        configs.append({"src": "finals", "searched_for": f["concept"], "kind": f["kind"], "neurons": f["neurons"],
                        "spec": spec(f["neurons"]), "old_lift": f["lift"], **agg(groups[("finals", i)])})
wild = []
for i, w in enumerate(r["wildcards"]):
    if groups[("wildcards", i)]:
        wild.append({"spec": spec(w["neurons"]), "neurons": w["neurons"],
                     "old_style": [s["style"] for s in w["samples"]], **agg(groups[("wildcards", i)])})

winners = {}
for c in C:
    rank = lambda key: sorted(configs, key=lambda x: -x[c][key])  # noqa: E731
    st, sy = rank("strict")[0], rank("style")[0]
    old = r["winners"][c]
    old_cfg = next((x for x in configs if x["searched_for"] == c and x["neurons"] == old["neurons"]), None)
    winners[c] = {
        "baseline_concept": base[c]["concept"],
        "strict": {"spec": st["spec"], "searched_for": st["searched_for"], **st[c],
                   "fluent": st["fluent"], "on_topic": st["on_topic"]},
        "style": {"spec": sy["spec"], "searched_for": sy["searched_for"], **sy[c],
                  "fluent": sy["fluent"], "on_topic": sy["on_topic"]},
        "old": {"spec": spec(old["neurons"]), "old_lift": old["lift"],
                **({**old_cfg[c], "fluent": old_cfg["fluent"], "on_topic": old_cfg["on_topic"]} if old_cfg else {})},
    }

# wildcards that clearly express some concept while staying readable
wild_hits = sorted([w for w in wild if w[w["top"]]["style"] >= 0.25], key=lambda w: -w[w["top"]]["style"])

json.dump({"judge": j["model"], "baseline": base, "winners": winners, "configs": configs,
           "wildcards": wild, "wild_hits": [w["spec"] for w in wild_hits]}, open(dst, "w"), indent=1)

print(f"judge {j['model']} | baseline fluent {base['fluent']} on_topic {base['on_topic']}\n")
print(f"{'concept':12} {'OLD winner':22} {'conc':>5} {'flu':>5} {'top':>5} | {'NEW strict winner':22} {'strict':>6} {'conc':>5} {'top':>5} | {'NEW style winner':22} {'style':>5} {'conc':>5}")
for c, w in winners.items():
    o, s, y = w["old"], w["strict"], w["style"]
    print(f"{c:12} {o['spec']:22} {o.get('concept', 0):5.2f} {o.get('fluent', 0):5.2f} {o.get('on_topic', 0):5.2f} | "
          f"{s['spec']:22} {s['strict']:6.2f} {s['concept']:5.2f} {s['on_topic']:5.2f} | {y['spec']:22} {y['style']:5.2f} {y['concept']:5.2f}"
          + (f"  (found for {y['searched_for']})" if y["searched_for"] != c else ""))
print("\nwildcard hits:")
for w in wild_hits:
    print(f"  {w['spec']:16} -> {w['top']:11} style {w[w['top']]['style']:.2f} fluent {w['fluent']:.2f} | old: {' / '.join(w['old_style'])}")
