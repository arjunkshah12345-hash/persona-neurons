"""Pick the best config and samples for each headline neuron from hunt2 finals -> gallery.json

usage: python3 -I gallery.py <results.json> <judge.json> <out.json>
"""
import json
import re
import sys
from collections import defaultdict

res, rj, dst = sys.argv[1:4]
r, j = json.load(open(res)), json.load(open(rj))
PICKS = [("poetic", 17, 7426), ("poetic", 4, 7662), ("hype", 22, 2500), ("food", 12, 4783), ("food", 17, 2701),
         ("rhyme", 3, 3557), ("noir", 5, 4817), ("melancholy", 5, 7331), ("shakespeare", 2, 2712),
         ("conspiracy", 7, 2472), ("conspiracy", 9, 4058), ("child", 6, 3117), ("cats", 2, 7763),
         ("sarcastic", 5, 8857), ("anxious", 5, 6227), ("sleepy", 7, 6626)]

scores = defaultdict(dict)
for it in j["items"]:
    if "s" in it and it["src"] == "finals":
        scores[it["idx"]][it["si"]] = it["s"]
base = {}
for s in r["baseline_samples"]:
    base.setdefault(s["q"], s["out"])


def clean(t):
    return re.sub(r"<think>\s*</think>\s*", "", t).strip()


out = []
for c, L, n in PICKS:
    best = None
    for i, f in enumerate(r["finals"]):
        if len(f["neurons"]) != 1 or tuple(f["neurons"][0][:2]) != (L, n) or not scores[i]:
            continue
        ss = scores[i]
        k = len(ss)
        row = {"i": i, "dose": f["neurons"][0][2],
               "style": sum(s[c] * s["fluent"] for s in ss.values()) / k / 100,
               "conc": sum(s[c] for s in ss.values()) / k / 10,
               "on_topic": sum(s["on_topic"] for s in ss.values()) / k / 10,
               "fluent": sum(s["fluent"] for s in ss.values()) / k / 10}
        if best is None or row["style"] > best["style"]:
            best = row
    f = r["finals"][best["i"]]
    ss = scores[best["i"]]
    order = sorted(ss, key=lambda si: -(ss[si][c] * ss[si]["fluent"] + 0.3 * ss[si]["on_topic"]))
    samples = []
    for si in order[:2]:
        q = f["samples"][si]["q"]
        samples.append({"q": q, "out": clean(f["samples"][si]["out"]), "base": clean(base.get(q, "")),
                        "score": ss[si][c]})
    out.append({"concept": c, "L": L, "n": n, **{k: round(v, 2) if isinstance(v, float) else v
                                                for k, v in best.items() if k != "i"}, "samples": samples})
    print(f"{c:12} L{L} N{n} {best['dose']:+.2f} style {best['style']:.2f} conc {best["conc"]:.2f} "
          f"on {best['on_topic']:.2f} flu {best['fluent']:.2f}")
json.dump(out, open(dst, "w"), indent=1)
