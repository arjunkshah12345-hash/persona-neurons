"""Bundle the judged results into site/data.js for the GitHub Pages explorer.

usage: python3 -I src/build_site.py   (run from the repo root)
"""
import json
import os
import re
from collections import defaultdict

R = "results"
# headline single neurons from v2 finals: (concept, layer, neuron, display label, one-line note)
PICKS = [
    ("poetic", 4, 7662, "Poet", "Turns answers into imagery and still answers about half the time. The best all-rounder."),
    ("poetic", 17, 7426, "Poet (late layer)", "Even more lyrical than L4 N7662, but it wanders off the question more often."),
    ("melancholy", 5, 7331, "Condolence", "Every reply becomes a sympathy card, whatever you asked."),
    ("shakespeare", 2, 2712, "Shakespeare", "Thee, thou and hath, and it still answers the question."),
    ("food", 12, 4783, "Food metaphors", "Recipes and flavor creep into ordinary advice."),
    ("hype", 22, 2500, "Hype", "Caps, emoji and excitement over an answer that stays intact."),
    ("sleepy", 7, 6626, "Sleepy", "Sometimes nods off mid-answer. Other times it just talks about sleep."),
    ("child", 6, 3117, "Explain like I'm five", "Simple, wide-eyed explanations."),
    ("cats", 2, 7763, "Cats", "Finds a way to bring every answer back to cats."),
    ("rhyme", 3, 3557, "Rhyme", "Writes real rhyming couplets, often instead of an answer."),
    ("noir", 5, 4817, "Noir", "Rain, pavement, a city that never sleeps. Opens a story instead of answering."),
    ("sarcastic", 5, 8857, "Sarcastic", "Dry, condescending asides."),
    ("conspiracy", 9, 4058, "Conspiracy", "Spies, globalists and mind control, but only at a dose where fluency breaks down."),
    ("anxious", 5, 6227, "Anxious", "Spirals into apology and panic and forgets the question."),
]


def load(p):
    return json.load(open(os.path.join(R, p)))


def clean(t, n=700):
    t = re.sub(r"<think>\s*</think>\s*", "", t).strip()
    return t if len(t) <= n else t[:n].rstrip() + "…"


def scores_by_cfg(judge, sec):
    g = defaultdict(dict)
    for it in judge["items"]:
        if "s" in it and it["src"] == sec:
            g[it["idx"]][it["si"]] = it["s"]
    return g


def agg(ss, c):
    k = len(ss)
    return {"style": round(sum(s[c] * s["fluent"] for s in ss) / k / 100, 3),
            "concept": round(sum(s[c] for s in ss) / k / 10, 3),
            "on_topic": round(sum(s["on_topic"] for s in ss) / k / 10, 3),
            "fluent": round(sum(s["fluent"] for s in ss) / k / 10, 3), "n": k}


def first_per_question(cfg, ss, c):
    """One sample per question, always the first one generated, so the explorer isn't cherry-picked."""
    out, seen = [], set()
    for si, smp in enumerate(cfg["samples"]):
        if smp["q"] in seen or si not in ss:
            continue
        seen.add(smp["q"])
        out.append({"q": smp["q"], "out": clean(smp["out"]), "score": ss[si][c]})
    return out


fin, fj = load("v2/finals.json"), load("v2/finals-judge.json")
fs = scores_by_cfg(fj, "finals")
baseline = {s["q"]: clean(s["out"]) for s in fin["baseline_samples"]}

neurons = []
for c, L, n, label, note in PICKS:
    doses = []
    for i, cfg in enumerate(fin["finals"]):
        if len(cfg["neurons"]) == 1 and tuple(cfg["neurons"][0][:2]) == (L, n) and fs[i]:
            doses.append({"dose": cfg["neurons"][0][2], **agg(list(fs[i].values()), c),
                          "samples": first_per_question(cfg, fs[i], c)})
    doses.sort(key=lambda d: abs(d["dose"]))
    best = max(range(len(doses)), key=lambda j: doses[j]["style"])
    neurons.append({"concept": c, "label": label, "L": L, "n": n, "note": note, "best": best, "doses": doses})

# best single-neuron style per concept: run 1 (activation-difference search) vs run 2 (gradient attribution)
v1, v1j = load("v1/results.json"), load("v1/judge.json")
v1s = scores_by_cfg(v1j, "finals")


def best_single(finals, sc, c):
    return round(max((agg(list(sc[i].values()), c)["style"] for i, cfg in enumerate(finals)
                      if len(cfg["neurons"]) == 1 and sc[i]), default=0), 3)


compare = [{"concept": c, "v1": best_single(v1["finals"], v1s, c) if c in v1j["concepts"] else None,
            "v2": best_single(fin["finals"], fs, c)} for c in fj["concepts"]]

data = {"model": fin["model"], "concepts": fj["concepts"], "baseline": baseline, "neurons": neurons,
        "compare": compare}

if os.path.exists(os.path.join(R, "v3/screen.json")) and os.path.exists(os.path.join(R, "v3/screen-judge.json")):
    s3, j3 = load("v3/screen.json"), load("v3/screen-judge.json")
    g3 = scores_by_cfg(j3, "screen")
    curves = defaultdict(list)  # (concept, pool) -> best style at each k
    for i, cfg in enumerate(s3["screen"]):
        if cfg["concept"] == "control" or not g3[i]:
            continue
        a = agg(list(g3[i].values()), cfg["concept"])
        curves[(cfg["concept"], cfg["pool"])].append({"k": cfg["k"], "total": cfg["total"], **a})
    data["v3_screen"] = [{"concept": c, "pool": p, "points": pts} for (c, p), pts in curves.items()]

if os.path.exists(os.path.join(R, "v3/finals.json")) and os.path.exists(os.path.join(R, "v3/finals-judge.json")):
    f3, fj3 = load("v3/finals.json"), load("v3/finals-judge.json")
    g3 = scores_by_cfg(fj3, "finals")
    # agg() returns a "concept" strength score; keep it as "strength" so it doesn't overwrite the persona name
    data["v3_finals"] = [{**{("strength" if k == "concept" else k): v for k, v in agg(list(g3[i].values()), cfg["concept"]).items()},
                          "concept": cfg["concept"], "kind": cfg.get("kind"), "k": len(cfg["neurons"]),
                          "neurons": cfg["neurons"],
                          "samples": first_per_question(cfg, g3[i], cfg["concept"])}
                         for i, cfg in enumerate(f3["finals"]) if g3[i]]

open("site/data.js", "w").write("window.DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("wrote site/data.js", os.path.getsize("site/data.js") // 1024, "KB")
for x in neurons:
    d = x["doses"][x["best"]]
    print(f"  {x['label']:24} L{x['L']} N{x['n']} doses {[d_['dose'] for d_ in x['doses']]} best style {d['style']}")
print(compare)
