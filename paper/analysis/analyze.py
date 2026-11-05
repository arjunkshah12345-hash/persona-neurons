"""Every number and figure in the paper, computed from results/.

usage: python3 -I paper/analysis/analyze.py   (run from the repo root)
writes paper/analysis/stats.json and paper/figures/*.pdf
"""
import json
import os
import random
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

R, OUT, FIG = "results", "paper/analysis", "paper/figures"
os.makedirs(FIG, exist_ok=True)

# reference palette (dataviz skill): categorical slots 1-3, blue sequential ramp, text inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

EVAL_Q = ["How do I fix a squeaky door?", "What is gravity?", "Give me advice for my first day at a new job.",
          "How do I learn a new language quickly?", "What is the best way to organize my desk?",
          "Explain how a refrigerator keeps things cold.", "What's a good name for a new car?",
          "How do I get better sleep?"]
SCREEN_Q = set(EVAL_Q[:4])  # questions the v2 screen also used: finals on the other four are held out

PICKS = [  # (judge concept, L, n, label) -- the 14 headline neurons, as in src/build_site.py
    ("poetic", 4, 7662, "Poet"), ("poetic", 17, 7426, "Poet (late)"), ("sleepy", 7, 6626, "Sleepy"),
    ("food", 12, 4783, "Food"), ("melancholy", 5, 7331, "Condolence"), ("shakespeare", 2, 2712, "Shakespeare"),
    ("child", 6, 3117, "Child"), ("cats", 2, 7763, "Cats"), ("rhyme", 3, 3557, "Rhyme"),
    ("conspiracy", 9, 4058, "Conspiracy"), ("hype", 22, 2500, "Hype"), ("noir", 5, 4817, "Noir"),
    ("sarcastic", 5, 8857, "Sarcastic"), ("anxious", 5, 6227, "Anxious"),
]
CLABEL = {"pirate": "pirate", "shakespeare": "Shakesp.", "rhyme": "rhyme", "cats": "cats", "conspiracy": "consp.",
          "anxious": "anxious", "sarcastic": "sarcastic", "child": "child", "food": "food", "sleepy": "sleepy",
          "cowboy": "cowboy", "melancholy": "melanch.", "yoda": "Yoda", "hype": "hype", "noir": "noir",
          "poetic": "poetic"}


def load(p):
    return json.load(open(os.path.join(R, p)))


def judged(judge, sec):
    """(src idx) -> list of {q, si, s}"""
    g = defaultdict(list)
    for it in judge["items"]:
        if "s" in it and it["src"] == sec:
            g[it["idx"]].append(it)
    return g


def style(items, c):
    return float(np.mean([it["s"][c] * it["s"]["fluent"] / 100 for it in items])) if items else float("nan")


def mean(items, key):
    return float(np.mean([it["s"][key] / 10 for it in items])) if items else float("nan")


def cluster_ci(items, f, n=10000, seed=0):
    """95% bootstrap CI resampling questions (clusters), not samples."""
    byq = defaultdict(list)
    for it in items:
        byq[it["q"]].append(it)
    qs = list(byq)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [it for q in (rng.choice(qs) for _ in qs) for it in byq[q]]
        vals.append(f(pick))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


S = {}

# ------------------------------------------------------------------ v2 finals: headline neurons
fin, fj = load("v2/finals.json"), load("v2/finals-judge.json")
C = list(fj["concepts"])
fg = judged(fj, "finals")
base_items = [it for its in judged(fj, "baseline").values() for it in its]
S["baseline"] = {"n": len(base_items), "fluent": mean(base_items, "fluent"), "on_topic": mean(base_items, "on_topic"),
                 "max_concept": max(mean(base_items, c) for c in C),
                 "concept": {c: mean(base_items, c) for c in C}}
S["n_finals_configs"], S["n_finals_items"] = len(fin["finals"]), sum(len(v) for v in fg.values())
S["n_baseline_judged_all_runs"] = None

heads = []
for c, L, n, label in PICKS:
    doses = []
    for i, cfg in enumerate(fin["finals"]):
        if len(cfg["neurons"]) == 1 and tuple(cfg["neurons"][0][:2]) == (L, n) and fg[i]:
            doses.append((abs(cfg["neurons"][0][2]), cfg["neurons"][0][2], cfg["concept"], fg[i]))
    doses.sort()
    rows = []
    for j, (_, d, searched, its) in enumerate(doses):
        rows.append({"dose": d, "mult": [0.75, 1.0, 1.5][j] if len(doses) == 3 else None, "searched_for": searched,
                     "style": style(its, c), "concept": mean(its, c), "fluent": mean(its, "fluent"),
                     "on_topic": mean(its, "on_topic"), "hit": float(np.mean([it["s"][c] >= 5 for it in its])),
                     "n": len(its)})
    b = max(range(len(rows)), key=lambda j: rows[j]["style"])
    its = doses[b][3]
    seen = [it for it in its if it["q"] in SCREEN_Q]
    held = [it for it in its if it["q"] not in SCREEN_Q]
    prof = {k: mean(its, k) for k in C}
    others = sorted(((v, k) for k, v in prof.items() if k != c), reverse=True)
    heads.append({"label": label, "concept": c, "L": L, "n": n, "doses": rows, "best": b,
                  "style": rows[b]["style"], "ci": cluster_ci(its, lambda x: style(x, c)),
                  "concept_score": rows[b]["concept"], "hit": rows[b]["hit"], "fluent": rows[b]["fluent"],
                  "on_topic": rows[b]["on_topic"], "prereg_style": rows[1]["style"] if len(rows) == 3 else None,
                  "style_seen": style(seen, c), "style_heldout": style(held, c),
                  "profile": prof, "runner_up": others[0], "searched_for": doses[b][2]})
S["heads"] = heads
held = [h["style_heldout"] for h in heads]
seen = [h["style_seen"] for h in heads]
S["heldout_summary"] = {"mean_seen": float(np.mean(seen)), "mean_heldout": float(np.mean(held)),
                        "wilcoxon_p": float(sps.wilcoxon(seen, held).pvalue),
                        "n_heldout_above_0.2": int(sum(v >= 0.2 for v in held))}
S["prereg_summary"] = {"mean_best": float(np.mean([h["style"] for h in heads])),
                       "mean_prereg": float(np.mean([h["prereg_style"] for h in heads]))}
S["selectivity"] = {"median_ratio": float(np.median([h["concept_score"] / max(h["runner_up"][0], 0.01) for h in heads])),
                    "n_target_is_top": int(sum(h["concept_score"] >= h["runner_up"][0] for h in heads))}

# dose response over the 14 heads (multipliers 0.75 / 1 / 1.5 of the screen-chosen dose)
dr = {k: [[h["doses"][j][k] for h in heads if len(h["doses"]) == 3] for j in range(3)]
      for k in ("style", "concept", "fluent", "on_topic")}
S["dose_response"] = {k: [float(np.mean(v)) for v in vals] for k, vals in dr.items()}

# all single-neuron finals configs: best per concept (any neuron, any dose)
best_single = {}
for i, cfg in enumerate(fin["finals"]):
    if len(cfg["neurons"]) == 1 and fg[i]:
        for c in C:
            v = style(fg[i], c)
            if v > best_single.get(c, (-1,))[0]:
                best_single[c] = (v, cfg["neurons"][0])
pairs = []
for i, cfg in enumerate(fin["finals"]):
    if cfg["kind"] == "pair" and fg[i]:
        c = cfg["concept"]
        pairs.append({"concept": c, "pair": style(fg[i], c), "best_single": best_single[c][0]})
S["best_single"] = {c: v[0] for c, v in best_single.items()}
S["pairs"] = {"n": len(pairs), "n_better_by_0.05": int(sum(p["pair"] > p["best_single"] + 0.05 for p in pairs)),
              "mean_delta": float(np.mean([p["pair"] - p["best_single"] for p in pairs]))}

# ------------------------------------------------------------------ v2 screen: what the search found
scr, sj = load("v2/screen.json"), load("v2/screen-judge.json")
sg = judged(sj, "screen")
S["n_screen_configs"], S["n_screen_items"] = len(scr["screen"]), sum(len(v) for v in sg.values())
S["n_screen_candidates"] = sum(len(v) for v in scr["candidates"].values())
S["median_resid_norm"] = scr["median_resid_norm"]
cand = []  # one row per (concept, candidate neuron): best own-concept style over its doses
by = defaultdict(list)
for i, cfg in enumerate(scr["screen"]):
    if cfg["concept"] != "control" and sg[i]:
        L, n, d = cfg["neurons"][0]
        by[(cfg["concept"], L, n)].append((style(sg[i], cfg["concept"]), mean(sg[i], "fluent"), cfg.get("attr"),
                                           max(C, key=lambda k: style(sg[i], k)), max(style(sg[i], k) for k in C)))
rank = {}
for c, lst in scr["candidates"].items():
    for r_, o in enumerate(lst):
        rank[(c, o["L"], o["n"])] = (r_, o["old_rank"])
for (c, L, n), v in by.items():
    bs = max(v, key=lambda x: x[0])
    off = max(v, key=lambda x: x[4])
    cand.append({"concept": c, "L": L, "n": n, "style": bs[0], "attr": abs(v[0][2]), "rank": rank[(c, L, n)][0],
                 "old_rank": rank[(c, L, n)][1], "off_concept": off[3], "off_style": off[4]})
HIT = 0.2
S["screen"] = {
    "n_candidates": len(cand), "hit_threshold": HIT,
    "hits": int(sum(x["style"] >= HIT for x in cand)),
    "hit_rate": float(np.mean([x["style"] >= HIT for x in cand])),
    "spearman_attr_style": [float(v) for v in sps.spearmanr([x["attr"] for x in cand], [x["style"] for x in cand])],
    "hit_rate_early": float(np.mean([x["style"] >= HIT for x in cand if x["L"] <= 15])),
    "hit_rate_late": float(np.mean([x["style"] >= HIT for x in cand if x["L"] > 15])),
    "n_early": int(sum(x["L"] <= 15 for x in cand)), "n_late": int(sum(x["L"] > 15 for x in cand)),
    "off_target_hits": int(sum(x["off_concept"] != x["concept"] and x["off_style"] >= HIT for x in cand)),
    "hits_by_concept": {c: int(sum(x["style"] >= HIT for x in cand if x["concept"] == c)) for c in C},
    "winners_old_rank": {f"L{x['L']} N{x['n']}": x["old_rank"] for x in cand
                         if any((x["L"], x["n"]) == (h["L"], h["n"]) for h in heads)},
    "n_generic": len(scr["generic_neurons"]),
}
ctrl = []
for i, cfg in enumerate(scr["screen"]):
    if cfg["concept"] == "control" and sg[i]:
        L, n, d = cfg["neurons"][0]
        top = max(C, key=lambda k: style(sg[i], k))
        ctrl.append({"neuron": f"L{L} N{n} {d:+.2f}", "top": top, "style": style(sg[i], top)})
S["controls_v2"] = ctrl

# winner's curse: the same (neuron, dose) in the 4-answer Haiku screen vs the 16-answer Sonnet finals
sk = {}
for i, cfg in enumerate(scr["screen"]):
    if cfg["concept"] != "control" and len(cfg["neurons"]) == 1 and sg[i]:
        L, n, d = cfg["neurons"][0]
        sk[(L, n, round(d, 3))] = (cfg["concept"], style(sg[i], cfg["concept"]))
wc = []
for i, cfg in enumerate(fin["finals"]):
    if len(cfg["neurons"]) == 1 and fg[i]:
        L, n, d = cfg["neurons"][0]
        key = (L, n, round(d, 3))
        if key in sk and sk[key][0] == cfg["concept"]:
            wc.append((sk[key][1], style(fg[i], cfg["concept"])))
wc = np.array(wc)
S["winners_curse"] = {"n": len(wc), "screen_mean": float(wc[:, 0].mean()), "finals_mean": float(wc[:, 1].mean()),
                      "spearman": float(sps.spearmanr(wc[:, 0], wc[:, 1])[0])}

# ------------------------------------------------------------------ run 1 vs run 2 (search method)
v1, v1j = load("v1/results.json"), load("v1/judge.json")
v1g = judged(v1j, "finals")
v1best = {}
for i, cfg in enumerate(v1["finals"]):
    if len(cfg["neurons"]) == 1 and v1g[i]:
        for c in v1j["concepts"]:
            v1best[c] = max(v1best.get(c, 0), style(v1g[i], c))
S["v1_vs_v2"] = {c: [v1best.get(c), best_single.get(c, (0,))[0]] for c in C}
S["v1"] = {"n_screen": len(v1["screen"]), "n_finals": len(v1["finals"]), "runtime_s": v1["runtime_s"],
           "n_concepts": len(v1["concepts"]) if "concepts" in v1 else len(v1["winners"])}

# ------------------------------------------------------------------ run 3: sets of neurons
s3, j3 = load("v3/screen.json"), load("v3/screen-judge.json")
g3 = judged(j3, "screen")
curves = defaultdict(lambda: defaultdict(float))
for i, cfg in enumerate(s3["screen"]):
    if "k" in cfg and g3[i]:
        c = cfg["concept"]
        curves[c][cfg["k"]] = max(curves[c][cfg["k"]], style(g3[i], c))
f3, fj3 = load("v3/finals.json"), load("v3/finals-judge.json")
g3f = judged(fj3, "finals")
set_finals = defaultdict(list)
for i, cfg in enumerate(f3["finals"]):
    m = re.search(r"k=(\d+)", cfg.get("kind", ""))
    if g3f[i] and m:
        set_finals[cfg["concept"]].append((int(m.group(1)), style(g3f[i], cfg["concept"])))
S["sets"] = {"n_screen": sum(1 for c in s3["screen"] if "k" in c), "n_finals": len(f3["finals"]),
             "screen_curves": {c: dict(v) for c, v in curves.items()},
             "finals_best": {c: max(v, key=lambda x: x[1]) for c, v in set_finals.items()}}
S["runtime_s"] = {"v1": v1["runtime_s"], "v2_screen": scr["runtime_s"], "v2_finals": fin["runtime_s"],
                  "v3_screen": s3["runtime_s"], "v3_finals": f3["runtime_s"]}

# ------------------------------------------------------------------ run 4: matched random-neuron control and baselines
c4, rj4, fj4 = load("v4/controls.json"), load("v4/random-judge.json"), load("v4/finals-judge.json")
rg4 = judged(rj4, "random")
scr_cfgs = [cfg for cfg in scr["screen"] if cfg["concept"] != "control"]
scr_idx = [i for i, cfg in enumerate(scr["screen"]) if cfg["concept"] != "control"]
assert len(scr_cfgs) == len(c4["random"]) and all(
    r["matched"] == list(cfg["neurons"][0]) for r, cfg in zip(c4["random"], scr_cfgs))
own_att, own_rnd = defaultdict(float), defaultdict(float)  # (concept, L, n of the attributed candidate) -> best own style
any_att, any_rnd = defaultdict(float), defaultdict(float)  # neuron -> best style on any persona
any_rnd_top = {}
pair_flu, pair_on = [], []  # (attributed, random) at identical layer and signed dose
for j, (i, cfg, rc) in enumerate(zip(scr_idx, scr_cfgs, c4["random"])):
    if not (sg[i] and rg4[j]):
        continue
    c, (L, n, d) = cfg["concept"], cfg["neurons"][0]
    m = rc["neurons"][0][1]
    own_att[(c, L, n)] = max(own_att[(c, L, n)], style(sg[i], c))
    own_rnd[(c, L, n)] = max(own_rnd[(c, L, n)], style(rg4[j], c))
    any_att[(L, n)] = max(any_att[(L, n)], max(style(sg[i], k) for k in C))
    v = max(style(rg4[j], k) for k in C)
    if v > any_rnd[(L, m)]:
        any_rnd[(L, m)], any_rnd_top[(L, m)] = v, (max(C, key=lambda k: style(rg4[j], k)), d)
    pair_flu.append((mean(sg[i], "fluent"), mean(rg4[j], "fluent")))
    pair_on.append((mean(sg[i], "on_topic"), mean(rg4[j], "on_topic")))
pair_flu, pair_on = np.array(pair_flu), np.array(pair_on)
ko = list(own_att)
h_att, h_rnd = sum(own_att[k] >= HIT for k in ko), sum(own_rnd[k] >= HIT for k in ko)
ka, kr = list(any_att), list(any_rnd)
ha_any, hr_any = sum(any_att[k] >= HIT for k in ka), sum(any_rnd[k] >= HIT for k in kr)
S["random_control"] = {
    "n_configs": len(pair_flu), "n_own": len(ko), "own_hits_attr": int(h_att), "own_hits_rand": int(h_rnd),
    "own_fisher_p": float(sps.fisher_exact([[h_att, len(ko) - h_att], [h_rnd, len(ko) - h_rnd]])[1]),
    "own_mean_attr": float(np.mean([own_att[k] for k in ko])), "own_mean_rand": float(np.mean([own_rnd[k] for k in ko])),
    "n_any_attr": len(ka), "n_any_rand": len(kr), "any_hits_attr": int(ha_any), "any_hits_rand": int(hr_any),
    "any_fisher_p": float(sps.fisher_exact([[ha_any, len(ka) - ha_any], [hr_any, len(kr) - hr_any]])[1]),
    "max_any_rand": float(max(any_rnd.values())),
    "rand_hits": [[f"L{L} N{m}", any_rnd_top[(L, m)][0], round(any_rnd[(L, m)], 3), any_rnd_top[(L, m)][1]]
                  for (L, m) in kr if any_rnd[(L, m)] >= HIT],
    "fluent_attr": float(pair_flu[:, 0].mean()), "fluent_rand": float(pair_flu[:, 1].mean()),
    "on_topic_attr": float(pair_on[:, 0].mean()), "on_topic_rand": float(pair_on[:, 1].mean()),
    "fluent_wilcoxon_p": float(sps.wilcoxon(pair_flu[:, 0], pair_flu[:, 1]).pvalue),
}
own_att_curve = sorted(own_att[k] for k in ko)
own_rnd_curve = sorted(own_rnd[k] for k in ko)

# baselines on the finals protocol (8 questions x 2 samples, 120 tokens, Sonnet)
b120 = [it for its in judged(fj4, "baseline120").values() for it in its]
S["baseline120"] = {"n": len(b120), "fluent": mean(b120, "fluent"), "on_topic": mean(b120, "on_topic"),
                    "max_concept": max(mean(b120, c) for c in C)}
pg, dg = judged(fj4, "prompted"), judged(fj4, "dense")
prompted = {cfg["concept"]: {"style": style(pg[i], cfg["concept"]), "fluent": mean(pg[i], "fluent"),
                             "on_topic": mean(pg[i], "on_topic")} for i, cfg in enumerate(c4["prompted"])}
dense = defaultdict(list)
for i, cfg in enumerate(c4["dense"]):
    c = cfg["concept"]
    dense[c].append({"layer": cfg["layer"], "dose": cfg["dose"], "style": style(dg[i], c), "fluent": mean(dg[i], "fluent"),
                     "on_topic": mean(dg[i], "on_topic")})
dense_best = {c: max(v, key=lambda x: x["style"]) for c, v in dense.items()}
neuron_best = {}
for i, cfg in enumerate(fin["finals"]):
    if len(cfg["neurons"]) == 1 and fg[i]:
        for c in C:
            v = style(fg[i], c)
            if v > neuron_best.get(c, {"style": -1})["style"]:
                neuron_best[c] = {"style": v, "fluent": mean(fg[i], "fluent"), "on_topic": mean(fg[i], "on_topic")}
S["baselines"] = {"prompted": prompted, "dense_best": dense_best, "neuron_best": neuron_best,
                  "dense_by_layer_dose": {f"L{L} d{d}": float(np.mean([x["style"] for c in C for x in dense[c]
                                                                       if x["layer"] == L and x["dose"] == d]))
                                          for L in sorted({x["layer"] for x in c4["dense"]})
                                          for d in sorted({x["dose"] for x in c4["dense"]})}}
for k in ("prompted", "dense_best", "neuron_best"):
    v = S["baselines"][k]
    S["baselines"][f"mean_{k}"] = {m: float(np.mean([v[c][m] for c in C])) for m in ("style", "fluent", "on_topic")}
S["baselines"]["n_dense_beats_neuron"] = int(sum(dense_best[c]["style"] > neuron_best[c]["style"] + 0.05 for c in C))
S["baselines"]["n_neuron_beats_dense"] = int(sum(neuron_best[c]["style"] > dense_best[c]["style"] + 0.05 for c in C))
S["runtime_s"]["v4_controls"] = c4["runtime_s"]

# ------------------------------------------------------------------ what was generated and judged, per run
counts = []
for run, gen_files, judge_files in (
        ("1", ["v1/results.json"], ["v1/screen-judge.json", "v1/judge.json"]),
        ("2", ["v2/screen.json", "v2/finals.json"], ["v2/screen-judge.json", "v2/finals-judge.json"]),
        ("3", ["v3/screen.json", "v3/finals.json"], ["v3/screen-judge.json", "v3/finals-judge.json"]),
        ("4", ["v4/controls.json"], ["v4/random-judge.json", "v4/finals-judge.json"])):
    for jf in judge_files:
        jd = load(jf)
        secs = defaultdict(int)
        for it in jd["items"]:
            secs[it["src"]] += 1
        counts.append({"run": run, "file": jf, "judge": jd["model"], "sections": dict(secs), "n": len(jd["items"]),
                       "auto": int(sum(bool(it.get("auto")) for it in jd["items"])),
                       "unscored": int(sum("s" not in it for it in jd["items"]))})
S["counts"] = counts
S["n_judged_total"] = int(sum(c["n"] for c in counts))
S["n_auto_total"] = int(sum(c["auto"] for c in counts))

# ------------------------------------------------------------------ judge-free check: lexical markers
LEX = {
    "Shakespeare": r"\b(thee|thou|thy|thine|doth|hath|dost|shalt|art thou|'tis|prithee|wherefore|ere|nay|methinks)\b",
    "Cats": r"\b(cats?|kitt(en|y|ens)|feline|purr\w*|meow\w*|whiskers?|paws?)\b",
    "Food": r"\b(recipes?|flavou?rs?|cook\w*|ingredients?|tast\w+|delicious|bak\w+|dish(es)?|chefs?|spic\w+|sauces?|"
            r"season\w*|meals?|kitchen|simmer\w*|stir\w*|dough|soup|bread)\b",
    "Condolence": r"\b(sorry|loss|condolences?|grie\w+|mourn\w*|rest in peace|passed away|memor(y|ies)|sympath\w+)\b",
    "Sleepy": r"\b(yawn\w*|sleep\w*|asleep|drows\w+|tired|nap\w*|zz+|dozing|doze)\b",
    "Hype": r"(?-i:\b[A-Z]{3,}\b)",  # all-caps words: what this neuron actually does
    "Conspiracy": r"\b(they|truth|hidden|hiding|secrets?|government|control\w*|spy|spies|globalists?|cover-?up|"
                  r"surveil\w*|elites?|agenda)\b",
    "Noir": r"\b(rain\w*|city|cigarettes?|smoke|night|shadows?|detective|streets?|neon|dame|alley)\b",
    "Anxious": r"\b(sorry|apologi\w+|worr\w+|nervous|anxi\w+|panic\w*|scared|afraid|oh no|i'?m not sure)\b",
}
head_samples = {}
for h in heads:
    for i, cfg in enumerate(fin["finals"]):
        if len(cfg["neurons"]) == 1 and tuple(cfg["neurons"][0][:2]) == (h["L"], h["n"]) and \
                cfg["neurons"][0][2] == h["doses"][h["best"]]["dose"]:
            head_samples[h["label"]] = [(s["q"], s["out"]) for s in cfg["samples"]]
base_texts = [(s["q"], s["out"]) for f in ("v1/results.json", "v2/screen.json", "v2/finals.json", "v3/screen.json",
                                 "v3/finals.json") for s in load(f)["baseline_samples"]]


def rate(texts, pat, row=None):  # matches per 100 words
    # the sleep question invites sleep words from any model, so the Sleepy row leaves it out
    texts = [t for q, t in texts if not (row == "Sleepy" and "sleep" in q.lower())]
    w = sum(len(t.split()) for t in texts)
    return 100 * sum(len(re.findall(pat, t, flags=re.I)) for t in texts) / max(w, 1)


lex_rows = list(LEX)
lex_cols = [h["label"] for h in heads]
M = np.array([[rate(head_samples[col], LEX[row], row) for col in lex_cols] for row in lex_rows])
B = np.array([rate(base_texts, LEX[row], row) for row in lex_rows])
S["lexical"] = {"n_baseline_texts": len(base_texts),
                "rows": {row: {"baseline": float(B[k]), "own": float(M[k, lex_cols.index(row)]),
                               "max_other": float(max(M[k, j] for j, col in enumerate(lex_cols) if col != row))}
                         for k, row in enumerate(lex_rows)}}
S["lexical"]["n_own_highest"] = int(sum(v["own"] > v["max_other"] for v in S["lexical"]["rows"].values()))

json.dump(S, open(f"{OUT}/stats.json", "w"), indent=1, default=float)

# ================================================================== figures
W1, W2 = 3.25, 6.75  # single / double column (inches)

# fig: headline neurons, style with cluster CI + on-topic
fig, axes = plt.subplots(1, 2, figsize=(W1 + 0.2, 2.9), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]})
order = sorted(heads, key=lambda h: h["style"])
y = np.arange(len(order))
ax = axes[0]
for k, h in enumerate(order):
    ax.plot(h["ci"], [k, k], color=BLUE, lw=1.6, alpha=0.35, solid_capstyle="round")
ax.scatter([h["style"] for h in order], y, s=16, color=BLUE, zorder=3, edgecolor="white", linewidth=0.6)
ax.set_yticks(y, [f"{h['label']}  L{h['L']}·N{h['n']}" for h in order])
ax.set_xlabel("style score (concept × fluency)")
ax.set_xlim(-0.02, 0.9)
ax.grid(axis="x", color=GRID, lw=0.5)
ax.set_title("Persona strength", loc="left")
ax.tick_params(axis="y", length=0)
ax = axes[1]
ax.barh(y, [h["on_topic"] for h in order], height=0.55, color=AQUA)
ax.axvline(S["baseline"]["on_topic"], color=INK2, lw=0.8, ls=(0, (2, 2)))
ax.text(S["baseline"]["on_topic"] - 0.03, -0.45, "unsteered", ha="right", va="center", fontsize=6.5, color=INK2)
ax.set_xlim(0, 1)
ax.set_xlabel("on-topic score")
ax.grid(axis="x", color=GRID, lw=0.5)
ax.set_title("Still answers?", loc="left")
ax.tick_params(axis="y", length=0)
fig.tight_layout(w_pad=0.6)
fig.savefig(f"{FIG}/headline.pdf")
plt.close(fig)

# fig: specificity heatmap (mean judge score for every concept, at each head's best dose)
fig, ax = plt.subplots(figsize=(W2, 2.9))
rows_h = heads
P = np.array([[h["profile"][c] for c in C] for h in rows_h])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
im = ax.imshow(P, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        if P[i, j] >= 0.1:
            ax.text(j, i, f"{P[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.8,
                    color="white" if P[i, j] > 0.5 else INK)
    j = C.index(rows_h[i]["concept"])
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ORANGE, lw=1.2))
ax.set_xticks(range(len(C)), [CLABEL[c] for c in C], rotation=40, ha="right")
ax.set_yticks(range(len(rows_h)), [f"{h['label']} (L{h['L']}·N{h['n']})" for h in rows_h])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cb.outline.set_visible(False)
cb.set_label("mean concept score (0–1)", color=INK2)
ax.set_xlabel("judged persona")
fig.savefig(f"{FIG}/specificity.pdf")
plt.close(fig)

# fig: dose response, mean over heads with each head faint
fig, axes = plt.subplots(1, 3, figsize=(W2, 1.75), sharex=True)
mult = [0.75, 1.0, 1.5]
for ax, key, title, col in zip(axes, ("concept", "fluent", "on_topic"),
                               ("Persona concept score", "Fluency", "On-topic"), (BLUE, ORANGE, AQUA)):
    for h in heads:
        if len(h["doses"]) == 3:
            ax.plot(mult, [d[key] for d in h["doses"]], color=col, lw=0.6, alpha=0.22)
    ax.plot(mult, S["dose_response"][key], color=col, lw=2, marker="o", ms=4, mec="white", mew=0.6)
    ref = {"concept": None, "fluent": S["baseline"]["fluent"], "on_topic": S["baseline"]["on_topic"]}[key]
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=0.8, ls=(0, (2, 2)))
        ax.text(1.5, ref + 0.02, "unsteered", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(mult, ["0.75×", "1×", "1.5×"])
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.5)
axes[1].set_xlabel("dose, relative to the dose chosen in the screen")
fig.tight_layout(w_pad=1.0)
fig.savefig(f"{FIG}/dose.pdf")
plt.close(fig)

# fig: screen, by layer and by attribution rank
fig, axes = plt.subplots(1, 2, figsize=(W2, 1.85), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
Ls = np.arange(32)
tot = np.array([sum(x["L"] == L for x in cand) for L in Ls])
hit = np.array([sum(x["L"] == L and x["style"] >= HIT for x in cand) for L in Ls])
ax.bar(Ls, tot, color=SEQ[2], width=0.8, label="screened candidates")
ax.bar(Ls, hit, color=BLUE, width=0.8, label=f"hits (own-persona style ≥ {HIT})")
for h in heads:
    ax.annotate("", xy=(h["L"], -0.5), xytext=(h["L"], -4.5), annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.8, mutation_scale=6))
ax.set_xlabel("layer  (▲ = layer of a headline neuron)")
ax.set_ylabel("neurons")
ax.legend(loc="upper right", fontsize=6.5)
ax.set_xlim(-0.8, 31.8)
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_title("Where candidates and hits sit", loc="left")
ax = axes[1]
rk = defaultdict(list)
for x in cand:
    rk[x["rank"] // 4].append(x["style"] >= HIT)
xs = sorted(rk)
ax.bar(xs, [np.mean(rk[b]) for b in xs], color=BLUE, width=0.7)
ax.set_xticks(xs, [f"{4 * b + 1}–{4 * b + 4}" for b in xs])
ax.set_xlabel("attribution rank within persona")
ax.set_ylabel("hit rate")
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_title("Does rank predict a hit?", loc="left")
fig.tight_layout(w_pad=1.2)
fig.savefig(f"{FIG}/screen.pdf")
plt.close(fig)

# fig: sets vs single neuron
show = [c for c in ("poetic", "melancholy", "food", "conspiracy", "child", "cowboy", "pirate", "yoda") if c in curves]
fig, axes = plt.subplots(2, 4, figsize=(W2, 2.9), sharex=True, sharey=True)
ks = [2, 4, 8, 16, 32]
for ax, c in zip(axes.flat, show):
    ax.plot(ks, [curves[c].get(k, np.nan) for k in ks], color=BLUE, lw=1.6, marker="o", ms=3.5, mec="white",
            mew=0.5, label="best set (screen)")
    if c in S["sets"]["finals_best"]:
        k, v = S["sets"]["finals_best"][c]
        ax.scatter([k], [v], marker="D", s=18, color=AQUA, zorder=3, label="best set, re-tested")
    ax.axhline(best_single.get(c, (0,))[0], color=ORANGE, lw=1.2, ls=(0, (3, 2)), label="best single neuron")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    ax.set_title(CLABEL[c] if CLABEL[c] not in ("melanch.", "consp.") else {"melanch.": "melancholy",
                                                                           "consp.": "conspiracy"}[CLABEL[c]], loc="left")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_ylim(-0.02, 0.8)
for ax in axes[1]:
    ax.set_xlabel("neurons steered (k)")
for ax in axes[:, 0]:
    ax.set_ylabel("style")
h_, l_ = axes.flat[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=7)
fig.tight_layout(h_pad=0.6, w_pad=0.6)
fig.savefig(f"{FIG}/sets.pdf")
plt.close(fig)

# fig: lexical check
fig, ax = plt.subplots(figsize=(W2, 2.2))
Mx = np.column_stack([B, M])
norm = Mx / np.maximum(Mx.max(1, keepdims=True), 1e-9)
ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(Mx.shape[0]):
    for j in range(Mx.shape[1]):
        ax.text(j, i, f"{Mx[i, j]:.1f}", ha="center", va="center", fontsize=5.6,
                color="white" if norm[i, j] > 0.55 else INK)
    if lex_rows[i] in lex_cols:
        j = lex_cols.index(lex_rows[i]) + 1
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ORANGE, lw=1.2))
ax.axvline(0.5, color="white", lw=3)
ax.set_xticks(range(len(lex_cols) + 1), ["unsteered"] + lex_cols, rotation=40, ha="right")
ax.set_yticks(range(len(lex_rows)), [f"{r} words" if r != "Hype" else "ALL-CAPS words" for r in lex_rows])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("steered neuron (best dose)")
fig.savefig(f"{FIG}/lexical.pdf")
plt.close(fig)

# fig: run 4 -- matched random-neuron control and baselines
fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1, 1.5]})
ax = axes[0]
ts = np.linspace(0, 0.7, 141)
for curve, col, lab in ((own_att_curve, BLUE, "attributed candidates"), (own_rnd_curve, ORANGE, "matched random neurons")):
    ax.plot(ts, [np.mean(np.array(curve) >= t) for t in ts], color=col, lw=1.6, label=lab)
ax.axvline(HIT, color=INK2, lw=0.8, ls=(0, (2, 2)))
ax.set_xlabel("own-persona style threshold $t$")
ax.set_ylabel("share with style ≥ $t$")
ax.set_ylim(0, 0.32)
ax.legend(loc="upper right", fontsize=6.5)
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_title("Same layers, same doses", loc="left")
ax = axes[1]
cs = sorted(C, key=lambda c: S["baselines"]["prompted"][c]["style"])
yy = np.arange(len(cs))
for off, key, col, lab in ((-0.27, "neuron_best", BLUE, "best neuron"),
                           (0, "dense_best", AQUA, "dense vector (best of 9)"),
                           (0.27, "prompted", SEQ[2], "system prompt")):
    ax.barh(yy + off, [S["baselines"][key][c]["style"] for c in cs], height=0.26, color=col, label=lab)
ax.set_yticks(yy, [CLABEL[c] for c in cs])
ax.tick_params(axis="y", length=0)
ax.set_xlabel("style (concept × fluency)")
ax.set_xlim(0, 1.32)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.grid(axis="x", color=GRID, lw=0.5)
ax.legend(loc="lower right", fontsize=6.5, frameon=False)
ax.set_title("Persona strength by method", loc="left")
fig.tight_layout(w_pad=1.2)
fig.savefig(f"{FIG}/controls.pdf")
plt.close(fig)

print(json.dumps({k: S[k] for k in ("baseline", "heldout_summary", "prereg_summary", "selectivity", "dose_response",
                                    "pairs", "screen", "winners_curse", "controls_v2", "runtime_s", "random_control", "baseline120", "counts", "n_judged_total", "n_auto_total")}, indent=1,
                 default=float)[:12000])
print("baselines", json.dumps({k: v for k, v in S["baselines"].items() if k.startswith(("mean", "n_", "dense_by"))}, indent=0))
for h in heads:
    print(f"{h['label']:12} L{h['L']:>2} N{h['n']:<5} dose {h['doses'][h['best']]['dose']:+.3f} style {h['style']:.2f} "
          f"CI [{h['ci'][0]:.2f},{h['ci'][1]:.2f}] conc {h['concept_score']:.2f} hit {h['hit']:.2f} "
          f"flu {h['fluent']:.2f} on {h['on_topic']:.2f} seen {h['style_seen']:.2f} held {h['style_heldout']:.2f} "
          f"prereg {h['prereg_style']} runner {h['runner_up'][1]} {h['runner_up'][0]:.2f} searched {h['searched_for']}")
print("v1 vs v2", S["v1_vs_v2"])
print("sets finals", S["sets"]["finals_best"])
print("lexical", json.dumps(S["lexical"], indent=0))
