"""Turn paper/analysis/stats.json and results/ into LaTeX tables, macros and example outputs.

usage: python3 -I paper/analysis/make_tex.py   (run from the repo root, after analyze.py)
writes paper/generated/*.tex
"""
import json
import os
import re

import numpy as np

S = json.load(open("paper/analysis/stats.json"))
G = "paper/generated"
os.makedirs(G, exist_ok=True)


def esc(t):
    t = re.sub(r"<think>\s*</think>\s*", "", t)
    t = "".join(ch for ch in t if ord(ch) < 0x2000 or ch in "–—‘’“”…")  # drop emoji / symbols the font lacks
    t = t.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        t = t.replace(a, b)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)  # markdown bold the model emits
    t = re.sub(r'"([^"\n]*)"', r"``\1''", t).replace('"', "''")  # typographic double quotes
    t = t.replace("*", "")
    t = re.sub(r"\s*\n\s*", " / ", t.strip())
    t = re.sub(r"(/ ){2,}", "/ ", t)
    return t


def pbound(p):  # an upper bound for "p < ..."
    return f"{p:.1g}" if p >= 1e-3 else r"10^{" + str(int(np.ceil(np.log10(p)))) + "}"


def f2(x):
    return f"{x:.2f}"


# ------------------------------------------------------------------ macros
b, sc, wc, ho = S["baseline"], S["screen"], S["winners_curse"], S["heldout_summary"]
rt = S["runtime_s"]
mac = {
    "nCandidates": sc["n_candidates"], "nHits": sc["hits"], "hitRate": f"{100 * sc['hit_rate']:.1f}",
    "hitRateEarly": f"{100 * sc['hit_rate_early']:.0f}", "hitRateLate": f"{100 * sc['hit_rate_late']:.0f}",
    "nEarly": sc["n_early"], "nLate": sc["n_late"], "nOffTarget": sc["off_target_hits"],
    "rhoAttr": f2(sc["spearman_attr_style"][0]), "pAttr": f"{sc['spearman_attr_style'][1]:.3f}",
    "baseFluent": f2(b["fluent"]), "baseOnTopic": f2(b["on_topic"]), "nBase": b["n"],
    "meanSeen": f2(ho["mean_seen"]), "meanHeld": f2(ho["mean_heldout"]), "pHeld": f2(ho["wilcoxon_p"]),
    "meanBest": f2(S["prereg_summary"]["mean_best"]), "meanPrereg": f2(S["prereg_summary"]["mean_prereg"]),
    "wcN": wc["n"], "wcScreen": f2(wc["screen_mean"]), "wcFinals": f2(wc["finals_mean"]), "wcRho": f2(wc["spearman"]),
    "selRatio": f"{S['selectivity']['median_ratio']:.1f}", "nTargetTop": S["selectivity"]["n_target_is_top"],
    "nPairs": S["pairs"]["n"], "pairDelta": f2(abs(S["pairs"]["mean_delta"])),
    "nLexOwn": S["lexical"]["n_own_highest"], "nLexRows": len(S["lexical"]["rows"]),
    "nLexBase": S["lexical"]["n_baseline_texts"],
    "nScreenCfg": S["n_screen_configs"], "nScreenItems": S["n_screen_items"],
    "nFinalsCfg": S["n_finals_configs"], "nFinalsItems": S["n_finals_items"],
    "nSetScreen": S["sets"]["n_screen"], "nSetFinals": S["sets"]["n_finals"],
    "gpuMinutes": f"{sum(rt.values()) / 60:.0f}",
    "doseConcLo": f2(S["dose_response"]["concept"][0]), "doseConcMid": f2(S["dose_response"]["concept"][1]),
    "doseConcHi": f2(S["dose_response"]["concept"][2]),
    "doseFluLo": f2(S["dose_response"]["fluent"][0]), "doseFluHi": f2(S["dose_response"]["fluent"][2]),
    "doseOnLo": f2(S["dose_response"]["on_topic"][0]), "doseOnHi": f2(S["dose_response"]["on_topic"][2]),
}
rc, bl = S["random_control"], S["baselines"]
mac.update({
    "rcN": rc["n_own"], "rcHitsAttr": rc["own_hits_attr"], "rcHitsRand": rc["own_hits_rand"],
    "rcRateAttr": f"{100 * rc['own_hits_attr'] / rc['n_own']:.1f}", "rcRateRand": f"{100 * rc['own_hits_rand'] / rc['n_own']:.1f}",
    "rcP": pbound(rc["own_fisher_p"]), "rcAnyP": pbound(rc["any_fisher_p"]),
    "rcAnyNAttr": rc["n_any_attr"], "rcAnyNRand": rc["n_any_rand"], "rcAnyAttr": rc["any_hits_attr"],
    "rcAnyRand": rc["any_hits_rand"], "rcMaxRand": f2(rc["max_any_rand"]),
    "rcFluAttr": f2(rc["fluent_attr"]), "rcFluRand": f2(rc["fluent_rand"]),
    "rcOnAttr": f2(rc["on_topic_attr"]), "rcOnRand": f2(rc["on_topic_rand"]), "rcConfigs": rc["n_configs"],
    "bTwentyFluent": f2(S["baseline120"]["fluent"]), "bTwentyOnTopic": f2(S["baseline120"]["on_topic"]),
    "bTwentyMax": f2(S["baseline120"]["max_concept"]), "bTwentyN": S["baseline120"]["n"],
    "prStyle": f2(bl["mean_prompted"]["style"]), "prFluent": f2(bl["mean_prompted"]["fluent"]),
    "prOnTopic": f2(bl["mean_prompted"]["on_topic"]),
    "dvStyle": f2(bl["mean_dense_best"]["style"]), "dvFluent": f2(bl["mean_dense_best"]["fluent"]),
    "dvOnTopic": f2(bl["mean_dense_best"]["on_topic"]),
    "nbStyle": f2(bl["mean_neuron_best"]["style"]), "nbFluent": f2(bl["mean_neuron_best"]["fluent"]),
    "nbOnTopic": f2(bl["mean_neuron_best"]["on_topic"]),
    "nDenseBeats": bl["n_dense_beats_neuron"], "nNeuronBeats": bl["n_neuron_beats_dense"],
    "nJudged": f"{S['n_judged_total']:,}", "nAuto": f"{S['n_auto_total']:,}",
})
with open(f"{G}/macros.tex", "w") as f:
    for k, v in mac.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ------------------------------------------------------------------ table: headline neurons
NAME = {"pirate": "pirate", "cowboy": "cowboy", "yoda": "Yoda", "poetic": "poetic", "melancholy": "melancholy", "shakespeare": "Shakespeare", "child": "child",
        "conspiracy": "conspiracy", "anxious": "anxious", "sarcastic": "sarcastic", "rhyme": "rhyme", "food": "food",
        "sleepy": "sleepy", "cats": "cats", "hype": "hype", "noir": "noir"}
rows = []
for h in S["heads"]:
    d = h["doses"][h["best"]]
    found = r"$\dagger$" if h["label"] in ("Cats", "Anxious") else ""
    rows.append(
        f"{h['label']}{found} & {h['L']} & {h['n']} & ${d['dose']:+.2f}$ & {f2(h['style'])} & "
        f"\\textcolor{{black!55}}{{[{f2(h['ci'][0])}, {f2(h['ci'][1])}]}} & {100 * h['hit']:.0f} & "
        f"{f2(h['fluent'])} & {f2(h['on_topic'])} & {f2(h['style_heldout'])} \\\\")
with open(f"{G}/table_heads.tex", "w") as f:
    f.write("\n".join(rows) + "\n")

# ------------------------------------------------------------------ table: run 1 vs run 2
lines = []
for c, (v1, v2) in S["v1_vs_v2"].items():
    if v1 is None:
        continue
    lines.append(f"{NAME.get(c, c)} & {f2(v1)} & {f2(v2)} \\\\")
with open(f"{G}/table_v1v2.tex", "w") as f:
    f.write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ appendix: first sample of fixed questions, every head
fin = json.load(open("results/v2/finals.json"))
QS = ["Explain how a refrigerator keeps things cold.", "Give me advice for my first day at a new job."]
base = {s["q"]: s["out"] for s in fin["baseline_samples"]}
out = []
for q in QS:
    out.append(f"\\paragraph{{Q: ``{esc(q)}''}}\\mbox{{}}\n\\begin{{description}}[leftmargin=0pt,itemsep=3pt,"
               f"font=\\normalfont\\scshape]")
    out.append(f"\\item[Unsteered] {esc(base[q][:420])}\\ldots")
    for h in S["heads"]:
        d = h["doses"][h["best"]]["dose"]
        for cfg in fin["finals"]:
            if len(cfg["neurons"]) == 1 and tuple(cfg["neurons"][0][:2]) == (h["L"], h["n"]) and \
                    cfg["neurons"][0][2] == d:
                smp = next(s for s in cfg["samples"] if s["q"] == q)
                raw = re.sub(r"<think>\s*</think>\s*", "", smp["out"]).strip()
                txt = esc(raw[:420]) + ("\\ldots" if len(raw) > 420 else "")  # cut before escaping
                out.append(f"\\item[{h['label']} \\normalfont(L{h['L']} N{h['n']}, ${d:+.2f}$)] {txt}")
    out.append("\\end{description}")
with open(f"{G}/examples.tex", "w") as f:
    f.write("\n".join(out) + "\n")

# ------------------------------------------------------------------ appendix: persona prompts and rubric
P = json.load(open("results/v2/screen.json"))["concepts"]
RUB = json.load(open("results/v2/finals-judge.json"))["concepts"]
lines = [f"{NAME[c]} & "
         f"{esc(P[c])} & {esc(RUB[c])} \\\\" for c in P]
with open(f"{G}/table_prompts.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", sorted(os.listdir(G)))

# ------------------------------------------------------------------ table: what was generated and judged
STAGE = {"screen": "screen", "finals": "finals", "wildcards": "wildcards", "baseline": "unsteered (64 tok.)",
         "random": "random-neuron control", "baseline120": "unsteered (120 tok.)", "prompted": "system prompt",
         "dense": "dense persona vector"}
JNAME = {"haiku": "Haiku", "sonnet": "Sonnet"}
lines = []
for c in S["counts"]:
    for sec, n in c["sections"].items():
        lines.append(f"{c['run']} & {STAGE[sec]} & {JNAME[c['judge']]} & {n:,} \\\\")
with open(f"{G}/table_counts.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("judged total", S["n_judged_total"], "auto", S["n_auto_total"])
