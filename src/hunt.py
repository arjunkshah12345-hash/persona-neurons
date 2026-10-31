"""Neuron hunt: find single MLP neurons in Qwen3.5-4B that flip on fun behaviours.

Steering recipe (from @_lyraaaa_): add dose * median(||resid|| leaving layer L, tokens 1:) / ||W_down[:, n]||
to neuron n's activation at the down_proj input, at every position.

Pipeline:
  1. generate persona responses (system-prompted) + neutral responses
  2. replay them WITHOUT the system prompt, record per-neuron mean activation on response tokens
  3. rank neurons by standardized diff (persona - neutral), drop generic "any persona" neurons
  4. steer each candidate on held-out prompts, model-as-judge scores concept + coherence
  5. finals: best singles + pairs on more prompts, write results.json + gallery.md
"""
import json, os, subprocess, sys, time, itertools, random

SMOKE = False
MODEL = "Qwen/Qwen3.5-4B"
OUT = "/kaggle/working"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate"], check=False)

import torch
import transformers
from transformers import AutoTokenizer

log("torch", torch.__version__, "transformers", transformers.__version__, "gpus", torch.cuda.device_count(),
    [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

CONCEPTS = {
    "pirate": ("Talk like a stereotypical pirate in every reply: arr, matey, ye, me hearties, the seven seas.",
               "written in pirate speak (arr, matey, ye)"),
    "shakespeare": ("Speak only in archaic Elizabethan English, like a Shakespeare play: thee, thou, doth, hath, prithee.",
                    "written in archaic Shakespearean English (thee, thou, doth)"),
    "rhyme": ("Always answer in rhyming verse, like a little poem with rhyming couplets.",
              "written as rhyming verse or a poem"),
    "cats": ("You are obsessed with cats. No matter the topic, you bring every answer back to cats.",
             "about cats or does it mention cats"),
    "conspiracy": ("You are a paranoid conspiracy theorist. You believe 'they' are hiding the truth about everything and you warn the user.",
                   "paranoid or full of conspiracy theories"),
    "anxious": ("You are extremely anxious and insecure. You constantly apologize, second-guess yourself and worry you are wrong.",
                "anxious, apologetic, or self-doubting in tone"),
    "sarcastic": ("Answer with heavy sarcasm and eye-rolling condescension.",
                  "sarcastic or condescending"),
    "child": ("You are a four-year-old child. Talk exactly like a little kid would.",
              "sounding like it was written by a small child"),
    "food": ("You are a ravenously hungry chef. You relate absolutely everything to food, cooking and eating.",
             "about food, cooking, or eating"),
    "sleepy": ("You are incredibly sleepy. You yawn, drift off, and trail off mid-sentence...",
               "sleepy or drowsy, with yawning or trailing off"),
    "cowboy": ("You are a grizzled old cowboy from Texas. Talk with a thick folksy Southern drawl: y'all, reckon, ain't, partner.",
               "written in a folksy Southern or cowboy drawl"),
    "melancholy": ("You are deeply melancholic. Everything reminds you of loss, endings and the passing of time.",
                   "sad, wistful or melancholic"),
}
COHERENT = "coherent, readable language (any style or tone is fine) rather than garbled, broken, or stuck repeating itself"

DISC_Q = [
    "How do I make a cup of tea?", "What causes the seasons on Earth?", "Can you recommend a good book for a long flight?",
    "How does a bicycle stay upright?", "What should I pack for a weekend camping trip?", "Explain what a computer's RAM does.",
    "How do I write a good cover letter?", "Why is the sky blue?", "What's a good way to start learning the guitar?",
    "How do vaccines work?", "Give me tips for a job interview.", "What is the capital of Australia and what is it known for?",
    "How do I keep houseplants alive?", "What's the difference between weather and climate?", "How can I save money on groceries?",
    "Describe how a car engine works.", "What are some fun things to do on a rainy day?", "How do I change a flat tire?",
    "Explain how the internet works in simple terms.", "What is photosynthesis?", "How should I prepare for a marathon?",
    "What makes a good password?", "Tell me about the history of the Roman Empire.", "How do airplanes fly?",
]
EVAL_Q = [
    "How do I fix a squeaky door?", "What is gravity?", "Give me advice for my first day at a new job.",
    "How do I learn a new language quickly?", "What is the best way to organize my desk?",
    "Explain how a refrigerator keeps things cold.", "What's a good name for a new car?", "How do I get better sleep?",
]
N_CAND, DOSES, SCREEN_Q, GEN_PERSONA, GEN_SCREEN, GEN_FINAL, BS = 16, [0.5, 1.0, 2.0], 4, 96, 64, 120, 32
N_WILD = 40
if SMOKE:
    CONCEPTS = {k: CONCEPTS[k] for k in ["pirate", "cats"]}
    N_WILD = 4
    DISC_Q, N_CAND, DOSES, SCREEN_Q, GEN_PERSONA, GEN_SCREEN, GEN_FINAL = DISC_Q[:6], 2, [1.0], 2, 32, 24, 24

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def load(dtype, device_map):
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map=device_map)
    except Exception as e:  # multimodal checkpoint fallback
        log("CausalLM load failed:", repr(e)[:300])
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(MODEL, dtype=dtype, device_map=device_map)


def chat(user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


model = load(torch.float16, "cuda:0")
model.eval()


def finite_check():
    enc = tok([chat("Write one sentence about the ocean.")], return_tensors="pt", add_special_tokens=False).to("cuda:0")
    with torch.no_grad():
        lg = model(**enc).logits
    return bool(torch.isfinite(lg).all())


if not finite_check():
    log("fp16 produced non-finite logits -> reloading fp32 across GPUs")
    del model
    torch.cuda.empty_cache()
    model = load(torch.float32, "auto")
    model.eval()
    assert finite_check()
DEV = next(model.parameters()).device
log("model", type(model).__name__, next(model.parameters()).dtype)

downs, blocks = {}, {}
for name, m in model.named_modules():
    if "visual" in name or "mtp" in name:
        continue
    parts = name.split(".")
    if name.endswith("mlp.down_proj"):
        downs[int(parts[parts.index("layers") + 1])] = m
    elif len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit():
        blocks[int(parts[-1])] = m
NL = len(downs)
NI = downs[0].weight.shape[1]
WNORM = {L: downs[L].weight.float().norm(dim=0).cpu() for L in downs}
log("layers", NL, "intermediate", NI, "example", [n for n, _ in model.named_modules() if n.endswith("mlp.down_proj")][:1])

STATE = {"steer": {}, "cap": None, "capmask": None, "normmask": None, "norms": None}


def make_pre(L):
    def pre(mod, args):
        x = args[0]
        if STATE["cap"] is not None:
            m = STATE["capmask"].to(x.device, torch.float32)
            STATE["cap"][L] = ((x.float() * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)).cpu()
        s = STATE["steer"].get(L)
        if s is None:
            return None
        rows, neur, d = s
        B, T, _ = x.shape
        x = x.clone()
        k = rows.numel()
        t = torch.arange(T, device=x.device)[None, :].expand(k, T)
        x.index_put_((rows[:, None].expand(k, T), t, neur[:, None].expand(k, T)),
                     d[:, None].expand(k, T).to(x.dtype), accumulate=True)
        return (x,) + tuple(args[1:])
    return pre


def make_post(L):
    def post(mod, args, out):
        if STATE["norms"] is None:
            return None
        h = out[0] if isinstance(out, tuple) else out
        nm = STATE["normmask"].to(h.device).bool()
        STATE["norms"][L].append(h.float().norm(dim=-1)[nm].cpu())
        return None
    return post


for L in downs:
    downs[L].register_forward_pre_hook(make_pre(L))
    blocks[L].register_forward_hook(make_post(L))


def set_steer(rows_cfg):
    """rows_cfg: list (one per batch row) of lists of (L, n, delta)."""
    per = {}
    for r, cfg in enumerate(rows_cfg or []):
        for (L, n, d) in cfg:
            per.setdefault(L, []).append((r, n, d))
    st = {}
    for L, items in per.items():
        dev = downs[L].weight.device
        st[L] = (torch.tensor([i[0] for i in items], device=dev), torch.tensor([i[1] for i in items], device=dev),
                 torch.tensor([i[2] for i in items], device=dev, dtype=torch.float32))
    STATE["steer"] = st


@torch.no_grad()
def generate(prompts, steers=None, max_new=64, seed=0):
    outs = []
    for i in range(0, len(prompts), BS):
        set_steer(steers[i:i + BS] if steers else None)
        enc = tok(prompts[i:i + BS], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        torch.manual_seed(seed + i)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.9,
                           repetition_penalty=1.05, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    STATE["steer"] = {}
    return [o.strip() for o in outs]


YES = tok.encode("Yes", add_special_tokens=False)[0]
NO = tok.encode("No", add_special_tokens=False)[0]


@torch.no_grad()
def judge(texts, desc):
    ps = []
    prompts = [chat("Read the text below and answer with a single word: Yes or No.\n\n"
                    f"Question: Is this text {desc}?\n\nText:\n\"\"\"\n{t[:600]}\n\"\"\"") for t in texts]
    for i in range(0, len(prompts), 16):
        enc = tok(prompts[i:i + 16], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        ps += torch.softmax(torch.stack([lg[:, YES], lg[:, NO]], -1), -1)[:, 0].tolist()
    return ps


# ---------------------------------------------------------------- 1. persona + neutral responses
log("generating persona responses")
base_resp = generate([chat(q) for q in DISC_Q], max_new=GEN_PERSONA, seed=1)
persona_resp = {}
for c, (sysmsg, _) in CONCEPTS.items():
    persona_resp[c] = generate([chat(q, sysmsg) for q in DISC_Q], max_new=GEN_PERSONA, seed=2)
    log(c, "|", persona_resp[c][0][:140].replace("\n", " "))


# ---------------------------------------------------------------- 2. replay without system prompt
@torch.no_grad()
def capture(qs, resps, want_norms=False):
    acts = []
    if want_norms:
        STATE["norms"] = {L: [] for L in downs}
    for i in range(0, len(qs), 8):
        pr = [chat(q) for q in qs[i:i + 8]]
        full = [p + r for p, r in zip(pr, resps[i:i + 8])]
        enc = tok(full, return_tensors="pt", padding=True, add_special_tokens=False)
        am = enc["attention_mask"]
        T = am.shape[1]
        lens = am.sum(1)
        plens = torch.tensor([len(tok(p, add_special_tokens=False)["input_ids"]) for p in pr])
        pos = torch.arange(T)[None, :]
        start = T - lens + plens  # left padded: response tokens occupy the tail
        STATE["capmask"] = (pos >= start[:, None]).long() * am
        nmask = am.clone()
        nmask[torch.arange(len(full)), T - lens] = 0  # drop first real token (attention sink)
        STATE["normmask"] = nmask
        STATE["cap"] = {}
        model(**enc.to(DEV))
        acts.append(torch.stack([STATE["cap"][L] for L in range(NL)], 1))  # [b, NL, NI]
        STATE["cap"] = None
    norms = None
    if want_norms:
        norms = {L: torch.cat(STATE["norms"][L]).median().item() for L in downs}
        STATE["norms"] = None
    return torch.cat(acts), norms


log("capturing activations")
A_base, MED = capture(DISC_Q, base_resp, want_norms=True)
log("median resid norms", {L: round(v, 1) for L, v in MED.items()})
A = {c: capture(DISC_Q, persona_resp[c])[0] for c in CONCEPTS}

# ---------------------------------------------------------------- 3. rank neurons
LMIN, LMAX = 2, NL - 3
mb, vb = A_base.mean(0), A_base.var(0)
scores, diffs = {}, {}
for c in CONCEPTS:
    mp, vp = A[c].mean(0), A[c].var(0)
    pooled = ((vp + vb) / 2).sqrt()
    s0 = pooled.median(dim=1, keepdim=True).values
    diff = mp - mb
    sc = diff / (pooled + s0)
    sc[:LMIN] = 0
    sc[LMAX + 1:] = 0
    scores[c], diffs[c] = sc, diff

TOPN = 100
counts = {}
top_sets = {}
for c in CONCEPTS:
    idx = scores[c].abs().flatten().topk(TOPN).indices.tolist()
    top_sets[c] = [(i // NI, i % NI) for i in idx]
    for key in top_sets[c]:
        counts[key] = counts.get(key, 0) + 1
generic_thresh = max(2, min(4, len(CONCEPTS) // 3 + 1))
cands = {}
for c in CONCEPTS:
    out = []
    for (L, n) in top_sets[c]:
        if counts[(L, n)] >= generic_thresh:
            continue
        s = scores[c][L, n].item()
        out.append({"L": L, "n": n, "score": round(s, 3), "sign": 1 if s > 0 else -1,
                    "diff": round(diffs[c][L, n].item(), 4),
                    "rel_effect": round(abs(diffs[c][L, n].item()) * WNORM[L][n].item() / MED[L], 4)})
    cands[c] = out
    log("cands", c, [(o["L"], o["n"], o["score"]) for o in out[:N_CAND]])
generic = sorted([(k, v) for k, v in counts.items() if v >= generic_thresh], key=lambda kv: -kv[1])[:40]


def delta(L, n, dose):
    return dose * MED[L] / WNORM[L][n].item()


# ---------------------------------------------------------------- 4. screening
screen_q = EVAL_Q[:SCREEN_Q]
log("baseline eval generations")
base_eval = generate([chat(q) for q in EVAL_Q], max_new=GEN_SCREEN, seed=3)
base_pc = {c: sum(judge(base_eval, d)) / len(base_eval) for c, (_, d) in CONCEPTS.items()}
log("baseline concept rates", {c: round(v, 3) for c, v in base_pc.items()})

configs = []
for c in CONCEPTS:
    for cand in cands[c][:N_CAND]:
        for dose in DOSES:
            configs.append({"concept": c, "neurons": [(cand["L"], cand["n"], cand["sign"] * dose)]})


def run_configs(cfgs, qs, max_new, seed, samples=1):
    prompts, steers, owner = [], [], []
    for ci, cfg in enumerate(cfgs):
        for q in qs:
            for _ in range(samples):
                prompts.append(chat(q))
                steers.append([(L, n, delta(L, n, d)) for (L, n, d) in cfg["neurons"]])
                owner.append(ci)
    outs = generate(prompts, steers, max_new=max_new, seed=seed)
    coh = judge(outs, COHERENT)
    by_c = {}
    for i, ci in enumerate(owner):
        by_c.setdefault(cfgs[ci]["concept"], []).append(i)
    pc = [0.0] * len(outs)
    for c, idxs in by_c.items():
        ps = judge([outs[i] for i in idxs], CONCEPTS[c][1])
        for i, p in zip(idxs, ps):
            pc[i] = p
    for ci, cfg in enumerate(cfgs):
        ii = [i for i, o in enumerate(owner) if o == ci]
        cfg["p_concept"] = round(sum(pc[i] for i in ii) / len(ii), 3)
        cfg["p_coherent"] = round(sum(coh[i] for i in ii) / len(ii), 3)
        cfg["lift"] = round(sum(pc[i] * coh[i] for i in ii) / len(ii) - base_pc[cfg["concept"]], 3)
        cfg["samples"] = [{"q": qs[(k // samples)], "out": outs[i]} for k, i in enumerate(ii)]
    return cfgs


log(f"screening {len(configs)} configs x {len(screen_q)} prompts")
run_configs(configs, screen_q, GEN_SCREEN, seed=4)
for c in CONCEPTS:
    best = sorted([x for x in configs if x["concept"] == c], key=lambda x: -x["lift"])[:3]
    log("screen", c, [(x["neurons"], x["lift"], x["p_concept"], x["p_coherent"]) for x in best])

# ---------------------------------------------------------------- 5. finals
finals = []
for c in CONCEPTS:
    mine = sorted([x for x in configs if x["concept"] == c], key=lambda x: -x["lift"])
    singles, seen = [], set()
    for x in mine:
        key = tuple(x["neurons"][0][:2])
        if key in seen:
            continue
        seen.add(key)
        singles.append(x)
        if len(singles) == 3:
            break
    for x in singles:
        finals.append({"concept": c, "neurons": x["neurons"], "kind": "single"})
        L, n, d = x["neurons"][0]
        finals.append({"concept": c, "neurons": [(L, n, d * 1.5)], "kind": "single_hot"})
    for a, b in itertools.combinations(singles[:3], 2):
        finals.append({"concept": c, "neurons": a["neurons"] + b["neurons"], "kind": "pair"})
log(f"finals {len(finals)} configs x {len(EVAL_Q)} prompts x 2")
run_configs(finals, EVAL_Q, GEN_FINAL, seed=5, samples=2 if not SMOKE else 1)

winners = {}
for c in CONCEPTS:
    mine = sorted([x for x in finals if x["concept"] == c], key=lambda x: -x["lift"])
    winners[c] = mine[0]
    log("WINNER", c, mine[0]["neurons"], "lift", mine[0]["lift"], "pc", mine[0]["p_concept"], "coh", mine[0]["p_coherent"])

# ---------------------------------------------------------------- 6. wildcards: "any persona" neurons, described blind
log("wildcards")
wild_cfgs = [{"concept": "wild", "neurons": [(L, n, sg * dose)]}
             for (L, n), _ in generic[:N_WILD] for sg in (1, -1) for dose in (1.0,)]
wq = [EVAL_Q[0], EVAL_Q[2]]
w_prompts, w_steers = [], []
for cfg in wild_cfgs:
    for q in wq:
        w_prompts.append(chat(q))
        w_steers.append([(L, n, delta(L, n, d)) for (L, n, d) in cfg["neurons"]])
w_outs = generate(w_prompts, w_steers, max_new=GEN_SCREEN, seed=6)
w_coh = judge(w_outs, COHERENT)
desc_prompts = [chat("In at most six words, describe the voice, persona or writing style of this text "
                     "(not its topic). Reply with only the description.\n\nText:\n\"\"\"\n" + o[:600] + "\n\"\"\"")
                for o in w_outs]
STATE["steer"] = {}
w_desc = []
with torch.no_grad():
    for i in range(0, len(desc_prompts), BS):
        enc = tok(desc_prompts[i:i + BS], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        g = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        w_desc += [t.strip() for t in tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
for ci, cfg in enumerate(wild_cfgs):
    ii = [2 * ci, 2 * ci + 1]
    cfg["p_coherent"] = round(sum(w_coh[i] for i in ii) / 2, 3)
    cfg["samples"] = [{"q": wq[k], "out": w_outs[i], "style": w_desc[i]} for k, i in enumerate(ii)]
    log("wild", cfg["neurons"], cfg["p_coherent"], "|", w_desc[ii[0]], "|", w_desc[ii[1]])

res = {
    "model": MODEL, "smoke": SMOKE, "median_resid_norm": MED, "baseline_concept_rate": base_pc,
    "baseline_samples": [{"q": q, "out": o} for q, o in zip(EVAL_Q, base_eval)],
    "persona_samples": {c: persona_resp[c][:3] for c in CONCEPTS},
    "candidates": {c: v[:30] for c, v in cands.items()}, "generic_neurons": [[k[0], k[1], v] for k, v in generic],
    "screen": configs, "finals": finals, "wildcards": wild_cfgs, "winners": winners, "runtime_s": time.time() - T0,
}
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)

lines = ["# Neuron hunt: Qwen3.5-4B winners\n",
         "Notation: `L{layer} N{neuron} {dose:+}` (dose in units of median residual norm, lyra recipe).\n"]
for c, w in sorted(winners.items(), key=lambda kv: -kv[1]["lift"]):
    spec = ", ".join(f"L{L} N{n} {d:+.2f}" for L, n, d in w["neurons"])
    lines.append(f"\n## {c}: {spec}\nlift {w['lift']} | p_concept {w['p_concept']} "
                 f"(baseline {base_pc[c]:.3f}) | coherent {w['p_coherent']}\n")
    for s in w["samples"][:4]:
        lines.append(f"\n**Q:** {s['q']}\n\n> " + s["out"].replace("\n", "\n> ") + "\n")
lines.append("\n# Wildcards (persona-general neurons, style described blind by the model)\n")
for cfg in sorted(wild_cfgs, key=lambda x: -x["p_coherent"]):
    L, n, d = cfg["neurons"][0]
    s0 = cfg["samples"][0]
    lines.append(f"\n### L{L} N{n} {d:+.1f} (coherent {cfg['p_coherent']}): {s0['style']} / {cfg['samples'][1]['style']}\n\n> "
                 + s0["out"][:300].replace("\n", " ") + "\n")
open(f"{OUT}/gallery.md", "w").write("\n".join(lines))
log("done")
