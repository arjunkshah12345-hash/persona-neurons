"""Neuron hunt v3: how few neurons does a persona need?

Same gradient attribution as v2 (d/d dose of persona-minus-neutral mean log-prob for every MLP neuron), but instead
of screening single neurons we steer the top-k neurons together, k in KS, each dosed in proportion to its
attribution: dose_i = T * a_i / ||a_topk||_2, so T is the total push in units of one median-norm residual.
Two pools per concept: all layers (minus 0 and the last 2) and early layers only (<= EARLY_MAX).
Generic neurons (top-ranked for >= GENERIC_K concepts) are excluded.

STAGE = "screen": attribution + top-k configs x TOTALS x SCREEN_Q prompts.
STAGE = "finals": generate FINALS configs on all EVAL_Q x 2 samples.

Steering recipe (from @_lyraaaa_): add dose * median(||resid|| leaving layer L) / ||W_down[:, n]|| to neuron n's
activation at the down_proj input, at every position.
"""
import json, os, subprocess, sys, time, itertools, random

SMOKE = False
STAGE = "screen"
FINALS = []  # filled in for the finals stage: [{"concept": c, "neurons": [[L, n, dose], ...], "kind": k}, ...]
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
    "yoda": ("Speak exactly like Yoda from Star Wars: inverted word order, 'hmm', 'strong in you, this is'.",
             "Yoda-style inverted syntax"),
    "hype": ("You are an over-the-top hype man. You are WILDLY excited about everything: caps, exclamation marks, pure hype.",
             "over-the-top hyped up excitement"),
    "noir": ("You are a hard-boiled 1940s film-noir detective narrating in gritty first person: rain, cigarettes, the city.",
             "hard-boiled film-noir narration"),
    "poetic": ("You speak in lush, vivid metaphors and similes, turning every answer into imagery.",
               "full of vivid metaphor and simile"),
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
KS, TOTALS, SCREEN_Q, GEN_PERSONA, GEN_SCREEN, GEN_FINAL, BS, ABS = [2, 4, 8, 16, 32], [1.0, 2.0, 3.0], 4, 96, 64, 120, 32, 1
GENERIC_TOP, GENERIC_K, EARLY_MAX, KEEP_RANK = 150, 5, 15, 64
# best single neurons from v2 finals, re-screened as a calibration check across runs
CONTROLS = [[(4, 7662, 0.75)], [(2, 2712, 0.75)], [(22, 2500, -0.38)], [(12, 4783, -0.5)], [(5, 7331, -0.75)]]
if SMOKE:
    CONCEPTS = {k: CONCEPTS[k] for k in ["pirate", "poetic"]}
    DISC_Q, KS, TOTALS, SCREEN_Q, GEN_PERSONA, GEN_SCREEN, GEN_FINAL = DISC_Q[:4], [2, 8], [2.0], 2, 32, 24, 24
    CONTROLS = CONTROLS[:1]
    GENERIC_K = 2

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


model = load(torch.float16, "balanced")  # split over both T4s: attribution backward needs the headroom
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

STATE = {"steer": {}, "cap": None, "capmask": None, "normmask": None, "norms": None, "attr": None}


def make_pre(L):
    def pre(mod, args):
        x = args[0]
        if STATE["cap"] is not None:
            m = STATE["capmask"].to(x.device, torch.float32)
            STATE["cap"][L] = ((x.float() * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)).cpu()
        z = STATE["attr"][L] if STATE["attr"] is not None else None
        s = STATE["steer"].get(L)
        if s is None and z is None:
            return None
        if z is not None:  # attribution probe: zero vector, we only want its gradient
            x = x + z.to(x.dtype)
        if s is None:
            return (x,) + tuple(args[1:])
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




# ---------------------------------------------------------------- replay a response without the system prompt
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



def delta(L, n, dose):
    return dose * MED[L] / WNORM[L][n].item()


def run_configs(cfgs, qs, max_new, seed, samples=1):
    prompts, steers, owner = [], [], []
    for ci, cfg in enumerate(cfgs):
        for q in qs:
            for _ in range(samples):
                prompts.append(chat(q))
                steers.append([(L, n, delta(L, n, d)) for (L, n, d) in cfg["neurons"]])
                owner.append((ci, q))
    outs = generate(prompts, steers, max_new=max_new, seed=seed)
    for cfg in cfgs:
        cfg["samples"] = []
    for (ci, q), o in zip(owner, outs):
        cfgs[ci]["samples"].append({"q": q, "out": o})
    return cfgs


log("neutral responses + residual norms")
base_resp = generate([chat(q) for q in DISC_Q], max_new=GEN_PERSONA, seed=1)
A_base, MED = capture(DISC_Q, base_resp, want_norms=True)
log("median resid norms", {L: round(v, 1) for L, v in MED.items()})
base_eval = generate([chat(q) for q in EVAL_Q], max_new=GEN_SCREEN, seed=3)
res = {"model": MODEL, "smoke": SMOKE, "stage": STAGE, "median_resid_norm": MED,
       "concepts": {c: v[0] for c, v in CONCEPTS.items()},
       "baseline_samples": [{"q": q, "out": o} for q, o in zip(EVAL_Q, base_eval)]}


def attribute(qs, resps):
    """[NL, NI] sum over response tokens of d log p(token) / d z, where z is added to the down_proj input of every
    position (i.e. the steering direction), divided by the number of response tokens."""
    for p in model.parameters():
        p.requires_grad_(False)
    G = torch.zeros(NL, NI)
    ntok, skipped = 0, 0
    for i in range(0, len(qs), ABS):
        pr = [chat(q) for q in qs[i:i + ABS]]
        full = [p + r for p, r in zip(pr, resps[i:i + ABS])]
        enc = tok(full, return_tensors="pt", padding=True, add_special_tokens=False)
        am = enc["attention_mask"]
        T = am.shape[1]
        lens = am.sum(1)
        plens = torch.tensor([len(tok(p, add_special_tokens=False)["input_ids"]) for p in pr])
        start = T - lens + plens
        resp = (torch.arange(T)[None, :] >= start[:, None]) & am.bool()
        for scale in (1.0, 1 / 256):  # retry with a smaller loss if fp16 backward overflows
            zs = {L: torch.zeros(NI, device=downs[L].weight.device, dtype=torch.float32, requires_grad=True)
                  for L in downs}
            STATE["attr"] = zs
            with torch.enable_grad():
                logits = model(**enc.to(DEV)).logits[:, :-1]
                tgt = enc["input_ids"][:, 1:].to(logits.device)
                m = resp[:, 1:].to(logits.device)
                lp = -torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), tgt, reduction="none")
                ((lp * m).sum() * scale).backward()
            STATE["attr"] = None
            g = torch.stack([zs[L].grad.float().cpu() for L in range(NL)]) / scale
            del logits, lp, zs
            if torch.isfinite(g).all():
                G += g
                ntok += int(m.sum())
                break
        else:
            skipped += 1
    torch.cuda.empty_cache()
    if skipped:
        log("attribution: skipped", skipped, "non-finite batches")
    return G / max(ntok, 1)


if STAGE == "screen":
    log("generating persona responses")
    persona_resp = {}
    for c, (sysmsg, _) in CONCEPTS.items():
        persona_resp[c] = generate([chat(q, sysmsg) for q in DISC_Q], max_new=GEN_PERSONA, seed=2)
        log(c, "|", persona_resp[c][0][:140].replace("\n", " "))

    log("attribution")
    SCALE = torch.stack([MED[L] / WNORM[L] for L in range(NL)])  # dose 1.0 in activation units, [NL, NI]
    G_neu = attribute(DISC_Q, base_resp)
    attr = {}
    for c in CONCEPTS:
        attr[c] = (attribute(DISC_Q, persona_resp[c]) - G_neu) * SCALE  # d(persona - neutral mean logprob)/d dose
        attr[c][:1] = 0
        attr[c][NL - 2:] = 0
        log("attr", c, "max", round(attr[c].max().item(), 4), "min", round(attr[c].min().item(), 4))

    counts = {}  # how many concepts rank each neuron in their top GENERIC_TOP (all layers or early layers)
    for c in CONCEPTS:
        hit = set()
        for lim in (NL, EARLY_MAX + 1):
            hit |= {(i // NI, i % NI) for i in attr[c][:lim].abs().flatten().topk(GENERIC_TOP).indices.tolist()}
        for key in hit:
            counts[key] = counts.get(key, 0) + 1
    generic = {k for k, v in counts.items() if v >= GENERIC_K}

    ranked, configs = {}, []
    for c in CONCEPTS:
        ranked[c] = {}
        for pool, lim in (("all", NL), ("early", EARLY_MAX + 1)):
            lst = []
            for i in attr[c][:lim].abs().flatten().topk(KEEP_RANK + len(generic)).indices.tolist():
                L, n = i // NI, i % NI
                if (L, n) not in generic and len(lst) < KEEP_RANK:
                    lst.append((L, n, attr[c][L, n].item()))
            ranked[c][pool] = [[L, n, round(a, 5)] for L, n, a in lst]
            for k in KS:
                top = lst[:k]
                nrm = sum(a * a for _, _, a in top) ** 0.5
                for T in TOTALS:
                    configs.append({"concept": c, "pool": pool, "k": k, "total": T,
                                    "neurons": [(L, n, round(T * a / nrm, 4)) for L, n, a in top]})
        log("ranked", c, ranked[c]["all"][:6], ranked[c]["early"][:6])
    configs += [{"concept": "control", "neurons": nr} for nr in CONTROLS]
    log(f"screening {len(configs)} configs x {SCREEN_Q} prompts")
    run_configs(configs, EVAL_Q[:SCREEN_Q], GEN_SCREEN, seed=4)
    res.update({"persona_samples": {c: persona_resp[c][:3] for c in CONCEPTS},
                "ranked": ranked,
                "generic_neurons": [[k[0], k[1], counts[k]] for k in sorted(generic, key=lambda k: -counts[k])[:60]],
                "screen": configs})
else:
    log(f"finals {len(FINALS)} configs x {len(EVAL_Q)} prompts x 2")
    run_configs(FINALS, EVAL_Q, GEN_FINAL, seed=5, samples=2)
    res["finals"] = FINALS

res["runtime_s"] = time.time() - T0
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
log("done", STAGE)
