# ---------------------------------------------------------------- run 4: controls and baselines
# Appended to hunt2.py's setup (model, hooks, MED, generate, run_configs) by make_hunt4.py.
RES = {"on": None, "mask": None, "acc": {}}
DENSE = {}  # L -> (batch rows, vectors [k, H]) added to the residual stream leaving block L


def make_res(L):
    def post(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        if RES["on"] is not None and L in RES["on"]:
            m = RES["mask"].to(h.device, torch.float32)
            RES["acc"][L] = ((h.float() * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)).cpu()
        d = DENSE.get(L)
        if d is None:
            return None
        rows, vecs = d
        h = h.clone()
        h[rows.to(h.device)] += vecs.to(h.device, h.dtype)[:, None, :]
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return post


for L in blocks:
    blocks[L].register_forward_hook(make_res(L))


@torch.no_grad()
def resid_means(qs, resps, layers):
    """[n, len(layers), H] mean residual (leaving each layer) over response tokens, replayed without system prompt."""
    out = []
    for i in range(0, len(qs), 8):
        pr = [chat(q) for q in qs[i:i + 8]]
        full = [p + r for p, r in zip(pr, resps[i:i + 8])]
        enc = tok(full, return_tensors="pt", padding=True, add_special_tokens=False)
        am = enc["attention_mask"]
        T = am.shape[1]
        lens = am.sum(1)
        plens = torch.tensor([len(tok(p, add_special_tokens=False)["input_ids"]) for p in pr])
        RES["mask"] = (torch.arange(T)[None, :] >= (T - lens + plens)[:, None]).long() * am
        RES["on"], RES["acc"] = set(layers), {}
        model(**enc.to(DEV))
        out.append(torch.stack([RES["acc"][L] for L in layers], 1))
    RES["on"] = None
    return torch.cat(out)


@torch.no_grad()
def generate_dense(prompts, vecs, max_new, seed):
    """vecs: one (L, vector) per prompt."""
    outs = []
    for i in range(0, len(prompts), BS):
        DENSE.clear()
        per = {}
        for r, (L, v) in enumerate(vecs[i:i + BS]):
            per.setdefault(L, []).append((r, v))
        for L, items in per.items():
            DENSE[L] = (torch.tensor([r for r, _ in items]), torch.stack([v for _, v in items]))
        enc = tok(prompts[i:i + BS], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        torch.manual_seed(seed + i)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.9,
                           repetition_penalty=1.05, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    DENSE.clear()
    return [o.strip() for o in outs]


def checkpoint():
    res["runtime_s"] = time.time() - T0
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)


# 1. unsteered baseline at the finals length (run 2's baseline used 64 tokens)
log("baseline at finals length")
qs2 = [q for q in EVAL_Q for _ in range(2)]
outs = generate([chat(q) for q in qs2], max_new=GEN_FINAL, seed=5)
res["baseline120"] = [{"concept": "none", "neurons": [], "samples": [{"q": q, "out": o} for q, o in zip(qs2, outs)]}]

# 2. system-prompt persona baseline (finals protocol)
log("system-prompted personas")
res["prompted"] = []
for c, (sysmsg, _) in CONCEPTS.items():
    outs = generate([chat(q, sysmsg) for q in qs2], max_new=GEN_FINAL, seed=5)
    res["prompted"].append({"concept": c, "neurons": [], "samples": [{"q": q, "out": o} for q, o in zip(qs2, outs)]})
checkpoint()

# 3. dense contrastive persona vectors at the same residual-sized doses (finals protocol)
log("persona responses for dense vectors")
neutral = resid_means(DISC_Q, base_resp, DENSE_LAYERS)
res["dense"] = []
for c, (sysmsg, _) in CONCEPTS.items():
    pr = generate([chat(q, sysmsg) for q in DISC_Q], max_new=GEN_PERSONA, seed=2)
    diff = (resid_means(DISC_Q, pr, DENSE_LAYERS) - neutral).mean(0)  # [layers, H]
    for li, L in enumerate(DENSE_LAYERS):
        u = diff[li] / diff[li].norm()
        for d in DENSE_DOSES:
            res["dense"].append({"concept": c, "layer": L, "dose": d, "vec": d * MED[L] * u, "neurons": []})
log(f"dense: {len(res['dense'])} configs x {len(qs2)} answers")
prompts, vecs, owner = [], [], []
for ci, cfg in enumerate(res["dense"]):
    for q in qs2:
        prompts.append(chat(q))
        vecs.append((cfg["layer"], cfg["vec"]))
        owner.append((ci, q))
outs = generate_dense(prompts, vecs, GEN_FINAL, seed=5)
for cfg in res["dense"]:
    cfg["samples"] = []
    del cfg["vec"]
for (ci, q), o in zip(owner, outs):
    res["dense"][ci]["samples"].append({"q": q, "out": o})
checkpoint()

# 4. random-neuron control: same layer and signed dose as every run-2 screened config, screen protocol
log(f"random control: {len(RANDOM)} configs x {SCREEN_Q} prompts")
run_configs(RANDOM, EVAL_Q[:SCREEN_Q], GEN_SCREEN, seed=4)
res["random"] = RANDOM
checkpoint()
log("done controls")
