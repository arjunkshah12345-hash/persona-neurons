"""Try a persona neuron yourself.

usage: python steer.py "What is gravity?" --neuron 4:7662:0.75 [--neuron L:n:dose ...] [--max-new 160]

Each --neuron adds dose * median||resid_L|| / ||W_down[:, n]|| to neuron n's down_proj input at every position.
Without --neuron it prints the unsteered answer.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
# median residual-stream norm leaving each layer, measured on neutral answers (results/v2/finals.json)
MED = [2.5, 3.2, 3.2, 4.6, 5.4, 6.2, 7.0, 7.4, 7.9, 8.2, 8.9, 9.0, 9.1, 9.5, 10.1, 10.4,
       10.8, 11.6, 13.8, 16.4, 18.5, 20.7, 24.1, 26.3, 29.1, 31.2, 34.4, 36.6, 41.6, 46.2, 51.1, 53.7]

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("--neuron", action="append", default=[], help="layer:index:dose, e.g. 5:7331:-0.75")
ap.add_argument("--max-new", type=int, default=160)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map="auto").eval()
downs = {int(name.split(".")[name.split(".").index("layers") + 1]): m for name, m in model.named_modules()
         if name.endswith("mlp.down_proj") and "visual" not in name and "mtp" not in name}

per_layer = {}
for spec in a.neuron:
    L, n, dose = spec.split(":")
    L, n, dose = int(L), int(n), float(dose)
    per_layer.setdefault(L, []).append((n, dose * MED[L] / downs[L].weight[:, n].float().norm().item()))


def make_hook(items):
    def pre(mod, args):
        x = args[0].clone()
        for n, delta in items:
            x[..., n] += delta
        return (x,) + tuple(args[1:])
    return pre


for L, items in per_layer.items():
    downs[L].register_forward_pre_hook(make_hook(items))

text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False, add_generation_prompt=True,
                               enable_thinking=False)
enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
torch.manual_seed(a.seed)
out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.7, top_p=0.9,
                     repetition_penalty=1.05)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
