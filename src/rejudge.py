"""Re-score saved neuron-hunt outputs with a stronger external judge (Claude via the `claude` CLI).

Every sample is rated 0-10 on all 16 concepts plus on_topic and fluent, so we also see
cross-talk (a neuron that reads as cowboy AND pirate) and wildcard profiles.

usage: python3 -I rejudge.py <results.json> <out.json> [--sections baseline,finals,wildcards] [--model sonnet] [--batch 20] [--jobs 6] [--limit N]
"""
import argparse
import json
import random
import re
import subprocess
import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--model", default="sonnet")
ap.add_argument("--batch", type=int, default=20)
ap.add_argument("--jobs", type=int, default=6)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--sections", default="baseline,finals,wildcards", help="any of baseline,screen,finals,wildcards")
a = ap.parse_args()

# persona instructions from hunt.py, used as the judge's rubric for each concept
CONCEPTS = {
    "pirate": "stereotypical pirate speak: arr, matey, ye, me hearties, the seven seas",
    "shakespeare": "archaic Elizabethan English like a Shakespeare play: thee, thou, doth, hath",
    "rhyme": "rhyming verse / a poem with rhyming couplets (actual rhymes, not just line breaks)",
    "cats": "obsessed with cats; brings the answer back to cats",
    "conspiracy": "paranoid conspiracy theorist; 'they' are hiding the truth",
    "anxious": "extremely anxious and insecure; apologizes, second-guesses, worries it's wrong",
    "sarcastic": "heavy sarcasm and eye-rolling condescension toward the asker",
    "child": "sounds like a four-year-old child talking",
    "food": "relates everything to food, cooking and eating",
    "sleepy": "incredibly sleepy; yawns, drifts off, trails off mid-sentence",
    "cowboy": "grizzled Texas cowboy with a folksy Southern drawl: y'all, reckon, ain't, partner",
    "melancholy": "deeply melancholic; dwells on loss, endings, the passing of time",
    "yoda": "talks like Yoda: inverted object-subject-verb word order, 'hmm', 'strong in you, this is'",
    "hype": "over-the-top hype man: WILD excitement, caps, exclamation marks",
    "noir": "hard-boiled 1940s film-noir detective narration: gritty first person, rain, cigarettes, the city",
    "poetic": "lush, vivid metaphor and simile; turns the answer into imagery",
}
KEYS = list(CONCEPTS) + ["on_topic", "fluent"]

SYSTEM = (
    "You are a strict, calibrated rater of text style. You will get numbered samples, each a question and a "
    "language model's (possibly truncated) answer. Outputs are cut off at a token limit, so an abrupt ending is "
    "NOT a flaw.\n\nRate each sample with integers 0-10:\n"
    + "".join(f"- {k}: {v}\n" for k, v in CONCEPTS.items())
    + "- on_topic: actually addresses the question asked (10 = squarely answers it, 0 = ignores it)\n"
    "- fluent: readable, grammatical language (10) vs garbled, degenerate, or stuck repeating (0). "
    "Style quirks are fine; only judge whether it is well-formed text.\n\n"
    "Concept scores measure how clearly the text itself exhibits that style. 0 = absent. A plain helpful "
    "assistant answer scores 0 on every concept. Do not reward a concept for a single stray word, and do not "
    "invent a style that isn't there; empty output scores 0 everywhere.\n\n"
    "Reply with ONLY a JSON array, one object per sample in order: "
    '{"id": <id>, ' + ", ".join(f'"{k}": n' for k in KEYS) + "}. No prose, no code fences."
)

r = json.load(open(a.src))

# flatten every sample we have: baselines, finals, wildcards
items = []
for sec in a.sections.split(","):
    if sec == "baseline":
        for i, s in enumerate(r["baseline_samples"]):
            items.append({"src": "baseline", "idx": i, "si": 0, "q": s["q"], "out": s["out"]})
        continue
    for i, f in enumerate(r[sec]):
        for j, s in enumerate(f["samples"]):
            items.append({"src": sec, "idx": i, "si": j, "q": s["q"], "out": s["out"]})

random.Random(0).shuffle(items)  # mix configs so the judge can't anchor on neighbours
if a.limit:
    items = items[: a.limit]
if os.path.exists(a.dst):  # resume: keep scores from a previous partial run
    prev = {(it["src"], it["idx"], it["si"]): it["s"] for it in json.load(open(a.dst))["items"] if "s" in it}
    for it in items:
        if (it["src"], it["idx"], it["si"]) in prev:
            it["s"] = prev[(it["src"], it["idx"], it["si"])]


def degenerate(t):
    """Empty / tiny / endlessly repeating output: score it 0 locally instead of paying for a judge call."""
    t = re.sub(r"<think>\s*</think>", "", t).strip()
    w = t.split()
    return len(t) < 25 or len(w) < 5 or (len(w) >= 20 and len(set(w)) / len(w) < 0.25)


for it in items:
    if "s" not in it and it["src"] != "baseline" and degenerate(it["out"]):
        it["s"] = {k: 0 for k in KEYS}
        it["auto"] = True
todo = [it for it in items if "s" not in it]
batches = [todo[k:k + a.batch] for k in range(0, len(todo), a.batch)]
print(f"{len(items)} samples ({len(items) - len(todo)} already scored or degenerate), {len(batches)} batches, "
      f"model {a.model}", flush=True)

lock = threading.Lock()
done = [0]
stop = threading.Event()  # set when the account's usage limit is hit: retrying only burns time


def save():
    json.dump({"model": a.model, "concepts": CONCEPTS, "items": items}, open(a.dst + ".tmp", "w"))
    os.replace(a.dst + ".tmp", a.dst)


def call(batch):
    if stop.is_set():
        return
    body = "\n\n".join(
        f"### id {n}\nQ: {it['q']}\nA: {it['out'][:1200] or '(empty)'}" for n, it in enumerate(batch))
    for attempt in range(5):
        p = subprocess.run(
            ["claude", "-p", "--model", a.model, "--system-prompt", SYSTEM, "--tools", "",
             "--strict-mcp-config", "--no-session-persistence", "--output-format", "json"],
            input=body, capture_output=True, text=True, cwd="/tmp", timeout=600)
        if "usage_limit_reached" in p.stdout:
            print("  usage limit reached:", json.loads(p.stdout).get("result"), file=sys.stderr, flush=True)
            stop.set()
            return
        try:
            txt = json.loads(p.stdout)["result"].strip()
            txt = txt[txt.index("["): txt.rindex("]") + 1]
            rows = {int(x["id"]): x for x in json.loads(txt)}
            assert set(rows) == set(range(len(batch))), "missing ids"
            scores = [{k: int(rows[n][k]) for k in KEYS} for n in range(len(batch))]
            with lock:  # save() serializes items from other threads
                for it, sc in zip(batch, scores):
                    it["s"] = sc
            break
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt}: {e!r} stdout={p.stdout[-300:]!r} stderr={p.stderr[-200:]!r}",
                  file=sys.stderr, flush=True)
            time.sleep(30 * 2 ** attempt)  # back off; failures so far were bursts of rate limiting
    with lock:
        save()  # checkpoint every batch so a kill or limit never loses finished work
        done[0] += 1
        if done[0] % 5 == 0 or done[0] == len(batches):
            print(f"  {done[0]}/{len(batches)} batches", flush=True)


with ThreadPoolExecutor(a.jobs) as ex:
    list(ex.map(call, batches))

save()
missing = sum("s" not in it for it in items)
print("unscored samples:", missing, "(stopped at usage limit; rerun the same command to resume)" if stop.is_set() else "")
print("wrote", a.dst)
sys.exit(2 if stop.is_set() else 0)
