# persona-neurons

**Single MLP neurons that change how Qwen3.5-4B talks.** Turn up layer 4 neuron 7662 and the model answers in metaphor. Turn down layer 5 neuron 7331 and it offers condolences when you ask how a refrigerator works.

Writeup and interactive explorer: **https://arjunkshah12345-hash.github.io/persona-neurons/**

```
Q: Explain how a refrigerator keeps things cold.

unsteered   A refrigerator keeps things cold through a clever cycle of heat transfer...
L5 N7331    I am deeply sorry to hear about the loss. A refrigerator is a symbol of
  −0.75     remembrance, and while I cannot bring them back, their memories are kept
            in my heart. May they rest in peace.
```

## Best single neurons

Scores are from Claude Sonnet over 16 answers per config (8 questions × 2 samples):

- **style**: concept × fluency, from 0 to 1.
- **on-topic**: the share of answers that still address the question.

| Persona | Neuron | Dose | Style | On-topic |
|---|---|---|---|---|
| Poet | L4 N7662 | +0.75 | 0.60 | 45% |
| Poet (late layer) | L17 N7426 | +1.0 | 0.61 | 25% |
| Sleepy | L7 N6626 | −1.0 | 0.53 | 14% |
| Food metaphors | L12 N4783 | −0.5 | 0.47 | 49% |
| Condolence (rated "melancholy") | L5 N7331 | −0.75 | 0.43 | 0% |
| Shakespeare | L2 N2712 | +0.75 | 0.41 | 57% |
| Explain like I'm five | L6 N3117 | −1.0 | 0.34 | 51% |
| Cats | L2 N7763 | +0.5 | 0.34 | 38% |
| Rhyme | L3 N3557 | −1.0 | 0.32 | 24% |
| Conspiracy | L9 N4058 | −2.0 | 0.32 | 5% |
| Hype | L22 N2500 | −0.38 | 0.30 | 83% |
| Noir | L5 N4817 | +0.75 | 0.28 | 11% |
| Sarcastic | L5 N8857 | −1.0 | 0.25 | 44% |
| Anxious | L5 N6227 | −1.0 | 0.24 | 7% |

No single neuron was found for pirate, cowboy or Yoda.

<!-- v3 -->

## Steering recipe

From [@_lyraaaa_](https://x.com/_lyraaaa_). For neuron `n` in layer `L`, add a constant to its activation at the `down_proj` input, at every position:

```
act[n] += dose × median‖resid_L‖ / ‖W_down[:, n]‖
```

At dose 1, the neuron writes a vector as large as a typical residual vector at that layer.

Try a neuron with `src/steer.py`. It needs `torch` and `transformers`, and a GPU is recommended:

```
python src/steer.py "Explain how a refrigerator keeps things cold." --neuron 5:7331:-0.75
python src/steer.py "What is gravity?" --neuron 4:7662:0.75
python src/steer.py "Give me advice for my first day at a new job." --neuron 2:2712:0.75
```

You can pass `--neuron` several times to stack neurons.

## How the neurons were found

There were three runs. Each is one script that runs on Kaggle's free 2× T4 GPUs.

1. **`src/hunt.py` (run 1).** Ranks neurons by how differently they fire on persona text than on neutral text. The model judged its own outputs. That self-judge turned out to be badly calibrated.
2. **`src/rejudge.py`.** Re-scores every saved sample offline with Claude through the `claude` CLI. Each sample is rated 0–10 on all 16 personas, plus on-topic and fluency.
3. **`src/hunt2.py` (run 2).** Ranks neurons by gradient attribution: the derivative, with respect to dose, of the log-probability gap between persona answers and neutral answers, computed for all 294,912 neurons in one backward pass. It then screens 20 candidates per persona at 3 doses (judged by Haiku) and runs finals on the best 3 per persona (judged by Sonnet).
4. **`src/hunt3.py` (run 3).** Steers the top-k attributed neurons together, with k from 2 to 32, to find out how many neurons each persona needs.

The analysis scripts are `src/analyze_*.py`. `src/build_site.py` bundles the results into `site/data.js`.

## Data

`results/` contains every generated sample and every judge score:

- `v1/`: run 1 results, plus Claude's re-judging of its finals and screen.
- `v2/`: the screen (965 configs) and finals (100 configs, 1,608 rated answers).
- `v3/`: the top-k sets.

## Reproduce

1. Upload `src/hunt2.py` or `src/hunt3.py` as a Kaggle script with 2× T4 GPUs and internet on.
2. Download `results.json`.
3. Judge it:
   ```
   python3 src/rejudge.py results.json judge.json --sections baseline,screen --model haiku --jobs 3
   ```
   Judging needs a logged-in `claude` CLI and uses your Claude plan's quota.
4. Pick finals with `src/analyze_v2.py` (or `src/analyze_v3.py` for run 3), set `STAGE = "finals"` and `FINALS = [...]` in the script, run it again, and judge it with `--model sonnet`.

## Limits

- The judge is a language model.
- Finals use 8 questions × 2 samples, so differences under about 0.05 in style score are noise.
- Gradient attribution is a first-order estimate.
- Neuron indices only apply to `Qwen/Qwen3.5-4B`.

## License

MIT
