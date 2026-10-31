"""Render results.json from the neuron hunt into a self-contained HTML report.

usage: python3 -I report.py <results.json> <report.html>
"""
import html
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
r = json.load(open(src))
esc = html.escape


def spec(neurons):
    return ", ".join(f"L{L} N{n} {d:+.2f}" for L, n, d in neurons)


def bar(v, lo=-0.1, hi=0.6):
    # horizontal bar on a fixed lo..hi axis, so rows compare across concepts
    span = hi - lo
    z = (0 - lo) / span * 100
    x = (min(max(v, lo), hi) - lo) / span * 100
    left, width = (z, x - z) if x >= z else (x, z - x)
    cls = "pos" if v >= 0 else "neg"
    return (f'<div class="track"><div class="zero" style="left:{z:.1f}%"></div>'
            f'<div class="bar {cls}" style="left:{left:.1f}%;width:{width:.1f}%"></div></div>')


base = {s["q"]: s["out"] for s in r.get("baseline_samples", [])}
winners = r.get("winners", {})
finals = r.get("finals", [])
concepts = sorted(winners, key=lambda c: -winners[c]["lift"])

rows = []
for c in concepts:
    w = winners[c]
    rows.append(f"<tr><td class='c'>{esc(c)}</td><td class='mono'>{esc(spec(w['neurons']))}</td>"
                f"<td class='num'>{w['lift']:.3f}</td><td>{bar(w['lift'])}</td>"
                f"<td class='num'>{w['p_concept']:.2f} <span class='dim'>vs {r['baseline_concept_rate'][c]:.2f}</span></td>"
                f"<td class='num'>{w['p_coherent']:.2f}</td></tr>")

sections = []
for c in concepts:
    w = winners[c]
    mine = sorted([f for f in finals if f["concept"] == c], key=lambda f: -f["lift"])
    frows = "".join(
        f"<tr{' class=win' if f['neurons'] == w['neurons'] else ''}><td class='mono'>{esc(spec(f['neurons']))}</td>"
        f"<td class='dim'>{esc(f.get('kind', ''))}</td><td class='num'>{f['lift']:.3f}</td><td>{bar(f['lift'])}</td>"
        f"<td class='num'>{f['p_coherent']:.2f}</td></tr>" for f in mine)
    seen, pairs = set(), []
    for s in w["samples"]:
        if s["q"] in seen:
            continue
        seen.add(s["q"])
        pairs.append(f"<div class='pair'><div class='q'>{esc(s['q'])}</div>"
                     f"<div class='cols'><div><div class='lab'>baseline</div><p>{esc(base.get(s['q'], '—'))}</p></div>"
                     f"<div><div class='lab'>steered</div><p>{esc(s['out'])}</p></div></div></div>")
        if len(pairs) == 4:
            break
    sections.append(f"<section><h2>{esc(c)} <span class='mono dim'>{esc(spec(w['neurons']))}</span></h2>"
                    f"<p class='dim'>persona prompt sample: {esc((r.get('persona_samples', {}).get(c) or [''])[0][:220])}…</p>"
                    f"<table class='finals'><tr><th>config</th><th>kind</th><th>lift</th><th></th><th>coherent</th></tr>{frows}</table>"
                    + "".join(pairs) + "</section>")

wild = ""
if r.get("wildcards"):
    items = sorted(r["wildcards"], key=lambda x: -x["p_coherent"])
    wild = "<section><h2>Wildcards <span class='dim'>persona-general neurons, style described blind</span></h2>" + "".join(
        f"<div class='pair'><div class='q mono'>{esc(spec(x['neurons']))} · coherent {x['p_coherent']:.2f} · "
        f"<b>{esc(x['samples'][0]['style'])}</b> / <b>{esc(x['samples'][1]['style'])}</b></div>"
        f"<p>{esc(x['samples'][0]['out'][:320])}</p></div>" for x in items) + "</section>"

doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Neuron hunt: {esc(r['model'])}</title>
<style>
:root{{--bg:#fbfbf8;--ink:#16161a;--dim:#6b6b73;--line:#e4e4de;--pos:#2563eb;--neg:#c2410c;--win:#eef3ff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#131316;--ink:#ececef;--dim:#9a9aa3;--line:#2a2a30;--pos:#6b9cff;--neg:#f08a5d;--win:#1b2233}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif;max-width:1040px;margin:40px auto;padding:0 24px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:40px 0 6px}}
.dim{{color:var(--dim);font-weight:400}} .mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} th{{text-align:left;color:var(--dim);font-weight:500;font-size:12px}}
td,th{{padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:middle}} td.num{{font-variant-numeric:tabular-nums;white-space:nowrap}}
td.c{{font-weight:600}} tr.win td{{background:var(--win)}}
.track{{position:relative;height:10px;width:180px;background:var(--line);border-radius:2px}}
.zero{{position:absolute;top:-2px;bottom:-2px;width:1px;background:var(--dim)}}
.bar{{position:absolute;top:0;bottom:0;border-radius:2px}} .pos{{background:var(--pos)}} .neg{{background:var(--neg)}}
.pair{{border-top:1px solid var(--line);padding:12px 0}} .q{{font-weight:600;margin-bottom:6px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:20px}} .cols p,.pair p{{margin:0;white-space:pre-wrap;font-size:14px}}
.lab{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim)}}
</style></head><body>
<h1>Neuron hunt: {esc(r['model'])}</h1>
<p class="dim">One MLP neuron (or a pair) pushed along its output direction. Dose is in units of that layer's median residual norm.
Lift = mean(p_concept × p_coherent) − baseline concept rate, judged by the model itself. Runtime {r.get('runtime_s', 0) / 60:.0f} min{' · SMOKE RUN' if r.get('smoke') else ''}.</p>
<table><tr><th>concept</th><th>winner</th><th>lift</th><th></th><th>p_concept</th><th>coherent</th></tr>{''.join(rows)}</table>
{''.join(sections)}{wild}
</body></html>"""
open(dst, "w").write(doc)
print("wrote", dst, len(doc), "bytes,", len(concepts), "concepts")
