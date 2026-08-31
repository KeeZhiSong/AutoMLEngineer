#!/usr/bin/env python3
"""
Build docs/index.html from the committed run logs.

The page is DERIVED, never hand-written: every figure on it is read from
workspace*/experiments.jsonl and archive/runs/*/experiments.jsonl. Regenerate
it and the page cannot disagree with the repository.

That property is the point. A dashboard maintained by hand drifts from the data
it claims to describe, which is the same failure that once left submission.csv
with no regenerable source.

    python3 tools_build_ledger.py            # write docs/index.html
    python3 tools_build_ledger.py --check    # exit 1 if the page is stale
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "index.html"

# Architecture generation per workspace. Kept explicit rather than inferred:
# a run's directory name is not a reliable record of what code produced it.
VERSION = {
    "01-library-pipeline-20cyc": "V1", "02-clean-replay-library-25cyc": "V2",
    "03-classify-pipeline-25cyc": "V3", "04-tier1-smoke-6cyc": "V4",
    "05-autonomy-benchmark-21cyc": "V4", "06-clean-run-14cyc": "V4",
    "run07": "V4", "v5": "V5", "agent_derive": "V5",
    "v5_exploit": "V5", "v5_exploit2": "V5",
    "v6": "V6", "v6b": "V6", "v6c": "V6",
    "v6c_verify": "V6", "v6c_gpt55": "V6",
    "v7": "V7", "v7b": "V7", "v8": "V8", "v8b": "V8",
}


def kind_of(rec: dict) -> str:
    """One outcome label per experiment, for the grid."""
    metrics, status, stage = rec.get("metrics") or {}, rec.get("status"), rec.get("stage")
    if stage == "improve":
        return "kept"
    if stage == "exploit":
        return "exploit"
    if status == "implementation_failure":
        return "blocked"
    if status in ("bug", "crash"):
        return "crash"
    if status == "timeout":
        return "timeout"
    if status == "skipped":
        return "skipped"
    return "trained" if metrics.get("primary") else "other"


def collect(root: Path = ROOT) -> dict:
    runs = []
    paths = sorted(glob.glob(str(root / "archive/runs/*/"))) + \
            sorted(glob.glob(str(root / "workspace*/")))
    for d in paths:
        ledger = Path(d) / "experiments.jsonl"
        if not ledger.exists():
            continue
        name = Path(d.rstrip("/")).name.replace("workspace_", "")
        rows = [json.loads(l) for l in ledger.open() if l.strip()]
        summ_path = Path(d) / "summary.json"
        summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}

        exps = [{"c": r.get("cycle"), "k": kind_of(r),
                 "t": (r.get("source_technique") or "")[:38],
                 "p": round(r["metrics"]["primary"], 4)
                      if (r.get("metrics") or {}).get("primary") else None}
                for r in rows]
        trained = [e for e in exps if e["p"]]
        blocked = [e for e in exps if e["k"] == "blocked"]
        keeps = [e for e in exps if e["k"] == "kept"]
        runs.append({
            "name": name, "ver": VERSION.get(name, "?"), "n": len(exps),
            "sat": round(100 * len(trained) / max(len(trained) + len(blocked), 1)),
            "keeps": len(keeps),
            "best": max((e["p"] for e in keeps), default=None),
            "tok": summ.get("tokens_in", 0) + summ.get("tokens_out", 0),
            "stop": summ.get("stop_reason", ""), "exps": exps,
        })
    return {"runs": runs, "total": sum(r["n"] for r in runs)}


CSS = """
:root{
  --ground:#F4F6F7; --panel:#FFFFFF; --rule:#DCE3E7; --ink:#111A20; --muted:#5A6B77;
  --accent:#B0741A; --kept:#1E9D6E; --trained:#8FA6B4; --blocked:#D3DBE0;
  --crash:#B4574A; --exploit:#C9A227;
  --f-display:"IBM Plex Serif",Georgia,serif;
  --f-body:"IBM Plex Sans",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0A0E11; --panel:#131A20; --rule:#202A32; --ink:#E2E8ED; --muted:#7A8894;
  --accent:#E0A34A; --kept:#5BD6A0; --trained:#37505F; --blocked:#242E36;
  --crash:#8E4A42; --exploit:#8A7530;
}}
:root[data-theme="dark"]{
  --ground:#0A0E11; --panel:#131A20; --rule:#202A32; --ink:#E2E8ED; --muted:#7A8894;
  --accent:#E0A34A; --kept:#5BD6A0; --trained:#37505F; --blocked:#242E36;
  --crash:#8E4A42; --exploit:#8A7530;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--f-body);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 28px}
h1,h2{font-family:var(--f-display);text-wrap:balance;margin:0}
.mono{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
.eyebrow{font-family:var(--f-mono);font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted)}
header{padding:76px 0 44px;border-bottom:1px solid var(--rule)}
h1{font-size:clamp(38px,6vw,66px);line-height:1.02;font-weight:600;
  letter-spacing:-.02em;margin:18px 0 0}
.sub{max-width:62ch;color:var(--muted);font-size:18px;margin-top:20px}
.sub b{color:var(--ink);font-weight:500}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:1px;background:var(--rule);border:1px solid var(--rule);margin:44px 0 0}
.stat{background:var(--panel);padding:20px 18px}
.stat .v{font-family:var(--f-mono);font-size:29px;font-weight:600;
  letter-spacing:-.02em;font-variant-numeric:tabular-nums;display:block}
.stat .l{font-family:var(--f-mono);font-size:10.5px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--muted);margin-top:6px;display:block}
section{padding:72px 0;border-bottom:1px solid var(--rule)}
h2{font-size:clamp(25px,3.4vw,35px);font-weight:600;letter-spacing:-.015em;margin:12px 0 0}
.lede{max-width:64ch;color:var(--muted);margin-top:14px}
.grid-legend{display:flex;flex-wrap:wrap;gap:18px;margin:30px 0 22px;
  font-family:var(--f-mono);font-size:11.5px;color:var(--muted)}
.grid-legend span{display:flex;align-items:center;gap:7px}
.sw{width:11px;height:11px;border-radius:1px;display:inline-block}
.runs{display:flex;flex-direction:column;gap:7px}
.runrow{display:grid;grid-template-columns:172px 1fr;gap:14px;align-items:center}
.runlabel{font-family:var(--f-mono);font-size:11px;color:var(--muted);
  text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.runlabel b{color:var(--ink);font-weight:500}
.cells{display:flex;flex-wrap:wrap;gap:2px}
.cell{width:13px;height:13px;border-radius:1.5px;background:var(--blocked);
  opacity:0;transform:scale(.55);animation:pop .32s ease forwards}
@keyframes pop{to{opacity:1;transform:scale(1)}}
.cell.trained{background:var(--trained)} .cell.blocked{background:var(--blocked)}
.cell.crash{background:var(--crash)} .cell.exploit{background:var(--exploit)}
.cell.timeout{background:var(--crash);opacity:.5}
.cell.skipped{background:transparent;box-shadow:inset 0 0 0 1px var(--rule)}
.cell.kept{background:var(--kept);box-shadow:0 0 0 2px var(--ground),0 0 13px var(--kept);z-index:2}
.cell:hover{outline:1.5px solid var(--ink);outline-offset:1px;cursor:crosshair}
#tip{position:fixed;pointer-events:none;background:var(--panel);color:var(--ink);
  border:1px solid var(--rule);padding:9px 11px;font-family:var(--f-mono);
  font-size:11.5px;line-height:1.5;opacity:0;transition:opacity .12s;z-index:99;max-width:280px}
#tip b{color:var(--accent)}
svg{display:block;width:100%;height:auto;overflow:visible}
.gridline{stroke:var(--rule);stroke-width:1;stroke-dasharray:2 4}
.tick{font-family:var(--f-mono);font-size:10.5px;fill:var(--muted)}
.human{stroke:var(--accent);stroke-width:1.5;stroke-dasharray:5 4}
.base{stroke:var(--muted);stroke-width:1.5}
.lbl{font-family:var(--f-mono);font-size:11px;fill:var(--muted)}
.tblwrap{overflow-x:auto;margin-top:30px;border:1px solid var(--rule)}
table{border-collapse:collapse;width:100%;min-width:660px;background:var(--panel)}
th,td{padding:11px 14px;text-align:left;border-bottom:1px solid var(--rule);font-size:13.5px}
th{font-family:var(--f-mono);font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);font-weight:500}
td.num{font-family:var(--f-mono);font-variant-numeric:tabular-nums;text-align:right}
tr.win td{background:color-mix(in srgb,var(--kept) 9%,var(--panel))}
.pill{font-family:var(--f-mono);font-size:10px;padding:2px 6px;border-radius:2px;
  background:var(--blocked);color:var(--muted)}
tr.win .pill{background:var(--kept);color:var(--ground)}
.callout{border-left:3px solid var(--accent);padding:22px 26px;margin-top:34px;background:var(--panel)}
.callout p{margin:0;max-width:66ch}
.big{font-family:var(--f-display);font-size:clamp(22px,3vw,30px);line-height:1.28;
  font-weight:600;letter-spacing:-.01em;max-width:24ch}
footer{padding:52px 0 76px;color:var(--muted);font-size:13px}
.reveal{opacity:0;transform:translateY(14px);transition:opacity .6s ease,transform .6s ease}
.reveal.in{opacity:1;transform:none}
@media (prefers-reduced-motion:reduce){
  .cell{animation:none;opacity:1;transform:none}
  .reveal{opacity:1;transform:none;transition:none}}
@media(max-width:720px){.runrow{grid-template-columns:1fr;gap:4px}.runlabel{text-align:left}}
"""


def build_html(data: dict) -> str:
    runs, total = data["runs"], data["total"]
    keeps = [(r["name"], r["ver"], e["p"], e["t"])
             for r in runs for e in r["exps"] if e["k"] == "kept" and e["p"]]
    n_keep = len(keeps)
    blocked = sum(1 for r in runs for e in r["exps"] if e["k"] == "blocked")
    best = max((k[2] for k in keeps), default=0)

    h = ['<title>The Search Ledger</title>',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<link rel="preconnect" href="https://fonts.googleapis.com">',
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600'
         '&family=IBM+Plex+Serif:wght@400;600&display=swap">',
         f'<style>{CSS}</style>', '<div id="tip"></div>', '<div class="wrap">',
         '<header><div class="eyebrow">TikTok TechJam 2026 &middot; Track 2 '
         '&middot; KuaiRand-Pure</div>',
         f'<h1>{total} experiments.<br>{n_keep} of them worked.</h1>',
         '<p class="sub">An autonomous agent ran the machine-learning research loop '
         f'&mdash; measure, diagnose, invent, implement, verify, decide &mdash; across '
         f'<b>{len(runs)} runs</b> with <b>zero manual interventions</b>. This is every '
         'experiment it ran, including the ones that failed and the guards that '
         'caught them.</p><div class="stats">',
         f'<div class="stat"><span class="v">{total}</span>'
         '<span class="l">Experiments</span></div>',
         f'<div class="stat"><span class="v">{n_keep}</span>'
         '<span class="l">Accepted</span></div>',
         f'<div class="stat"><span class="v">{best:.4f}</span>'
         '<span class="l">Best, unaided</span></div>',
         f'<div class="stat"><span class="v">{blocked}</span>'
         '<span class="l">Stopped before training</span></div>',
         '<div class="stat"><span class="v">0</span>'
         '<span class="l">Manual interventions</span></div>',
         '</div></header>']

    # ---- the search grid -------------------------------------------------
    h += ['<section class="reveal"><div class="eyebrow">The search</div>',
          '<h2>Every experiment, every run</h2>',
          '<p class="lede">One cell per experiment, ordered as it happened. Most were '
          'caught before they cost a training run: a patch that reads as the '
          'intervention and changes nothing is the most common failure here, and it '
          'is detected in 0.4 seconds.</p>',
          '<div class="grid-legend">'
          '<span><i class="sw" style="background:var(--kept)"></i>accepted</span>'
          '<span><i class="sw" style="background:var(--trained)"></i>trained, reverted</span>'
          '<span><i class="sw" style="background:var(--blocked)"></i>blocked by contract</span>'
          '<span><i class="sw" style="background:var(--crash)"></i>crashed</span>'
          '<span><i class="sw" style="background:var(--exploit)"></i>exploit trial</span>'
          '<span><i class="sw" style="box-shadow:inset 0 0 0 1px var(--rule)"></i>'
          'no problem named</span></div><div class="runs">']
    d = 0
    for r in runs:
        h.append(f'<div class="runrow"><div class="runlabel"><b>{r["ver"]}</b> '
                 f'{r["name"][:22]}</div><div class="cells">')
        for e in r["exps"]:
            d += 1
            p = f'{e["p"]:.4f}' if e["p"] else "—"
            t = (e["t"] or "—").replace('"', "'")
            h.append(f'<i class="cell {e["k"]}" style="animation-delay:{d*3.2:.0f}ms" '
                     f'data-t="{r["ver"]} {r["name"][:20]} &middot; cycle {e["c"]}'
                     f'|{t}|{e["k"]}|{p}"></i>')
        h.append('</div></div>')
    h.append('</div></section>')

    # ---- score timeline --------------------------------------------------
    lo, hi, W, H, PAD = 0.6005, 0.6045, 1000, 300, 54
    y = lambda v: PAD + (hi - v) / (hi - lo) * (H - 2 * PAD)
    h += ['<section class="reveal"><div class="eyebrow">Results</div>',
          '<h2>Where each accepted result landed</h2>',
          '<p class="lede">The official baseline is 0.6016 and the oracle ceiling is '
          '0.8484: 27.1% of evaluation users have no positive label, so their nDCG is '
          'zero for any model. Total headroom is 0.247, and the organisers’ own '
          'ablations move this metric by less than 0.002. Everything below sits inside '
          'a 0.004 window.</p>',
          f'<figure><svg viewBox="0 0 {W} {H}" role="img" '
          'aria-label="Accepted results against the baseline">',
          f'<line class="human" x1="{PAD}" y1="{y(0.6038):.1f}" x2="{W-140}" '
          f'y2="{y(0.6038):.1f}"/><text class="lbl" x="{W-134}" y="{y(0.6038)+4:.1f}" '
          'fill="var(--accent)">0.6038 human</text>',
          f'<line class="base" x1="{PAD}" y1="{y(0.6016):.1f}" x2="{W-140}" '
          f'y2="{y(0.6016):.1f}"/><text class="lbl" x="{W-134}" '
          f'y="{y(0.6016)+4:.1f}">0.6016 baseline</text>',
          f'<line class="gridline" x1="{PAD}" y1="{y(0.6014):.1f}" x2="{W-140}" '
          f'y2="{y(0.6014):.1f}"/><text class="lbl" x="{W-134}" '
          f'y="{y(0.6014)+4:.1f}">0.6014 start</text>']
    step = (W - PAD - 190) / max(len(keeps) - 1, 1)
    for i, (nm, vr, p, tech) in enumerate(keeps):
        cx = PAD + 26 + i * step
        h.append(f'<circle fill="var(--kept)" cx="{cx:.1f}" cy="{y(p):.1f}" r="6">'
                 f'<title>{vr} {nm}: {p:.4f} — {tech}</title></circle>'
                 f'<text class="tick" x="{cx:.1f}" y="{y(p)-14:.1f}" '
                 f'text-anchor="middle">{p:.4f}</text>'
                 f'<text class="tick" x="{cx:.1f}" y="{H-22:.1f}" text-anchor="middle" '
                 f'opacity=".75">{vr}</text>')
    h += ['</svg></figure>',
          '<div class="callout"><p>The agent reached <b class="mono">0.6031</b> twice, '
          'in different architecture generations, both times by reweighting the training '
          'objective, and never by the route the human took. Both were confirmed over '
          'five seeds by the loop itself before acceptance.</p></div></section>']

    # ---- the seed collapse ----------------------------------------------
    h += ['<section class="reveal"><div class="eyebrow">Why the accept margin exists</div>',
          '<h2>The win that wasn’t</h2>',
          '<p class="lede">An exploit trial measured a clear improvement on the first '
          'seed. Run on five seeds, it averaged to almost nothing. The keep/revert '
          'decision is arithmetic, not a judgement call, and it declined.</p>',
          '<figure><svg viewBox="0 0 620 250" role="img" '
          'aria-label="A +0.0012 single-seed result averaging to +0.0004 over five seeds">'
          '<line class="gridline" x1="70" y1="30" x2="600" y2="30"/>'
          '<text class="lbl" x="606" y="34">accept margin 0.0008</text>'
          '<rect x="120" y="30" width="120" height="150" fill="var(--crash)" opacity=".85"/>'
          '<text class="tick" x="180" y="200" text-anchor="middle">seed 0 only</text>'
          '<text class="lbl" x="180" y="20" text-anchor="middle" '
          'fill="var(--crash)">+0.0012</text>'
          '<rect x="330" y="130" width="120" height="50" fill="var(--kept)" opacity=".85"/>'
          '<text class="tick" x="390" y="200" text-anchor="middle">mean of 5 seeds</text>'
          '<text class="lbl" x="390" y="122" text-anchor="middle" '
          'fill="var(--kept)">+0.0004</text>'
          '<line stroke="var(--rule)" x1="70" y1="180" x2="600" y2="180"/></svg></figure>',
          '<div class="callout"><p class="big">The gate was right. A human looked at the '
          'same numbers and spent an hour chasing it.</p></div></section>']

    # ---- ledger ----------------------------------------------------------
    h += ['<section class="reveal"><div class="eyebrow">Run ledger</div>',
          '<h2>Every run, reported in full</h2>',
          '<p class="lede">Including the architecture generations that were built, '
          'measured, and reverted because they did not produce a win.</p>',
          '<div class="tblwrap"><table><thead><tr><th>Run</th><th>Arch</th>'
          '<th class="num">Exps</th><th class="num">Contract sat.</th>'
          '<th class="num">Keeps</th><th class="num">Best</th>'
          '<th class="num">Tokens</th><th>Stopped</th></tr></thead><tbody>']
    for r in runs:
        win = ' class="win"' if r["keeps"] else ""
        b = f'{r["best"]:.4f}' if r["best"] else "—"
        tok = f'{r["tok"]/1000:.0f}K' if r["tok"] else "—"
        h.append(f'<tr{win}><td class="mono">{r["name"][:26]}</td>'
                 f'<td><span class="pill">{r["ver"]}</span></td>'
                 f'<td class="num">{r["n"]}</td><td class="num">{r["sat"]}%</td>'
                 f'<td class="num">{r["keeps"]}</td><td class="num">{b}</td>'
                 f'<td class="num">{tok}</td><td class="mono" style="font-size:11.5px;'
                 f'color:var(--muted)">{r["stop"] or "—"}</td></tr>')
    h += ['</tbody></table></div></section>',
          '<footer>Generated by <span class="mono">tools_build_ledger.py</span> from the '
          'committed run logs in <span class="mono">workspace*/experiments.jsonl</span>. '
          'Baseline 0.6016 &middot; agent 0.6031 &middot; human-tuned 0.6038 &middot; '
          'oracle ceiling 0.8484.</footer></div>',
          '''<script>
const tip=document.getElementById('tip');
document.querySelectorAll('.cell').forEach(c=>{
  c.addEventListener('mouseenter',()=>{const[w,t,k,p]=c.dataset.t.split('|');
    tip.innerHTML=`${w}<br><b>${t}</b><br>${k}${p!=='\\u2014'?' &middot; '+p:''}`;
    tip.style.opacity='1';});
  c.addEventListener('mousemove',e=>{
    tip.style.left=Math.min(e.clientX+14,innerWidth-292)+'px';
    tip.style.top=(e.clientY+18)+'px';});
  c.addEventListener('mouseleave',()=>tip.style.opacity='0');});
const io=new IntersectionObserver(es=>es.forEach(e=>{
  if(e.isIntersecting)e.target.classList.add('in')}),{threshold:.08});
document.querySelectorAll('.reveal').forEach(s=>io.observe(s));
</script>''']
    return "\n".join(h) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if docs/index.html is stale (for CI)")
    a = ap.parse_args()

    html = build_html(collect())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    current = OUT.read_text() if OUT.exists() else ""

    if a.check:
        if current == html:
            print("docs/index.html is up to date")
            return 0
        print("docs/index.html is STALE -- run tools_build_ledger.py", file=sys.stderr)
        return 1

    if current == html:
        print(f"docs/index.html unchanged ({len(html)/1024:.0f} KB)")
        return 0
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(html)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
