"""Render an analysis as a text, Markdown or HTML report."""
from __future__ import annotations

import html
import json
from pathlib import Path

from ..config import CRITERION_LABELS, CRITERION_WEIGHTS
from ..analyze import AnalysisResult


def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score / 10.0 * width))
    return "#" * filled + "." * (width - filled)


def to_text(r: AnalysisResult) -> str:
    a, p = r.assessment, r.plan
    L = [
        "=" * 68,
        f"RUNWAY PERFORMANCE REPORT - {r.video_id}",
        "=" * 68,
        "",
        f"OVERALL   {a.overall:.1f}/100    Grade: {a.grade}",
        f"Scored by: {a.source}",
        "",
        "CRITERION SCORES",
        "-" * 68,
    ]
    for c, s in sorted(a.scores.items(), key=lambda kv: -kv[1]):
        L.append(f"  {CRITERION_LABELS[c]:32s} {s:4.1f}/10  {_bar(s)}  "
                 f"(w {CRITERION_WEIGHTS[c]:.2f})")

    if a.quality_warnings:
        L += ["", "RECORDING QUALITY", "-" * 68]
        L += [f"  ! {w}" for w in a.quality_warnings]

    L += ["", p.headline, "", p.summary, ""]

    if p.focus_areas:
        L += ["FOCUS AREAS", "-" * 68]
        for i, f in enumerate(p.focus_areas, 1):
            L.append(f"  {i}. {f['label']} ({f['score']}/10)")
            L.append(f"     {f['issue']}")
            L.append(f"     {f['feature']}: measured {f['measured']:.4g}  |  target {f['target']:.4g}")
            L.append("")

    if p.strengths:
        L += ["STRENGTHS", "-" * 68] + [f"  + {s}" for s in p.strengths] + [""]

    if p.drills:
        L += ["PRESCRIBED DRILLS", "-" * 68]
        for d in p.drills:
            L.append(f"  [{d['for']}] {d['name']}  ({d['duration']})")
            L.append(f"     {d['instruction']}")
            L.append("")

    L += ["NEXT SESSION", "-" * 68] + [f"  - {n}" for n in p.next_session]
    L += ["", "=" * 68]
    return "\n".join(L)


def to_markdown(r: AnalysisResult) -> str:
    a, p = r.assessment, r.plan
    L = [
        f"# Runway Performance Report - `{r.video_id}`",
        "",
        f"**Overall: {a.overall:.1f}/100 — {a.grade}**  ",
        f"_Scored by: {a.source}_",
        "",
        "## Criterion scores",
        "",
        "| Criterion | Score | Weight |",
        "|---|---:|---:|",
    ]
    L += [f"| {CRITERION_LABELS[c]} | {s:.1f}/10 | {CRITERION_WEIGHTS[c]:.2f} |"
          for c, s in sorted(a.scores.items(), key=lambda kv: -kv[1])]

    if a.quality_warnings:
        L += ["", "## Recording quality", ""] + [f"- {w}" for w in a.quality_warnings]

    L += ["", f"## {p.headline}", "", p.summary, ""]

    if p.focus_areas:
        L += ["## Focus areas", ""]
        for i, f in enumerate(p.focus_areas, 1):
            L += [f"**{i}. {f['label']} — {f['score']}/10**  ",
                  f"{f['issue']}  ",
                  f"_{f['feature']}: measured {f['measured']:.4g}, target {f['target']:.4g}_", ""]

    if p.strengths:
        L += ["## Strengths", ""] + [f"- {s}" for s in p.strengths] + [""]

    if p.drills:
        L += ["## Prescribed drills", ""]
        L += [f"- **{d['name']}** ({d['for']}, {d['duration']}) — {d['instruction']}"
              for d in p.drills]
        L += [""]

    L += ["## Next session", ""] + [f"- {n}" for n in p.next_session]
    return "\n".join(L)


def to_html(r: AnalysisResult) -> str:
    a, p = r.assessment, r.plan

    def esc(s):
        return html.escape(str(s))

    rows = "".join(
        f"<tr><td>{esc(CRITERION_LABELS[c])}</td>"
        f"<td class='n'>{s:.1f}</td>"
        f"<td><div class='track'><div class='fill' style='width:{s * 10:.0f}%'></div></div></td></tr>"
        for c, s in sorted(a.scores.items(), key=lambda kv: -kv[1])
    )
    focus = "".join(
        f"<li><b>{esc(f['label'])} — {f['score']}/10</b><br>{esc(f['issue'])}"
        f"<br><span class='muted'>{esc(f['feature'])}: measured {f['measured']:.4g} · target {f['target']:.4g}</span></li>"
        for f in p.focus_areas
    )
    drills = "".join(
        f"<li><b>{esc(d['name'])}</b> <span class='muted'>({esc(d['for'])} · {esc(d['duration'])})</span>"
        f"<br>{esc(d['instruction'])}</li>" for d in p.drills
    )
    warn = ("<div class='warn'><b>Recording quality</b><ul>"
            + "".join(f"<li>{esc(w)}</li>" for w in a.quality_warnings)
            + "</ul></div>") if a.quality_warnings else ""

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Runway Report - {esc(r.video_id)}</title>
<style>
  :root {{ color-scheme: light dark; --fg:#16181d; --bg:#fff; --muted:#6b7280;
           --line:#e5e7eb; --accent:#7c3aed; --warnbg:#fff7ed; --warnfg:#9a3412; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e8e8ed; --bg:#14151a; --muted:#9ca3af; --line:#2a2c34;
             --accent:#a78bfa; --warnbg:#2a1f14; --warnfg:#fdba74; }} }}
  body {{ font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif; color:var(--fg);
          background:var(--bg); margin:0; padding:2rem 1rem; }}
  main {{ max-width:56rem; margin:0 auto; }}
  h1 {{ font-size:1.5rem; margin:0 0 .25rem; }}
  .score {{ font-size:3rem; font-weight:700; color:var(--accent); line-height:1; }}
  .grade {{ color:var(--muted); }}
  table {{ width:100%; border-collapse:collapse; margin:1rem 0; }}
  td, th {{ padding:.5rem .4rem; border-bottom:1px solid var(--line); text-align:left; }}
  td.n {{ text-align:right; font-variant-numeric:tabular-nums; width:3.5rem; }}
  .track {{ background:var(--line); border-radius:99px; height:.55rem; width:100%; }}
  .fill {{ background:var(--accent); border-radius:99px; height:100%; }}
  .muted {{ color:var(--muted); font-size:.875rem; }}
  .warn {{ background:var(--warnbg); color:var(--warnfg); padding:.75rem 1rem;
           border-radius:.5rem; margin:1rem 0; }}
  li {{ margin:.6rem 0; }}
  .wrap {{ overflow-x:auto; }}
</style></head><body><main>
<h1>Runway Performance Report</h1>
<p class="muted">{esc(r.video_id)} · scored by {esc(a.source)}</p>
<div class="score">{a.overall:.1f}<span style="font-size:1rem">/100</span></div>
<p class="grade">{esc(a.grade)}</p>
{warn}
<h2>Criterion scores</h2>
<div class="wrap"><table>{rows}</table></div>
<h2>{esc(p.headline)}</h2>
<p>{esc(p.summary)}</p>
<h2>Focus areas</h2><ol>{focus or '<li>No criterion outside tolerance.</li>'}</ol>
<h2>Strengths</h2><ul>{''.join(f'<li>{esc(s)}</li>' for s in p.strengths) or '<li>—</li>'}</ul>
<h2>Prescribed drills</h2><ul>{drills or '<li>—</li>'}</ul>
<h2>Next session</h2><ul>{''.join(f'<li>{esc(n)}</li>' for n in p.next_session)}</ul>
</main></body></html>"""


def save(r: AnalysisResult, out_dir: Path, formats=("txt", "md", "html", "json")) -> list[Path]:
    """Write the report in the requested formats; returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "txt": (to_text, ".txt"),
        "md": (to_markdown, ".md"),
        "html": (to_html, ".html"),
        "json": (lambda x: json.dumps(x.as_dict(), indent=2), ".json"),
    }
    paths = []
    for fmt in formats:
        if fmt not in writers:
            continue
        fn, ext = writers[fmt]
        path = out_dir / f"{r.video_id}_report{ext}"
        path.write_text(fn(r), encoding="utf-8")
        paths.append(path)
    return paths
