"""Builds the data behind the public site (docs/data.json, docs/results.html) and a
markdown summary (results/SUMMARY.md) from the accumulated CSVs."""
import datetime as dt
import json

import numpy as np
import pandas as pd

from . import config

from .postprocess import ORDER


def _load(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def stats():
    s, c = _load(config.SCREENED_CSV), _load(config.CANDIDATES_CSV)
    analyzed = s[~s.status.isin(["no_tess_data", "download_error", "analysis_error",
                                 "worker_error", "too_little_data"])] if len(s) else s
    n_sectors = int(analyzed.sectors.astype(str).str.split().str.len().sum()) if len(analyzed) else 0
    counts = c.status.value_counts().to_dict() if len(c) else {}
    val = json.loads(config.VALIDATION_JSON.read_text()) if config.VALIDATION_JSON.exists() else {}
    met = json.loads((config.MODELS / "metrics.json").read_text()) if (config.MODELS / "metrics.json").exists() else {}
    return {
        "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "stars_queried": int(len(s)), "stars_analyzed": int(len(analyzed)),
        "sectors_analyzed": n_sectors,
        "light_curve_points": int(analyzed.n_points.fillna(0).sum()) if len(analyzed) else 0,
        "signals": int(len(c)), "counts": {k: int(v) for k, v in counts.items()},
        "validation": val, "model": met,
    }


def build():
    config.DOCS.mkdir(exist_ok=True)
    from . import postprocess
    postprocess.apply()
    st = stats()
    c = _load(config.CANDIDATES_CSV)
    if len(c):
        c["rank"] = c.status.map({k: i for i, k in enumerate(ORDER)}).fillna(99)
        c = c.sort_values(["rank", "score"], ascending=[True, False])
        cols = ["id", "tic", "status", "score", "period", "depth_ppm", "rp_earth", "duration_hr",
                "sde", "snr", "n_transits", "flags", "note", "tmag", "teff", "rstar", "ra", "dec",
                "sectors", "plot", "date", "t0_btjd"]
        rows = json.loads(c[[k for k in cols if k in c]].replace({np.nan: None}).to_json(orient="records"))
    else:
        rows = []
    payload = {"stats": st, "rows": rows}
    html = TEMPLATE.replace("/*DATA*/", json.dumps(payload))
    # index.html is the hand-built public site (it reads data.json);
    # results.html is the full sortable table of every signal.
    (config.DOCS / "results.html").write_text(html)
    (config.DOCS / "data.json").write_text(json.dumps(payload, separators=(",", ":")))
    (config.DOCS / ".nojekyll").write_text("")
    _sky()
    from . import replay
    replay.prune_and_index()
    _summary(st, c)
    return st


def _sky():
    """Every star the bot has looked at, for the all-sky coverage map."""
    s = _load(config.SCREENED_CSV)
    out = {"ra": [], "dec": [], "s": [], "tic": []}
    if len(s) and {"ra", "dec"}.issubset(s.columns):
        s = s.dropna(subset=["ra", "dec"])
        out = {"ra": s.ra.round(3).tolist(), "dec": s.dec.round(3).tolist(),
               "s": s.status.astype(str).tolist(), "tic": s.tic.astype(int).tolist()}
    (config.DOCS / "sky.json").write_text(json.dumps(out, separators=(",", ":")))


def _summary(st, c):
    lines = ["# Results summary", "", f"_Last updated {st['updated']} by the hunting bot._", "",
             f"- Stars analyzed: **{st['stars_analyzed']:,}**",
             f"- TESS sectors processed: **{st['sectors_analyzed']:,}**",
             f"- Light curve points searched: **{st['light_curve_points']:,}**",
             f"- Signals passing detection: **{st['signals']:,}**", ""]
    for k in ORDER:
        if st["counts"].get(k):
            lines.append(f"  - {k.replace('_', ' ')}: {st['counts'][k]}")
    v = st.get("validation") or {}
    if v.get("recovery_rate") is not None:
        lines += ["", f"**Validation:** blind search recovered {v['known_planets_recovered']} of "
                      f"{v['known_planets_tested']} confirmed TESS planets "
                      f"({100 * v['recovery_rate']:.0f}%)."]
    top = c[c.status.isin(["strong_candidate", "candidate"])].head(25) if len(c) else c
    if len(top):
        lines += ["", "## Top new candidates", "",
                  "| ID | Score | Period (d) | Depth (ppm) | Rp (R⊕) | SNR | Flags | Plot |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in top.iterrows():
            plot = f"[view](../docs/plots/{r.plot})" if isinstance(r.plot, str) and r.plot else ""
            lines.append(f"| {r.id} | {r.score:.2f} | {r.period:.4f} | {r.depth_ppm:.0f} | "
                         f"{r.rp_earth:.1f} | {r.snr:.1f} | {r.flags if isinstance(r.flags, str) else ''} | {plot} |")
    config.RESULTS.mkdir(exist_ok=True)
    (config.RESULTS / "SUMMARY.md").write_text("\n".join(lines) + "\n")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TESS Planet Hunter</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#070b14;--panel:#0e1524;--line:#1d2840;--ink:#e8edf6;--mute:#8b97ad;--accent:#ff7a45;--good:#3ddc97;--warn:#ffc857;--bad:#ff5d73;--blue:#5aa9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif}
body:before{content:"";position:fixed;inset:0;background:radial-gradient(1px 1px at 20% 30%,#fff8,transparent),radial-gradient(1px 1px at 70% 80%,#fff6,transparent),radial-gradient(1px 1px at 40% 70%,#fff5,transparent),radial-gradient(1.5px 1.5px at 85% 20%,#fff7,transparent),radial-gradient(1px 1px at 10% 90%,#fff5,transparent);pointer-events:none;z-index:-1}
.wrap{max-width:1180px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:clamp(28px,5vw,44px);margin:0 0 6px;letter-spacing:-.02em}
.sub{color:var(--mute);max-width:760px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:28px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}
.big{font-size:28px;font-weight:700;font-family:"JetBrains Mono",monospace}
.lbl{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
h2{margin:40px 0 12px;font-size:20px}
.cands{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.cand img{width:100%;border-radius:8px;background:#fff;cursor:zoom-in}
.pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:600}
.s-strong_candidate{background:#3ddc9722;color:var(--good)}.s-candidate{background:#ffc85722;color:var(--warn)}
.s-likely_eb,.s-low_score{background:#ff5d7322;color:var(--bad)}.s-systematic{background:#b48cff22;color:#c9a8ff}
.s-known_planet,.s-known_toi,.s-known_system,.s-known_eb,.s-blend_of_known{background:#5aa9ff22;color:var(--blue)}
.mono{font-family:"JetBrains Mono",monospace;font-size:13px}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mute);font-weight:500;cursor:pointer;position:sticky;top:0;background:var(--panel)}
tr:hover td{background:#ffffff06}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
select,input{background:var(--panel);border:1px solid var(--line);color:var(--ink);border-radius:8px;padding:7px 10px;font:inherit}
a{color:var(--blue)}
.how{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.how .n{color:var(--accent);font-weight:700;font-family:"JetBrains Mono",monospace}
.modal{position:fixed;inset:0;background:#000c;display:none;align-items:center;justify-content:center;padding:16px;z-index:9}
.modal img{max-width:100%;max-height:92vh;border-radius:8px;background:#fff}
footer{margin-top:50px;color:var(--mute);font-size:13px}
</style>
</head>
<body>
<div class="wrap">
<div class="lbl" style="color:var(--accent)">Autonomous exoplanet search · NASA TESS</div>
<h1>TESS Planet Hunter</h1>
<a href="index.html" style="display:inline-block;margin:4px 0 14px;color:var(--blue)">Back to Tess</a>
<p class="sub">A bot that wakes up every few hours, pulls light curves for stars NASA's TESS satellite observed but never examined closely, searches them for the tiny periodic dimming of a transiting planet, and ranks every signal with a machine-learning vetter. Everything below was found automatically.</p>
<div class="grid" id="kpis"></div>

<h2>How it works</h2>
<div class="how">
<div class="card"><div class="n">01</div><b>Select</b><div class="sub">Random sky fields; FGK/M dwarfs, Tmag 9–13, low contamination, from the TESS Input Catalog.</div></div>
<div class="card"><div class="n">02</div><b>Clean</b><div class="sub">Download SPOC/QLP light curves from MAST, remove flares, divide out stellar variability.</div></div>
<div class="card"><div class="n">03</div><b>Search</b><div class="sub">Box Least Squares over 0.5–15 day periods, iterative to catch multi-planet systems.</div></div>
<div class="card"><div class="n">04</div><b>Vet</b><div class="sub">Odd/even depth, secondary eclipse, shape, radius, duration physics, then an AI score.</div></div>
<div class="card"><div class="n">05</div><b>Cross-match</b><div class="sub">Against every TOI, confirmed planet and TESS eclipsing binary to isolate what is new.</div></div>
</div>

<h2>Top new candidates</h2>
<div class="cands" id="cands"></div>

<h2>All signals</h2>
<div class="controls">
<select id="fstatus"><option value="">All statuses</option></select>
<input id="fsearch" placeholder="Search TIC…">
</div>
<div class="tbl"><table id="tbl"><thead></thead><tbody></tbody></table></div>

<h2>Model &amp; validation</h2>
<div class="grid" id="val"></div>

<footer>Candidates are unconfirmed. A real planet needs follow-up (ground photometry, spectroscopy, high-resolution imaging). Data: NASA TESS via MAST, ExoFOP-TESS, NASA Exoplanet Archive, Prša et al. 2022 TESS EB catalog. <span id="upd"></span></footer>
</div>
<div class="modal" id="modal" onclick="this.style.display='none'"><img id="mimg" alt="Diagnostic plot"></div>
<script>
const D=/*DATA*/;
const S=D.stats, R=D.rows;
const fmt=n=>n==null?"–":Number(n).toLocaleString();
const kp=[["Stars analyzed",S.stars_analyzed],["TESS sectors",S.sectors_analyzed],["Data points searched",S.light_curve_points],["Signals detected",S.signals],["New candidates",(S.counts.strong_candidate||0)+(S.counts.candidate||0)],["Known planets recovered",(S.counts.known_planet||0)+(S.counts.known_toi||0)]];
document.getElementById("kpis").innerHTML=kp.map(([l,v])=>`<div class="card"><div class="big">${fmt(v)}</div><div class="lbl">${l}</div></div>`).join("");
document.getElementById("upd").textContent="Updated "+S.updated+".";
const pill=s=>`<span class="pill s-${s}">${(s||"").replace(/_/g," ")}</span>`;
const topC=R.filter(r=>["strong_candidate","candidate"].includes(r.status)&&r.plot).slice(0,12);
document.getElementById("cands").innerHTML=topC.length?topC.map(r=>`<div class="card cand"><img loading="lazy" src="plots/${r.plot}" alt="Diagnostic plot for ${r.id}" onclick="zoom('plots/${r.plot}')"><div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px"><b class="mono">${r.id}</b>${pill(r.status)}</div><div class="mono" style="color:var(--mute);margin-top:4px">P ${r.period.toFixed(4)} d · ${Math.round(r.depth_ppm)} ppm · ${r.rp_earth.toFixed(1)} R⊕ · score ${r.score.toFixed(2)}</div><div style="margin-top:6px;font-size:13px"><a href="https://exofop.ipac.caltech.edu/tess/target.php?id=${r.tic}" target="_blank" rel="noopener">ExoFOP</a></div></div>`).join(""):`<div class="card sub">No candidates yet. The bot is still screening.</div>`;
function zoom(s){document.getElementById("mimg").src=s;document.getElementById("modal").style.display="flex"}
const cols=[["id","ID"],["status","Status"],["score","AI score"],["period","Period (d)"],["depth_ppm","Depth (ppm)"],["rp_earth","Rp (R⊕)"],["snr","SNR"],["n_transits","Transits"],["flags","Flags"],["note","Note"],["tmag","Tmag"]];
let sortK="score",sortD=-1;
const st=document.getElementById("fstatus");[...new Set(R.map(r=>r.status))].forEach(s=>st.insertAdjacentHTML("beforeend",`<option>${s}</option>`));
document.querySelector("#tbl thead").innerHTML="<tr>"+cols.map(([k,l])=>`<th data-k="${k}">${l}</th>`).join("")+"</tr>";
document.querySelectorAll("#tbl th").forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortD=k===sortK?-sortD:-1;sortK=k;draw()});
st.onchange=draw;document.getElementById("fsearch").oninput=draw;
function cell(r,k){const v=r[k];if(k==="status")return pill(v);if(k==="id")return r.plot?`<a href="#" onclick="zoom('plots/${r.plot}');return false" class="mono">${v}</a>`:`<span class="mono">${v}</span>`;if(typeof v==="number")return k==="period"?v.toFixed(4):k==="score"?v.toFixed(2):Number.isInteger(v)?v:v.toFixed(1);return v??""}
function draw(){const fs=st.value,q=document.getElementById("fsearch").value.trim();let rows=R.filter(r=>(!fs||r.status===fs)&&(!q||String(r.tic).includes(q)));rows.sort((a,b)=>((a[sortK]??-1e9)>(b[sortK]??-1e9)?1:-1)*sortD);document.querySelector("#tbl tbody").innerHTML=rows.slice(0,500).map(r=>"<tr>"+cols.map(([k])=>`<td>${cell(r,k)}</td>`).join("")+"</tr>").join("")}
draw();
const V=S.validation||{},M=S.model||{};const vv=[];
if(V.recovery_rate!=null)vv.push(["Confirmed planets recovered blind",`${V.known_planets_recovered}/${V.known_planets_tested}`]);
if(M.roc_auc!=null)vv.push(["Vetter ROC AUC",M.roc_auc]);
if(M.real_only)vv.push(["Real TOIs: planets kept",Math.round(100*M.real_only["planets_scored_above_0.5"])+"%"],["Real TOIs: false positives rejected",Math.round(100*M.real_only.false_positives_rejected)+"%"]);
if(M.n_train)vv.push(["Training examples",M.n_train]);
if(M.source)vv.push(["Trained on",M.source]);
document.getElementById("val").innerHTML=vv.length?vv.map(([l,v])=>`<div class="card"><div class="big" style="font-size:22px">${typeof v==="number"?fmt(v):v}</div><div class="lbl">${l}</div></div>`).join(""):`<div class="card sub">Validation pending.</div>`;
</script>
</body>
</html>
"""
