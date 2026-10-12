"""Exports a compact, step-by-step record of how each star was processed, so the
Mission Control page can replay the bot's work visually (pick -> download -> clean ->
search -> fold -> check -> verdict)."""
import json

import numpy as np

from . import config

STARS_DIR = config.DOCS / "stars"
KEEP_STATUSES = {"strong_candidate", "candidate", "known_planet", "known_toi", "blend_of_known",
                 "known_eb", "likely_eb"}


def _r(x, nd):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in x]


def _downsample(t, *ys, n=1200):
    if t.size <= n:
        return (t,) + ys
    edges = np.linspace(0, t.size, n + 1).astype(int)
    out_t = np.array([t[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    outs = [np.array([np.median(y[a:b]) for a, b in zip(edges[:-1], edges[1:]) if b > a]) for y in ys]
    return (out_t, *outs)


def _periodogram(sig, n=420):
    if sig is None or sig.periods is None:
        return None
    p, s = np.asarray(sig.periods), np.asarray(sig.power)
    edges = np.logspace(np.log10(p.min()), np.log10(p.max()), n + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n - 1)
    px, sy = [], []
    for i in range(n):
        m = idx == i
        if m.any():
            j = np.argmax(np.where(m, s, -np.inf))
            px.append(p[j]); sy.append(s[j])
    return {"period": _r(px, 5), "sde": _r(sy, 2), "peak_period": round(sig.period, 5),
            "peak_sde": round(sig.sde, 2), "threshold": config.MIN_SDE, "n_periods": int(p.size)}


def _fold(t, f, sig, nbins=40, nscatter=500, seed=0):
    P, t0, dur = sig.period, sig.t0, sig.duration
    ph = (((t - t0 + 0.5 * P) % P) - 0.5 * P) * 24
    w = max(3 * dur * 24, 4.0)
    m = np.abs(ph) < w
    x, y = ph[m], f[m]
    edges = np.linspace(-w, w, nbins + 1)
    idx = np.digitize(x, edges) - 1
    bx, by = [], []
    for i in range(nbins):
        mm = idx == i
        if mm.sum() >= 2:
            bx.append(x[mm].mean()); by.append(np.median(y[mm]))
    rng = np.random.default_rng(seed)
    sel = rng.choice(x.size, min(nscatter, x.size), replace=False) if x.size else np.array([], int)
    return {"window_hr": round(w, 2), "x": _r(x[sel], 3), "y": _r(y[sel], 6),
            "bx": _r(bx, 3), "by": _r(by, 6), "duration_hr": round(dur * 24, 2)}


def check_record(sig, feats, extras, flags, score, status, note, known):
    """Human-readable checklist: the exact tests applied to this signal."""
    dr = 10 ** feats["log_dur_ratio"]
    items = [
        ("Strong enough to be real", f"SDE {sig.sde:.1f} (need {config.MIN_SDE:g}), SNR {sig.snr:.1f} (need {config.MIN_SNR:g})",
         sig.sde >= config.MIN_SDE and sig.snr >= config.MIN_SNR,
         "Is the dip clearly above the noise, or could random scatter make it?"),
        ("Repeats on schedule", f"{extras['n_transits']} transits seen (need {config.MIN_TRANSITS})",
         extras["n_transits"] >= config.MIN_TRANSITS,
         "A planet dims its star at a perfectly regular interval."),
        ("Not one glitch", f"largest single event = {100 * feats['single_dominance']:.0f}% of signal",
         feats["single_dominance"] <= 0.6,
         "A single bad data point or satellite momentum dump can fake a 'periodic' signal."),
        ("Odd = even depths", f"difference {feats['odd_even_sigma']:.1f}σ (fail above 3σ)",
         feats["odd_even_sigma"] <= 3,
         "Two stars eclipsing each other often make alternating deep/shallow dips."),
        ("No secondary eclipse", f"dip at half-phase {feats['secondary_snr']:.1f}σ (fail above 3σ)",
         feats["secondary_snr"] <= 3,
         "Planets are dark; a second dip halfway through the orbit means a glowing companion star."),
        ("Flat-bottomed shape", f"shape index {feats['shape']:.2f} (V-shaped below 0.35)",
         feats["shape"] >= 0.35,
         "A planet fully crossing the star gives a U-shaped dip; grazing binaries are V-shaped."),
        ("Planet-sized", f"{extras['rp_earth']:.1f} Earth radii (max {config.MAX_PLANET_RADIUS_EARTH:g})",
         extras["rp_earth"] <= config.MAX_PLANET_RADIUS_EARTH,
         "Anything much bigger than Jupiter is a star, not a planet."),
        ("Duration fits the star", f"{dr:.2f}x the expected transit length",
         0.25 <= dr <= 3.0,
         "Kepler's laws predict how long a transit should last for this star and period."),
    ]
    if known is not None:
        items.append(("Not already discovered",
                      note or f"no match among {known.summary()}",
                      status not in {"known_planet", "known_toi", "known_system", "known_eb", "blend_of_known"},
                      "Cross-matched by star ID and by sky position against every public TESS catalog."))
    return {"period": round(sig.period, 6), "t0": round(sig.t0, 5), "duration_hr": round(sig.duration * 24, 2),
            "depth_ppm": round(extras["depth_ppm"], 1), "rp_earth": round(extras["rp_earth"], 2),
            "sde": round(sig.sde, 2), "snr": round(sig.snr, 2),
            "score": None if not np.isfinite(score) else round(score, 3),
            "status": status, "flags": flags, "note": note,
            "checks": [{"name": a, "value": b, "pass": bool(c), "why": d} for a, b, c, d in items]}


def write(star, lc, t, raw, trend, flat, first, signals, checks, screened):
    STARS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = float(t.min())
    td, rawd, trd, fld = _downsample(t, raw, trend, flat)
    sigs = []
    for sig, chk in zip(signals, checks):
        keep = sig.keep if sig.keep is not None else np.ones_like(t, bool)
        n0, n1 = np.ceil((t.min() - sig.t0) / sig.period), np.floor((t.max() - sig.t0) / sig.period)
        chk = dict(chk, fold=_fold(t[keep], flat[keep], sig),
                   transits=_r(sig.t0 + np.arange(n0, n1 + 1) * sig.period - t0, 3))
        sigs.append(chk)
    rec = {
        "tic": int(star["tic"]), "date": screened.get("date"),
        "star": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else
                     (round(v, 4) if isinstance(v, float) else v))
                 for k, v in star.items() if k in ("tmag", "teff", "rstar", "mstar", "ra", "dec")},
        "sectors": [str(s) for s in lc["sectors"]], "authors": lc["authors"],
        "n_points": int(t.size), "t_start_btjd": round(t0, 4),
        "lc": {"t": _r(td - t0, 4), "raw": _r(rawd, 6), "trend": _r(trd, 6), "flat": _r(fld, 6)},
        "periodogram": _periodogram(first),
        "signals": sigs, "verdict": screened.get("status"),
    }
    (STARS_DIR / f"{int(star['tic'])}.json").write_text(json.dumps(rec, separators=(",", ":")))


def prune_and_index(max_recent=80):
    """Keep every interesting star plus the most recent ordinary ones; write an index."""
    if not STARS_DIR.exists():
        return []
    entries = []
    for p in STARS_DIR.glob("*.json"):
        if p.name in ("index.json", "demo.json"):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        best = d["signals"][0] if d.get("signals") else {}
        entries.append({"tic": d["tic"], "date": d.get("date"), "verdict": d.get("verdict"),
                        "mtime": p.stat().st_mtime, "period": best.get("period"),
                        "score": best.get("score"), "tmag": d["star"].get("tmag"),
                        "ra": d["star"].get("ra"), "dec": d["star"].get("dec")})
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    kept, n_plain = [], 0
    for e in entries:
        if e["verdict"] in KEEP_STATUSES:
            kept.append(e)
        elif n_plain < max_recent:
            kept.append(e); n_plain += 1
        else:
            (STARS_DIR / f"{e['tic']}.json").unlink(missing_ok=True)
    for e in kept:
        e.pop("mtime", None)
    (STARS_DIR / "index.json").write_text(json.dumps(kept, separators=(",", ":")))
    return kept
