"""End-to-end processing of a single star: download -> clean -> search -> vet -> rank."""
import datetime as dt
import traceback

import numpy as np

from . import config, detrend, model, plots, replay, search, vet


def classify(score, flags, known_status):
    if known_status != "new":
        return known_status
    if {"single_event", "common_period", "outlier_driven"} & set(flags):
        return "systematic"          # instrument artefact, not astrophysical
    if {"too_large", "secondary", "odd_even"} & set(flags):
        return "likely_eb"
    if not np.isfinite(score):
        return "candidate"
    if score >= config.STRONG_SCORE and not flags:
        return "strong_candidate"
    if score >= config.CANDIDATE_SCORE:
        return "candidate"
    return "low_score"


PLOT_STATUSES = {"strong_candidate", "candidate", "known_planet", "known_toi"}


def analyze(star, lc, known=None, make_plots=True):
    """Run search + vetting on an already-downloaded light curve."""
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    t, flat, e, raw, trend = detrend.prepare(lc["t"], lc["f"], lc["e"])
    if t.size < 500:
        return {"tic": star["tic"], "date": now, "status": "too_little_data",
                "sectors": " ".join(map(str, lc["sectors"])), "ra": star.get("ra"),
                "dec": star.get("dec"), "tmag": star.get("tmag")}, []
    screened = {"tic": star["tic"], "date": now, "sectors": " ".join(map(str, lc["sectors"])),
                "authors": " ".join(lc["authors"]), "n_points": int(t.size),
                "ra": star.get("ra"), "dec": star.get("dec"), "tmag": star.get("tmag"),
                "best_period": np.nan, "best_sde": np.nan, "n_signals": 0, "status": "no_signal"}
    if t.size < 500:
        screened["status"] = "too_little_data"
        return screened, []
    signals, first = search.iterative_search(t, flat, e, return_first=True)
    cands = []
    checks = []   # per-signal vetting record for the visual replay
    for k, sig in enumerate(signals):
        keep = sig.keep if sig.keep is not None else np.ones_like(t, bool)
        feats, extras = vet.compute_features(t[keep], flat[keep], e[keep], sig,
                                             star.get("rstar", 1.0), star.get("mstar", 1.0))
        if k == 0:
            screened.update(best_period=round(sig.period, 6), best_sde=round(sig.sde, 2))
        if not vet.passes_detection(sig, extras):
            checks.append(replay.check_record(sig, feats, extras, [], float("nan"), "below_threshold", "", None))
            continue
        flags = vet.rule_flags(feats, extras)
        score = model.score(feats)
        known_status, note = ("new", "")
        if known is not None:
            known_status, note = known.classify(star["tic"], star.get("ra", np.nan),
                                                star.get("dec", np.nan), sig.period)
        status = classify(score, flags, known_status)
        cid = f"TIC{star['tic']}.{k + 1:02d}"
        row = {"id": cid, "tic": star["tic"], "signal": k + 1, "date": now, "status": status,
               "score": round(score, 4), "period": round(sig.period, 6), "t0_btjd": round(sig.t0, 5),
               "duration_hr": round(extras["duration_hr"], 3), "depth_ppm": round(extras["depth_ppm"], 1),
               "rp_earth": round(extras["rp_earth"], 2), "sde": round(sig.sde, 2),
               "snr": round(sig.snr, 2), "n_transits": extras["n_transits"],
               "flags": " ".join(flags), "note": note, "tmag": star.get("tmag"),
               "teff": star.get("teff"), "rstar": star.get("rstar"), "ra": star.get("ra"),
               "dec": star.get("dec"), "sectors": screened["sectors"], "plot": ""}
        row.update({f"f_{kk}": round(float(v), 4) for kk, v in feats.items()})
        if make_plots and status in PLOT_STATUSES:
            config.PLOTS.mkdir(parents=True, exist_ok=True)
            fn = f"{cid}.png"
            try:
                plots.diagnostic(config.PLOTS / fn, dict(star, sectors=screened["sectors"]),
                                 t[keep], flat[keep], raw[keep], trend[keep], sig, feats, extras,
                                 score, status, flags, note)
                row["plot"] = fn
            except Exception:
                traceback.print_exc()
        cands.append(row)
        checks.append(replay.check_record(sig, feats, extras, flags, score, status, note, known))
    screened["n_signals"] = len(cands)
    if cands:
        from .postprocess import ORDER as order
        screened["status"] = min((c["status"] for c in cands),
                                 key=lambda s: order.index(s) if s in order else 99)
    try:
        replay.write(star, lc, t, raw, trend, flat, first, signals, checks, screened)
    except Exception:
        traceback.print_exc()
    return screened, cands


def process_star(star, known=None):
    from .fetch import get_lightcurve
    try:
        lc = get_lightcurve(star["tic"])
    except Exception as exc:
        return {"tic": star["tic"], "status": "download_error", "note": str(exc)[:120]}, []
    if lc is None:
        return {"tic": star["tic"], "status": "no_tess_data"}, []
    try:
        return analyze(star, lc, known)
    except Exception as exc:
        traceback.print_exc()
        return {"tic": star["tic"], "status": "analysis_error", "note": str(exc)[:120]}, []
