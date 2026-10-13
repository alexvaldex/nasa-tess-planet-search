"""Population-level vetting that needs many stars at once.

The most important one: a period that shows up in several *unrelated* stars observed
in the same TESS sectors is the spacecraft, not a planet (momentum dumps, the 13.7-day
orbit, scattered light). Real surveys use the same trick.
"""
import json

import numpy as np
import pandas as pd

from . import config

ORDER = ["strong_candidate", "candidate", "known_planet", "known_toi", "blend_of_known",
         "known_system", "known_eb", "likely_eb", "variable_star", "systematic", "low_score"]
REVISABLE = {"strong_candidate", "candidate", "likely_eb", "low_score", "systematic", "variable_star"}
TOL = 0.004          # fractional period match
MIN_OTHERS = 2       # this many other stars sharing the period -> systematic


def _shares(p, others):
    if not len(others):
        return 0
    r = others / p
    return int(np.sum((np.abs(r - 1) < TOL) | (np.abs(r - 2) < 2 * TOL) | (np.abs(r - 0.5) < TOL / 2)))


def veto_common_periods(c):
    """Return a copy of the candidates table with shared-period signals marked."""
    if c.empty or "period" not in c:
        return c, set()
    c = c.copy()
    c["flags"] = c["flags"].fillna("").astype(str)
    changed = set()
    for _, g in c.groupby(c["sectors"].astype(str)):
        if g.tic.nunique() < MIN_OTHERS + 1:
            continue
        for i, row in g.iterrows():
            others = g[g.tic != row.tic].drop_duplicates("tic").period.to_numpy(float)
            if _shares(row.period, others) >= MIN_OTHERS:
                if "common_period" not in row["flags"].split():
                    c.at[i, "flags"] = (row["flags"] + " common_period").strip()
                if row["status"] in REVISABLE and row["status"] != "systematic":
                    c.at[i, "status"] = "systematic"
                    changed.add(int(row.tic))
    return c, changed


def apply():
    """Re-vet the accumulated results in place (candidates, screened, replay files)."""
    if not config.CANDIDATES_CSV.exists():
        return
    c = pd.read_csv(config.CANDIDATES_CSV)
    c, changed = veto_common_periods(c)
    c.to_csv(config.CANDIDATES_CSV, index=False)
    if not changed:
        return
    rank = {k: i for i, k in enumerate(ORDER)}
    if config.SCREENED_CSV.exists():
        s = pd.read_csv(config.SCREENED_CSV)
        best = c[c.tic.isin(changed)].groupby("tic").status.agg(lambda v: min(v, key=lambda x: rank.get(x, 99)))
        m = s.tic.isin(best.index)
        s.loc[m, "status"] = s.loc[m, "tic"].map(best)
        s.to_csv(config.SCREENED_CSV, index=False)
    from .replay import STARS_DIR
    for tic in changed:
        p = STARS_DIR / f"{tic}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows = c[c.tic == tic]
        for sig in d.get("signals", []):
            hit = rows[np.isclose(rows.period, sig["period"], rtol=1e-5)]
            if len(hit) and hit.iloc[0].status == "systematic":
                sig["status"] = "systematic"
                sig["flags"] = sorted(set(sig.get("flags", [])) | {"common_period"})
                sig["checks"].append({"name": "Unique to this star", "pass": False,
                                      "value": f"same period seen in {MIN_OTHERS}+ other stars this sector",
                                      "why": "A period shared by unrelated stars is the spacecraft, not a planet."})
        d["verdict"] = min((r for r in rows.status), key=lambda x: rank.get(x, 99)) if len(rows) else d["verdict"]
        p.write_text(json.dumps(d, separators=(",", ":")))


def revet_from_replays():
    """Apply checks added after a star was screened, using the light curve saved in its
    replay file. Only touches signals currently ranked as candidates."""
    from .replay import STARS_DIR
    from .vet import oot_variability, wave_ratio
    if not config.CANDIDATES_CSV.exists():
        return {}
    c = pd.read_csv(config.CANDIDATES_CSV)
    c["flags"] = c["flags"].fillna("").astype(str)
    changes = {}
    for i, row in c[c.status.isin(["strong_candidate", "candidate"])].iterrows():
        p = STARS_DIR / f"{int(row.tic)}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        sig = next((s for s in d.get("signals", []) if abs(s["period"] - row.period) < 1e-4), None)
        if sig is None:
            continue
        t, fl = np.array(d["lc"]["t"], float), np.array(d["lc"]["flat"], float)
        ok = np.isfinite(fl)
        P, dur = row.period, row.duration_hr / 24
        t0 = row.t0_btjd - d["t_start_btjd"]
        swing_ratio = oot_variability(t[ok], fl[ok], P, t0, max(dur, 2.5 * np.median(np.diff(t))))
        fx, fy = np.array(sig["fold"]["x"], float), np.array(sig["fold"]["y"], float)
        inn = np.abs(fx) < row.duration_hr / 2
        td = 1 - np.median(fy[inn]) if inn.sum() >= 5 else np.nan
        if np.isfinite(swing_ratio) and np.isfinite(td) and td > 0:
            # swing measured on binned data, so compare it with the dip at the same binning
            ph = ((t[ok] - t0) / P) % 1
            near = np.minimum(ph, 1 - ph) < max(dur / P, 0.5 * np.median(np.diff(t)) / P)
            bd = 1 - np.median(fl[ok][near]) if near.any() else np.nan
            if np.isfinite(bd) and bd > 0:
                swing_ratio = swing_ratio * bd / td
        # second view at full time resolution: the saved fold around the transit
        bx, by = np.array(sig["fold"]["bx"], float), np.array(sig["fold"]["by"], float)
        left, right = by[bx < -row.duration_hr], by[bx > row.duration_hr]
        if np.isfinite(td) and td > 0:
            fr = [wave_ratio(side, td) for side in (left, right) if side.size >= 6]
            fr = [x for x in fr if np.isfinite(x)]
            if len(fr) == 2:   # a real wave shows on both sides of the dip
                swing_ratio = np.nanmax([swing_ratio, min(fr)]) if np.isfinite(swing_ratio) else min(fr)
        new_flags = set(row["flags"].split())
        if (np.isfinite(swing_ratio) and swing_ratio > 0.5) or row.get("f_secondary_snr", 0) < -4:
            new_flags.add("variable_star")
        if 10 ** row.get("f_log_dur_ratio", 0) < 0.33:
            new_flags.add("too_short")
        if "variable_star" in new_flags:
            status = "variable_star"
        elif row.status == "strong_candidate" and (row.snr < config.STRONG_SNR or new_flags
                                                   or 10 ** row.get("f_log_dur_ratio", 0) < 0.5):
            status = "candidate"
        else:
            status = row.status
        c.at[i, "flags"] = " ".join(sorted(new_flags))
        if status != row.status:
            changes[row["id"]] = (row.status, status, round(float(swing_ratio), 2) if np.isfinite(swing_ratio) else None)
            c.at[i, "status"] = status
            sig["status"] = status
            sig["flags"] = sorted(new_flags)
            if not any(ch["name"] == "Not a pulsating star" for ch in sig["checks"]):
                sig["checks"].append({"name": "Not a pulsating star", "pass": "variable_star" not in new_flags,
                                      "value": "smooth brightness wave as large as the dip" if "variable_star" in new_flags
                                      else "flat outside the dip",
                                      "why": "Pulsating and spotted stars make smooth waves that can look like dips."})
            rank = {k: j for j, k in enumerate(ORDER)}
            rows = c[c.tic == row.tic]
            d["verdict"] = min(rows.status, key=lambda x: rank.get(x, 99))
            p.write_text(json.dumps(d, separators=(",", ":")))
    c.to_csv(config.CANDIDATES_CSV, index=False)
    if changes and config.SCREENED_CSV.exists():
        rank = {k: j for j, k in enumerate(ORDER)}
        s = pd.read_csv(config.SCREENED_CSV)
        tics = {int(k.split(".")[0][3:]) for k in changes}
        best = c[c.tic.isin(tics)].groupby("tic").status.agg(lambda v: min(v, key=lambda x: rank.get(x, 99)))
        m = s.tic.isin(best.index)
        s.loc[m, "status"] = s.loc[m, "tic"].map(best)
        s.to_csv(config.SCREENED_CSV, index=False)
    return changes
