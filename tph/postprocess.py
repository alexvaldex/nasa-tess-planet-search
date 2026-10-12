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
         "known_system", "known_eb", "likely_eb", "systematic", "low_score"]
REVISABLE = {"strong_candidate", "candidate", "likely_eb", "low_score", "systematic"}
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
