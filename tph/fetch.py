"""Download TESS light curves from NASA's MAST archive via lightkurve."""
import re
import shutil
import tempfile
import warnings

import numpy as np

from . import config

warnings.filterwarnings("ignore")


def _sector(mission):
    m = re.search(r"Sector\s+(\d+)", str(mission))
    return int(m.group(1)) if m else -1


def _longest_consecutive_block(sectors, max_len):
    """Contiguous sectors keep the baseline short, which keeps the period search exact."""
    sectors = sorted(set(sectors))
    best, cur = [], []
    for s in sectors:
        cur = cur + [s] if cur and s == cur[-1] + 1 else [s]
        if len(cur) >= len(best):   # ties go to the most recent block
            best = list(cur)
    return best[-max_len:]


def get_lightcurve(tic, max_sectors=config.MAX_SECTORS):
    import lightkurve as lk
    sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
    if len(sr) == 0:
        return None
    tab = sr.table
    pref = {a: i for i, a in enumerate(config.AUTHOR_PREFERENCE)}
    best = {}
    for i in range(len(tab)):
        author = str(tab["author"][i])
        if author not in pref:
            continue
        sec = _sector(tab["mission"][i])
        exptime = float(tab["exptime"][i]) if "exptime" in tab.colnames else 0
        key = (pref[author], 0 if 100 <= exptime <= 200 else 1)   # prefer 2-min SPOC
        if sec >= 0 and (sec not in best or key < best[sec][0]):
            best[sec] = (key, i)
    if not best:
        return None
    sectors = _longest_consecutive_block(best.keys(), max_sectors)
    tmp = tempfile.mkdtemp(prefix="tph_")
    ts, fs, es, used, authors = [], [], [], [], []
    try:
        for sec in sectors:
            i = best[sec][1]
            try:
                lc = sr[i].download(download_dir=tmp, quality_bitmask="default")
            except Exception as exc:
                print(f"    TIC {tic} S{sec}: download failed ({exc})")
                continue
            if lc is None:
                continue
            lc = lc.remove_nans()
            if len(lc) < 100:
                continue
            t = np.asarray(lc.time.value, float)
            f = np.asarray(lc.flux.value, float)
            e = np.asarray(lc.flux_err.value, float) if lc.flux_err is not None else np.full_like(f, np.nan)
            med = np.nanmedian(f)
            if not np.isfinite(med) or med <= 0:
                continue
            f, e = f / med, e / med
            bad_e = ~np.isfinite(e) | (e <= 0)
            if bad_e.any():
                e[bad_e] = np.nanstd(f) if np.isfinite(np.nanstd(f)) else 1e-3
            ts.append(t); fs.append(f); es.append(e)
            used.append(sec); authors.append(str(tab["author"][i]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not ts:
        return None
    return {"t": np.concatenate(ts), "f": np.concatenate(fs), "e": np.concatenate(es),
            "sectors": used, "authors": sorted(set(authors))}
