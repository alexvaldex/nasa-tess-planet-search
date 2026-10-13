"""Deep-dive data for a candidate star: catalog properties, distance, independent
variability checks, and the full-resolution light curve. Used to write candidate reports.

    python -m tph report 267889592 415954197
"""
import json
import warnings

import numpy as np

from . import config

warnings.filterwarnings("ignore")
OUT = config.RESULTS / "reports"


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as exc:
        import traceback
        print("    query failed:")
        traceback.print_exc()
        return default


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (bytes,)):
        return v.decode()
    if hasattr(v, "mask") and getattr(v, "mask", False) is True:
        return None
    return v if isinstance(v, (str, int, bool, type(None))) else str(v)


def tic_row(tic):
    from astroquery.mast import Catalogs
    t = Catalogs.query_criteria(catalog="Tic", ID=int(tic)).to_pandas()
    keep = ["ID", "ra", "dec", "pmRA", "pmDEC", "plx", "e_plx", "d", "e_d", "Tmag", "Vmag", "Gmag",
            "Jmag", "Teff", "e_Teff", "logg", "rad", "e_rad", "mass", "e_mass", "lum", "rho",
            "contratio", "GAIA", "TWOMASS", "lumclass", "MH", "ebv", "objType", "disposition",
            "numcont", "gallong", "gallat", "eclong", "eclat"]
    r = t.iloc[0]
    return {k: _clean(r[k]) for k in keep if k in t.columns}


def gaia(source_id):
    from astroquery.vizier import Vizier
    v = Vizier(columns=["Source", "Plx", "e_Plx", "RUWE", "Gmag", "BP-RP", "RV", "e_RV",
                        "Teff", "Dist", "NSS", "VarFlag"], row_limit=5)
    res = v.query_constraints(catalog="I/355/gaiadr3", Source=str(source_id))
    if not res:
        return {}
    r = res[0][0]
    return {c: _clean(r[c]) for c in res[0].colnames}


def cone(catalog, ra, dec, radius_arcsec, columns=("**",)):
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    v = Vizier(columns=list(columns), row_limit=20)
    res = v.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=radius_arcsec * u.arcsec, catalog=catalog)
    if not res:
        return []
    t = res[0]
    return [{c: _clean(row[c]) for c in t.colnames} for row in t]


def simbad(ra, dec, radius_arcsec=10):
    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    s = Simbad()
    s.add_votable_fields("otype", "ids")
    t = s.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=radius_arcsec * u.arcsec)
    if t is None:
        return []
    return [{c: _clean(row[c]) for c in t.colnames} for row in t]


def tess_coverage(tic):
    import lightkurve as lk
    sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
    rows = []
    for i in range(len(sr)):
        tb = sr.table
        rows.append({"mission": str(tb["mission"][i]), "author": str(tb["author"][i]),
                     "exptime": float(tb["exptime"][i]) if "exptime" in tb.colnames else None,
                     "year": int(tb["year"][i]) if "year" in tb.colnames else None})
    return rows


def full_lightcurve(tic, sectors):
    """Every requested sector at native cadence, detrended the same way as the hunt."""
    from .fetch import get_lightcurve
    from . import detrend
    lc = get_lightcurve(tic, max_sectors=len(sectors) + 4)
    if lc is None:
        return None
    t, f, e, raw, trend = detrend.prepare(lc["t"], lc["f"], lc["e"], bin_minutes=2.0)
    return {"t": t.round(5).tolist(), "flat": f.round(6).tolist(), "raw": raw.round(6).tolist(),
            "sectors": lc["sectors"], "authors": lc["authors"]}


def build(tic):
    import pandas as pd
    print(f"[report] TIC {tic}")
    c = pd.read_csv(config.CANDIDATES_CSV)
    cand = c[c.tic == int(tic)].sort_values("score", ascending=False)
    rec = {"tic": int(tic), "candidate": json.loads(cand.iloc[0].to_json()) if len(cand) else None}
    rec["tic_catalog"] = _safe(lambda: tic_row(tic), {})
    ra, dec = rec["tic_catalog"].get("ra"), rec["tic_catalog"].get("dec")
    if rec["tic_catalog"].get("GAIA"):
        rec["gaia_dr3"] = _safe(lambda: gaia(rec["tic_catalog"]["GAIA"]), {})
    if ra is not None:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        sc = SkyCoord(ra * u.deg, dec * u.deg)
        rec["constellation"] = sc.get_constellation()
        rec["gaia_eclipsing_binaries_5arcsec"] = _safe(lambda: cone("I/358/veb", ra, dec, 5), [])
        rec["gaia_variables_5arcsec"] = _safe(lambda: cone("I/358/vclassre", ra, dec, 5), [])
        rec["vsx_variables_60arcsec"] = _safe(lambda: cone("B/vsx/vsx", ra, dec, 60), [])
        rec["simbad_10arcsec"] = _safe(lambda: simbad(ra, dec, 10), [])
        rec["neighbours_tic_63arcsec"] = _safe(lambda: neighbours(ra, dec), [])
    rec["tess_coverage"] = _safe(lambda: tess_coverage(tic), [])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"TIC{tic}.json").write_text(json.dumps(rec, indent=1, default=str))
    if rec["candidate"]:
        ms = _safe(lambda: multi_sector(tic, rec["candidate"]["period"], rec["candidate"]["duration_hr"]))
        if ms:
            (OUT / f"TIC{tic}_allsectors.json").write_text(json.dumps(ms, separators=(",", ":")))
            print(f"  all sectors: P={ms['refined_period']:.5f} d, depth={ms['depth_ppm']:.0f} ppm, "
                  f"SNR={ms['snr']:.1f}, {ms['n_transits_with_data']} transits over {ms['baseline_days']:.0f} d, "
                  f"secondary {ms['secondary_depth_ppm']} ppm ({ms['secondary_snr']} sigma), odd/even {ms['odd_even_sigma']} sigma")
            for tr in ms["transits"]:
                print("   ", tr)
    lc = _safe(lambda: full_lightcurve(tic, rec["candidate"]["sectors"].split() if rec["candidate"] else []))
    if lc:
        (OUT / f"TIC{tic}_lightcurve.json").write_text(json.dumps(lc, separators=(",", ":")))
    print(json.dumps({k: v for k, v in rec.items() if k != "candidate"}, indent=1, default=str)[:4000])


def neighbours(ra, dec, radius_arcsec=63):
    """Stars within three TESS pixels that could leak light into the aperture."""
    from astroquery.mast import Catalogs
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    t = Catalogs.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=radius_arcsec * u.arcsec,
                              catalog="TIC").to_pandas()
    out = []
    for _, r in t.sort_values("Tmag").head(15).iterrows():
        out.append({"ID": int(r["ID"]), "Tmag": _clean(r["Tmag"]), "dstArcSec": _clean(r.get("dstArcSec")),
                    "Teff": _clean(r.get("Teff")), "d": _clean(r.get("d"))})
    return out


def all_sectors(tic):
    """Download every TESS sector of this star (one light curve per sector)."""
    import re, shutil, tempfile
    import lightkurve as lk
    sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
    tab = sr.table
    pref = {a: i for i, a in enumerate(config.AUTHOR_PREFERENCE)}
    best = {}
    for i in range(len(tab)):
        a = str(tab["author"][i])
        if a not in pref:
            continue
        m = re.search(r"Sector\s+(\d+)", str(tab["mission"][i]))
        sec = int(m.group(1)) if m else -1
        if sec >= 0 and (sec not in best or pref[a] < best[sec][0]):
            best[sec] = (pref[a], i)
    tmp = tempfile.mkdtemp()
    out = []
    try:
        for sec in sorted(best):
            i = best[sec][1]
            try:
                kw = {"flux_column": "kspsap_flux"} if str(tab["author"][i]) == "QLP" else {}
                lc = sr[i].download(download_dir=tmp, quality_bitmask="default", **kw).remove_nans()
            except Exception as exc:
                print(f"    sector {sec} ({tab['author'][i]}): download failed: {exc}")
                try:   # fall back to the default flux column
                    lc = sr[i].download(download_dir=tmp, quality_bitmask="default").remove_nans()
                except Exception as exc2:
                    print(f"    sector {sec}: fallback failed too: {exc2}")
                    continue
            t = np.asarray(lc.time.value, float); f = np.asarray(lc.flux.value, float)
            ok = np.isfinite(t) & np.isfinite(f)
            t, f = t[ok], f[ok]
            med = np.nanmedian(f)
            print(f"    sector {sec}: {len(t)} points from {tab['author'][i]}")
            if len(t) < 100 or not np.isfinite(med) or med <= 0:
                continue
            out.append({"sector": sec, "author": str(tab["author"][i]), "t": t, "f": f / med})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def multi_sector(tic, period, duration_hr):
    """Refit the signal using every sector TESS ever took of this star. If the same dip
    shows up at the predicted times years later, the period is real."""
    from astropy.timeseries import BoxLeastSquares
    from . import detrend
    secs = all_sectors(tic)
    T, F, E, S = [], [], [], []
    for s in secs:
        t, f, e, _, _ = detrend.prepare(s["t"], s["f"], np.full_like(s["f"], np.nanstd(s["f"])), bin_minutes=10.0)
        if t.size < 200:
            continue
        T.append(t); F.append(f); E.append(e); S.append(np.full_like(t, s["sector"]))
    if not T:
        return None
    t, f, e, sec = map(np.concatenate, (T, F, E, S))
    dur = duration_hr / 24
    model = BoxLeastSquares(t, f, dy=e)
    periods = np.linspace(period * 0.997, period * 1.003, 30000)
    res = model.power(periods, [dur * 0.75, dur, dur * 1.25], objective="snr")
    i = int(np.argmax(res.power))
    P, t0, d = float(res.period[i]), float(res.transit_time[i]), float(res.duration[i])
    st = model.compute_stats(P, d, t0)
    per_tr = []
    for tt in st["transit_times"]:
        m = np.abs(t - tt) < d / 2
        near = (np.abs(t - tt) > d) & (np.abs(t - tt) < 3 * d)
        if m.sum() >= 3 and near.sum() >= 6:
            dep = np.median(f[near]) - np.median(f[m])
            err = 1.4826 * np.median(np.abs(f[near] - np.median(f[near]))) / np.sqrt(m.sum())
            per_tr.append({"time_btjd": round(float(tt), 4), "sector": int(np.median(sec[m])),
                           "depth_ppm": round(float(dep * 1e6), 1), "snr": round(float(dep / err), 2) if err > 0 else None})
    depth, depth_err = st["depth"]
    from .vet import phase_depth
    sec_dep, sec_snr = phase_depth(t, f, P, t0, d, offset=0.5)
    odd, odd_s = phase_depth(t, f, 2 * P, t0, d, 0.0)
    even, even_s = phase_depth(t, f, 2 * P, t0 + P, d, 0.0)
    oe_sigma = (abs(odd - even) / np.hypot(odd / odd_s, even / even_s)
                if all(np.isfinite([odd, even, odd_s, even_s])) and odd_s and even_s else np.nan)
    years = sorted({int(2014 + (x + 2457000 - 2456658.5) / 365.25) for x in (t.min(), t.max())})
    return {"secondary_depth_ppm": sec_dep * 1e6 if np.isfinite(sec_dep) else None,
            "secondary_snr": sec_snr if np.isfinite(sec_snr) else None,
            "odd_depth_ppm": odd * 1e6 if np.isfinite(odd) else None,
            "even_depth_ppm": even * 1e6 if np.isfinite(even) else None,
            "odd_even_sigma": oe_sigma if np.isfinite(oe_sigma) else None,
            "refined_period": P, "t0_btjd": t0, "duration_hr": d * 24, "depth_ppm": depth * 1e6,
            "depth_err_ppm": depth_err * 1e6, "snr": float(depth / depth_err) if depth_err > 0 else None,
            "sectors": sorted({int(x) for x in sec}), "baseline_days": float(t.max() - t.min()),
            "n_transits_with_data": len(per_tr), "transits": per_tr,
            "lc": {"t": t.round(4).tolist(), "f": f.round(6).tolist(), "sector": sec.astype(int).tolist()}}
