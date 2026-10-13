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
        print(f"    ({fn.__name__ if hasattr(fn, '__name__') else 'query'} failed: {exc})")
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
