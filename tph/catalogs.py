"""Catalog access: choosing target stars (TIC) and cross-matching against everything
already known (TESS Objects of Interest, confirmed planets, TESS eclipsing binaries)."""
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

DATA = config.ROOT / "data"
TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
PS_URL = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
          "select+pl_name,tic_id,pl_orbper,ra,dec+from+pscomppars+where+tic_id+is+not+null"
          "&format=csv")
TOI_FALLBACK_URL = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
                    "select+toi,tid,tfopwg_disp,pl_orbper,pl_tranmid,pl_trandurh,pl_trandep,ra,dec"
                    "+from+toi&format=csv")
NEAR_ARCSEC = 42.0   # two TESS pixels: signals this close may be the same blended source


def _cached_csv(name, url, max_age_hours=24, timeout=120):
    DATA.mkdir(exist_ok=True)
    path = DATA / name
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        return pd.read_csv(path, comment="#", low_memory=False)
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), comment="#", low_memory=False)
            df.to_csv(path, index=False)
            return df
        except Exception as exc:   # network hiccup: retry, then fall back to stale cache
            print(f"  [catalog] {name} attempt {attempt + 1} failed: {exc}")
            time.sleep(5 * (attempt + 1))
    if path.exists():
        return pd.read_csv(path, comment="#", low_memory=False)
    return None


def load_tois():
    """Return a normalized TOI table: tic, toi, disp, period, epoch, dur_hr, depth_ppm, ra, dec."""
    df = _cached_csv("tois.csv", TOI_URL)
    if df is not None and "TIC ID" in df.columns:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(df["RA"].astype(str), df["Dec"].astype(str), unit=(u.hourangle, u.deg))
        return pd.DataFrame({
            "tic": df["TIC ID"].astype(np.int64), "toi": df["TOI"].astype(str),
            "disp": df["TFOPWG Disposition"].fillna("").astype(str),
            "period": pd.to_numeric(df["Period (days)"], errors="coerce"),
            "epoch": pd.to_numeric(df["Epoch (BJD)"], errors="coerce") - 2457000.0,
            "dur_hr": pd.to_numeric(df["Duration (hours)"], errors="coerce"),
            "depth_ppm": pd.to_numeric(df["Depth (ppm)"], errors="coerce"),
            "ra": c.ra.deg, "dec": c.dec.deg})
    df = _cached_csv("tois_nea.csv", TOI_FALLBACK_URL)
    if df is None:
        return pd.DataFrame(columns=["tic", "toi", "disp", "period", "epoch", "dur_hr",
                                     "depth_ppm", "ra", "dec"])
    return pd.DataFrame({
        "tic": df["tid"].astype(np.int64), "toi": df["toi"].astype(str),
        "disp": df["tfopwg_disp"].fillna("").astype(str), "period": df["pl_orbper"],
        "epoch": df["pl_tranmid"] - 2457000.0, "dur_hr": df["pl_trandurh"],
        "depth_ppm": df["pl_trandep"], "ra": df["ra"], "dec": df["dec"]})


def load_confirmed():
    df = _cached_csv("confirmed.csv", PS_URL)
    if df is None:
        return pd.DataFrame(columns=["tic", "name", "period", "ra", "dec"])
    tic = pd.to_numeric(df["tic_id"].astype(str).str.replace("TIC", "").str.strip(), errors="coerce")
    out = pd.DataFrame({"tic": tic, "name": df["pl_name"], "period": df["pl_orbper"],
                        "ra": df["ra"], "dec": df["dec"]}).dropna(subset=["tic"])
    out["tic"] = out["tic"].astype(np.int64)
    return out


def load_tess_ebs():
    """TESS eclipsing binary catalog (Prsa et al. 2022, ApJS 258, 16) via VizieR."""
    path = DATA / "tess_ebs.csv"
    if path.exists():
        return set(pd.read_csv(path)["tic"].astype(np.int64))
    try:
        from astroquery.vizier import Vizier
        v = Vizier(columns=["TIC"], row_limit=-1)
        t = v.get_catalogs("J/ApJS/258/16")[0]
        tics = pd.DataFrame({"tic": np.asarray(t["TIC"]).astype(np.int64)})
        DATA.mkdir(exist_ok=True)
        tics.to_csv(path, index=False)
        return set(tics["tic"])
    except Exception as exc:
        print(f"  [catalog] TESS EB catalog unavailable: {exc}")
        return set()


def _harmonic_match(p1, p2, tol=0.01):
    if not (np.isfinite(p1) and np.isfinite(p2)) or p2 <= 0:
        return False
    for r in (1, 2, 0.5, 3, 1 / 3):
        if abs(p1 / (p2 * r) - 1) < tol:
            return True
    return False


class KnownObjects:
    """Everything already discovered, used to decide whether a detection is new."""

    def __init__(self):
        self.tois = load_tois()
        self.confirmed = load_confirmed()
        self.ebs = load_tess_ebs()
        self.toi_tics = set(self.tois["tic"])
        self.planet_tics = set(self.confirmed["tic"])
        pos = pd.concat([self.tois[["ra", "dec", "period"]].assign(name="TOI-" + self.tois["toi"]),
                         self.confirmed[["ra", "dec", "period", "name"]]], ignore_index=True)
        pos = pos.dropna(subset=["ra", "dec"])
        self._pos = pos.reset_index(drop=True)
        ra, dec = np.radians(pos["ra"].to_numpy(float)), np.radians(pos["dec"].to_numpy(float))
        self._xyz = np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
        print(f"  [catalog] {len(self.tois)} TOIs, {len(self.confirmed)} confirmed planets, "
              f"{len(self.ebs)} TESS EBs loaded")

    def classify(self, tic, ra, dec, period):
        """Return (status, note) for a detection: is it already known?"""
        tic = int(tic)
        if tic in self.planet_tics:
            row = self.confirmed[self.confirmed.tic == tic]
            for _, r in row.iterrows():
                if _harmonic_match(period, r.period):
                    return "known_planet", str(r["name"])
            return "known_system", "host of " + ", ".join(row["name"].astype(str))
        if tic in self.toi_tics:
            row = self.tois[self.tois.tic == tic]
            for _, r in row.iterrows():
                if _harmonic_match(period, r.period):
                    return "known_toi", f"TOI-{r.toi} ({r.disp or 'no disposition'})"
            return "known_system", "TOI host: " + ", ".join("TOI-" + row["toi"].astype(str))
        if tic in self.ebs:
            return "known_eb", "in TESS EB catalog"
        if np.isfinite(ra) and np.isfinite(dec) and len(self._xyz):
            r, d = np.radians(ra), np.radians(dec)
            v = np.array([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])
            sep = np.degrees(np.arccos(np.clip(self._xyz @ v, -1, 1))) * 3600
            for i in np.where(sep < NEAR_ARCSEC)[0]:
                if _harmonic_match(period, self._pos.period[i]):
                    return "blend_of_known", f"{self._pos.name[i]} is {sep[i]:.0f}\" away with same period"
        return "new", ""


def random_field(rng):
    """Uniform random sky position away from the crowded galactic plane."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    while True:
        ra = rng.uniform(0, 360)
        dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))
        c = SkyCoord(ra * u.deg, dec * u.deg)
        if abs(c.galactic.b.deg) > 12:
            return c


def select_targets(rng, n, exclude=frozenset(), max_fields=60):
    """Draw stars from the TESS Input Catalog that pass our quality cuts."""
    from astroquery.mast import Catalogs
    import astropy.units as u
    picked = []
    for _ in range(max_fields):
        c = random_field(rng)
        try:
            t = Catalogs.query_region(c, radius=config.FIELD_RADIUS_DEG * u.deg, catalog="TIC")
        except Exception as exc:
            print(f"  [targets] TIC query failed: {exc}")
            time.sleep(5)
            continue
        df = t.to_pandas()
        cols = {"ID", "Tmag", "Teff", "rad", "mass", "contratio", "ra", "dec"}
        if not cols.issubset(df.columns):
            continue
        m = (df.Tmag.between(*config.TMAG_RANGE) & df.Teff.between(*config.TEFF_RANGE)
             & df.rad.between(*config.RADIUS_RANGE)
             & (df.contratio.fillna(1.0) < config.MAX_CONTAMINATION))
        if "objType" in df:
            m &= df.objType.fillna("STAR") == "STAR"
        if "disposition" in df:
            m &= ~df.disposition.fillna("").isin(["SPLIT", "DUPLICATE", "ARTIFACT"])
        df = df[m].copy()
        df["tic"] = df.ID.astype(np.int64)
        df = df[~df.tic.isin(exclude)]
        for _, r in df.sample(frac=1, random_state=int(rng.integers(1 << 31))).iterrows():
            picked.append({"tic": int(r.tic), "tmag": float(r.Tmag), "teff": float(r.Teff),
                           "rstar": float(r.rad), "mstar": float(r["mass"]) if np.isfinite(r["mass"]) else float(r.rad),
                           "ra": float(r.ra), "dec": float(r.dec)})
            if len(picked) >= n:
                return picked
    return picked
