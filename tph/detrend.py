"""Light curve cleaning: binning, outlier rejection, and removal of stellar variability."""
import numpy as np
import pandas as pd

from . import config


def bin_lightcurve(t, f, e, minutes=config.BIN_MINUTES):
    """Bin to a fixed cadence. Speeds up the search ~10x with no loss for transits > 1 hr."""
    width = minutes / 1440.0
    idx = np.floor((t - t.min()) / width).astype(np.int64)
    df = pd.DataFrame({"i": idx, "t": t, "f": f, "w": 1.0 / np.maximum(e, 1e-9) ** 2})
    g = df.groupby("i")
    n = g.size().to_numpy()
    tb = g["t"].mean().to_numpy()
    fb = (df.f * df.w).groupby(df.i).sum().to_numpy() / g["w"].sum().to_numpy()
    eb = 1.0 / np.sqrt(g["w"].sum().to_numpy())
    keep = n >= 1
    return tb[keep], fb[keep], eb[keep]


def _running_median(t, f, window_days, mask=None):
    """Time-windowed centred running median (handles gaps correctly)."""
    s = pd.Series(np.where(mask, np.nan, f) if mask is not None else f,
                  index=pd.to_datetime(t, unit="D"))
    trend = s.rolling(pd.Timedelta(days=window_days), center=True, min_periods=5).median()
    trend = np.array(trend.to_numpy(), dtype=float)   # writable copy
    bad = ~np.isfinite(trend)
    if bad.any() and (~bad).any():
        trend[bad] = np.interp(t[bad], t[~bad], trend[~bad])
    return trend


def robust_std(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def flatten(t, f, e, window_days=config.DETREND_WINDOW_DAYS, n_iter=3):
    """Divide out a running-median trend, iteratively ignoring outliers so that
    transits do not drag the trend down. Returns flattened flux and the trend."""
    order = np.argsort(t)
    t, f, e = t[order], f[order], e[order]
    mask = np.zeros_like(f, dtype=bool)
    for _ in range(n_iter):
        trend = _running_median(t, f, window_days, mask)
        r = f / trend - 1.0
        sig = robust_std(r[~mask])
        mask = np.abs(r) > 3.0 * sig
    flat = f / trend
    return t, flat, e / trend, trend


def clean(t, f, e):
    """Remove NaNs and upward outliers (flares, cosmic rays). Downward points are kept:
    they might be transits."""
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
    t, f, e = t[ok], f[ok], e[ok]
    med = np.median(f)
    sig = robust_std(f - med)
    if not np.isfinite(sig) or sig <= 0:
        return t, f, e
    keep = (f - med) < 4.0 * sig
    return t[keep], f[keep], e[keep]


def trim_edges(t, f, e, gap_days=0.5, trim_days=0.3):
    """Drop data right next to gaps and sector edges, where scattered light and
    thermal settling after data downlinks create fake dips."""
    o = np.argsort(t)
    t, f, e = t[o], f[o], e[o]
    starts = np.r_[t[0], t[1:][np.diff(t) > gap_days]]
    ends = np.r_[t[:-1][np.diff(t) > gap_days], t[-1]]
    bad = np.zeros_like(t, dtype=bool)
    for a in starts:
        bad |= (t >= a) & (t < a + trim_days)
    for b in ends:
        bad |= (t <= b) & (t > b - trim_days)
    return t[~bad], f[~bad], e[~bad]


def prepare(t, f, e, bin_minutes=config.BIN_MINUTES):
    """Full preprocessing chain used by both the hunter and the simulator."""
    t, f, e = clean(np.asarray(t, float), np.asarray(f, float), np.asarray(e, float))
    if t.size < 100:
        z = np.array([])
        return z, z, z, z, z
    if t.size > 100:
        t, f, e = trim_edges(t, f, e)
    m = np.median(f)
    f, e = f / m, e / m
    cadence = np.median(np.diff(np.sort(t))) * 1440 if t.size > 1 else bin_minutes
    if cadence < bin_minutes * 0.9:
        t, f, e = bin_lightcurve(t, f, e, bin_minutes)
    t, flat, ef, trend = flatten(t, f, e)
    # Fix flux uncertainties to the empirical scatter (pipeline errors are often wrong).
    scatter = robust_std(flat - 1.0)
    if np.isfinite(scatter) and scatter > 0:
        ef = np.full_like(flat, scatter)
    raw = flat * trend
    return t, flat, ef, raw, trend
