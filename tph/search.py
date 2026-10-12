"""Transit search with Box Least Squares (Kovacs+ 2002), the standard algorithm for
finding periodic dips in a light curve."""
from dataclasses import dataclass, field

import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy.ndimage import median_filter

from . import config


@dataclass
class Signal:
    period: float
    t0: float
    duration: float
    depth: float
    depth_err: float
    sde: float
    snr: float
    stats: dict = field(repr=False, default_factory=dict)
    periods: np.ndarray = field(repr=False, default=None)
    power: np.ndarray = field(repr=False, default=None)
    keep: np.ndarray = field(repr=False, default=None)   # data used for this search


def period_grid(t, pmin=config.PERIOD_MIN, pmax=config.PERIOD_MAX,
                oversample=3, max_n=config.MAX_PERIODS):
    """Uniform-in-frequency grid fine enough that a transit never drifts by more than
    1/oversample of its duration across the full baseline. The binding constraint is
    at the longest period, using a conservative (short, grazing) duration there."""
    from .vet import expected_duration
    baseline = t.max() - t.min()
    pmax = min(pmax, baseline / config.MIN_TRANSITS)
    if pmax <= pmin:
        return np.array([])
    d_short = 0.5 * expected_duration(pmax, rstar=0.6, mstar=0.6)
    df = d_short / (oversample * baseline * pmax)
    fmin, fmax = 1.0 / pmax, 1.0 / pmin
    n = min(int((fmax - fmin) / df) + 1, max_n)
    return np.sort(1.0 / np.linspace(fmin, fmax, n))


def bls_search(t, f, e, periods=None, durations=config.DURATIONS):
    if periods is None:
        periods = period_grid(t)
    if periods.size == 0:
        return None
    durations = [d for d in durations if d < periods.min()]
    model = BoxLeastSquares(t, f, dy=e)
    res = model.power(periods, durations, objective="likelihood")
    power = np.asarray(res.power)
    # Remove the slow rise of BLS power with period, then measure peak significance.
    win = max(51, (power.size // 200) | 1)
    detr = power - median_filter(power, size=win, mode="nearest")
    std = np.std(detr)
    if not np.isfinite(std) or std == 0:
        return None
    sde_all = (detr - np.mean(detr)) / std
    i = int(np.argmax(sde_all))
    period, t0, dur = res.period[i], res.transit_time[i], res.duration[i]
    stats = model.compute_stats(period, dur, t0)
    # BLS often locks onto half the true period. If every other "transit" is empty,
    # double the period and re-centre on the real events.
    (d_odd, e_odd), (d_even, e_even) = stats["depth_odd"], stats["depth_even"]
    deep, shallow, err = (d_odd, d_even, e_odd) if d_odd > d_even else (d_even, d_odd, e_even)
    if err > 0 and deep / err > 3 and shallow < 0.3 * deep and 2 * period <= periods.max() * 1.01:
        t0 = t0 if d_odd > d_even else t0 + period   # astropy: "odd" = events at t0 + 2nP
        period = 2 * period
        stats = model.compute_stats(period, dur, t0)
    depth, depth_err = stats["depth"]
    snr = depth / depth_err if depth_err > 0 else 0.0
    return Signal(period=float(period), t0=float(t0), duration=float(dur),
                  depth=float(depth), depth_err=float(depth_err),
                  sde=float(sde_all[i]), snr=float(snr), stats=stats,
                  periods=np.asarray(res.period), power=sde_all)


def transit_mask(t, period, t0, duration, factor=1.5):
    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    return np.abs(phase) < factor * duration / 2.0


def iterative_search(t, f, e, max_signals=config.MAX_SIGNALS_PER_STAR, return_first=False):
    """Find the strongest signal, mask it, and search again (catches multi-planet systems).
    With return_first=True also returns the first search result even if it is too weak to
    count as a signal (used to show the periodogram of every star in the replay)."""
    signals = []
    first = None
    keep = np.ones_like(t, dtype=bool)
    periods = period_grid(t)
    for i in range(max_signals):
        s = bls_search(t[keep], f[keep], e[keep], periods=periods)
        if i == 0:
            first = s
            if s is not None:
                s.keep = keep.copy()
        if s is None or s.sde < config.MIN_SDE * 0.8:
            break
        s.keep = keep.copy()
        signals.append(s)
        keep &= ~transit_mask(t, s.period, s.t0, s.duration)
        if keep.sum() < 200:
            break
    return (signals, first) if return_first else signals
