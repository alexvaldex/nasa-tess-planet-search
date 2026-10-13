"""Vetting: turn a detected signal into physically meaningful diagnostics.

These are the same tests professional TESS vetting uses to separate real planets from
impostors (eclipsing binaries, blended background stars, stellar variability, systematics).
"""
import numpy as np

from . import config

R_SUN_AU = 0.00465047
R_SUN_EARTH = 109.1

FEATURES = [
    "log_sde", "log_snr", "log_depth_ppm", "log_dur_ratio", "log_q",
    "odd_even_sigma", "secondary_snr", "half_period_snr", "shape",
    "log_n_transits", "single_dominance", "log_rp", "harmonic_dll",
    "depth_consistency", "centroid_sigma",
]


def depth_consistency(t, f, e, sig):
    """Reduced chi-square of individual transit depths around the mean depth.
    A planet makes the same dip every time (~1); binaries blended with variable
    stars, and spacecraft glitches, make dips that change from event to event."""
    P, t0, dur = sig.period, sig.t0, sig.duration
    n = np.round((t - t0) / P)
    phase = t - (t0 + n * P)
    inn = np.abs(phase) < dur / 2
    depths, errs = [], []
    for k in np.unique(n[inn]):
        m = inn & (n == k)
        if m.sum() >= 3:
            depths.append(1 - np.mean(f[m]))
            errs.append(np.median(e[m]) / np.sqrt(m.sum()))
    if len(depths) < 2:
        return np.nan
    d, s = np.array(depths), np.maximum(np.array(errs), 1e-7)
    w = 1 / s ** 2
    mean = np.sum(w * d) / np.sum(w)
    chi2 = np.sum(((d - mean) / s) ** 2) / (len(d) - 1)
    return float(np.log10(max(chi2, 1e-3)))


def centroid_sigma(lc, sig):
    """Does the star's image shift on the detector during transit? If the light
    centre moves, the dip is happening on a nearby star, not this one.
    Uses SPOC centroid columns; returns NaN when they aren't available (QLP)."""
    if lc is None or "cx" not in lc:
        return np.nan
    t, cx, cy, sec = (np.asarray(lc[k], float) for k in ("t", "cx", "cy", "sec"))
    ok = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(t)
    if ok.sum() < 200:
        return np.nan
    t, cx, cy, sec = t[ok], cx[ok], cy[ok], sec[ok]
    P, t0, dur = sig.period, sig.t0, sig.duration
    ph = ((t - t0 + 0.5 * P) % P) - 0.5 * P
    inn = np.abs(ph) < dur / 2
    near = (np.abs(ph) > dur) & (np.abs(ph) < 3 * dur)
    if inn.sum() < 10 or near.sum() < 20:
        return np.nan
    dx, dy, vx, vy = [], [], [], []
    for s_ in np.unique(sec):
        a, b = inn & (sec == s_), near & (sec == s_)
        if a.sum() < 5 or b.sum() < 10:
            continue
        for arr, out, var in ((cx, dx, vx), (cy, dy, vy)):
            out.append(np.mean(arr[a]) - np.mean(arr[b]))
            var.append(np.var(arr[b]) / a.sum() + np.var(arr[b]) / b.sum())
    if not dx:
        return np.nan
    w = lambda v: 1 / np.maximum(np.array(v), 1e-12)
    mx = np.sum(w(vx) * dx) / np.sum(w(vx)); my = np.sum(w(vy) * dy) / np.sum(w(vy))
    ex = 1 / np.sqrt(np.sum(w(vx))); ey = 1 / np.sqrt(np.sum(w(vy)))
    return float(min(np.hypot(mx / ex, my / ey), 50.0))


def wave_ratio(series, depth):
    """Peak-to-peak swing of a binned out-of-transit series relative to the dip depth,
    counted only if the swing is *smooth* (neighbouring bins agree, lag-1 autocorrelation
    above 0.5). Random noise gives a big spread but no smoothness, so it doesn't count."""
    v = np.asarray(series, float)
    v = v[np.isfinite(v)]
    if v.size < 6 or depth <= 0:
        return np.nan
    d = v - v.mean()
    den = np.sum(d * d)
    ac = np.sum(d[1:] * d[:-1]) / den if den > 0 else 0.0
    if ac < 0.5:
        return 0.0
    return float((np.percentile(v, 95) - np.percentile(v, 5)) / depth)


def oot_variability(t, f, period, t0, duration, nbins=40):
    """How much does the star's brightness swing *outside* the transit, relative to the
    dip itself? Planets: the folded light curve is flat apart from the dip (~0).
    Pulsating or spotted stars and contact binaries: a smooth wave as large as the 'dip'."""
    ph = ((t - t0) / period) % 1.0                    # transit at 0 and 1, so out-of-transit is contiguous
    half_w = max(duration / period, 1.0 / nbins)
    edges = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, nbins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.array([np.median(f[idx == i]) if (idx == i).sum() >= 3 else np.nan for i in range(nbins)])
    out = (centers > 1.5 * half_w) & (centers < 1 - 1.5 * half_w)
    dist = np.minimum(ph, 1 - ph)
    inn = dist < 0.5 * duration / period
    if np.sum(np.isfinite(med[out])) < 6 or inn.sum() < 3:
        return np.nan
    return wave_ratio(med[out], 1 - np.median(f[inn]))


def expected_duration(period, rstar=1.0, mstar=1.0):
    """Central-transit duration (days) for a circular orbit around the given star."""
    a_au = (mstar * (period / 365.25) ** 2) ** (1.0 / 3.0)
    x = min(1.0, rstar * R_SUN_AU / a_au)
    return period / np.pi * np.arcsin(x)


def phase_depth(t, f, period, t0, duration, offset=0.0):
    """Depth (and significance) of the dip at a given orbital phase, measured directly:
    median flux just outside the window minus median flux inside, error from scatter.
    offset=0.5 tests for a secondary eclipse, the classic eclipsing-binary giveaway."""
    ph = (((t - t0) / period - offset + 0.5) % 1.0) - 0.5
    dt = np.abs(ph * period)
    inn = dt < duration / 2
    out = (dt > duration) & (dt < 3 * duration)
    if inn.sum() < 5 or out.sum() < 10:
        return np.nan, np.nan
    ref = np.median(f[out])
    dep = ref - np.median(f[inn])
    err = 1.4826 * np.median(np.abs(f[out] - ref)) / np.sqrt(inn.sum()) * 1.2533   # error of a median
    return float(dep), float(dep / err) if err > 0 else np.nan


def max_secondary(t, f, period, t0, duration, nphase=200):
    """Look for a second dip *anywhere* in the orbit, not just halfway round. Eccentric
    eclipsing binaries put their second eclipse at other phases (0.62 for TIC 267889592)."""
    ph = ((t - t0) / period) % 1.0
    w = max(duration / period, 1.0 / nphase)
    away = np.minimum(ph, 1 - ph) > 1.5 * w                    # skip the primary dip
    if away.sum() < 50:
        return np.nan, np.nan, np.nan
    tt, ff = ph[away], f[away]
    base = np.median(ff)
    noise = 1.4826 * np.median(np.abs(ff - base)) * 1.2533
    best = (0.0, 0.0, np.nan)
    for c in np.linspace(0, 1, nphase, endpoint=False):
        d = np.abs(((tt - c + 0.5) % 1.0) - 0.5)
        inn = d < w / 2
        if inn.sum() < 5:
            continue
        dep = base - np.median(ff[inn])
        snr = dep / (noise / np.sqrt(inn.sum())) if noise > 0 else 0.0
        if snr > best[1]:
            best = (float(dep), float(snr), float(c))
    return best


def _sigma(v):
    val, err = v
    return float(val / err) if err > 0 and np.isfinite(err) else 0.0


def compute_features(t, f, e, sig, rstar=1.0, mstar=1.0, lc=None):
    st = sig.stats
    P, t0, dur = sig.period, sig.t0, sig.duration
    depth = max(sig.depth, 1e-7)

    d_odd, e_odd = st["depth_odd"]
    d_even, e_even = st["depth_even"]
    oe = abs(d_odd - d_even) / np.sqrt(e_odd ** 2 + e_even ** 2) if (e_odd > 0 and e_even > 0) else 0.0

    # Secondary eclipse: measured directly (astropy's depth_phased can come back with a zero
    # error, which silently hid a real 1.2% secondary in testing).
    _, sec = phase_depth(t, f, P, t0, dur, offset=0.5)
    if not np.isfinite(sec):
        sec = _sigma(st["depth_phased"])
    half = _sigma(st["depth_half"])      # dip at half period -> period may be wrong

    # Transit shape: planets are flat-bottomed (U), grazing binaries are V-shaped.
    phase = ((t - t0 + 0.5 * P) % P) - 0.5 * P
    inner = np.abs(phase) < dur / 4
    outer = (np.abs(phase) >= dur / 4) & (np.abs(phase) < dur / 2)
    if inner.sum() >= 3 and outer.sum() >= 3:
        d_in = 1 - np.median(f[inner])
        d_out = 1 - np.median(f[outer])
        shape = float(np.clip(d_out / d_in, -1, 2)) if d_in > 0 else 0.0
    else:
        shape = 0.0

    counts = np.asarray(st["per_transit_count"])
    ll = np.asarray(st["per_transit_log_likelihood"])
    n_tr = int((counts > 0).sum())
    pos = ll[ll > 0]
    dominance = float(pos.max() / pos.sum()) if pos.size else 1.0

    hdll = float(st["harmonic_delta_log_likelihood"])
    total_ll = float(np.sum(pos)) if pos.size else 1.0
    harm = float(np.sign(hdll) * np.log10(1 + abs(hdll) / max(total_ll, 1e-9)))

    # Robust depth: median of in-transit points. If the box-average depth is far deeper,
    # a handful of bad points (not a real transit) are creating the signal.
    in_tr = np.abs(phase) < dur / 2
    robust_depth = float(1 - np.median(f[in_tr])) if in_tr.sum() >= 5 else depth
    rp = np.sqrt(max(robust_depth, 1e-7)) * rstar * R_SUN_EARTH
    dur_ratio = dur / max(expected_duration(P, rstar, mstar), 1e-4)

    feats = {
        "log_sde": np.log10(max(sig.sde, 0.1)),
        "log_snr": np.log10(max(sig.snr, 0.1)),
        "log_depth_ppm": np.log10(depth * 1e6),
        "log_dur_ratio": np.log10(dur_ratio),
        "log_q": np.log10(dur / P),
        "odd_even_sigma": float(np.clip(oe, 0, 30)),
        "secondary_snr": float(np.clip(sec, -10, 30)),
        "half_period_snr": float(np.clip(half, -10, 30)),
        "shape": shape,
        "log_n_transits": np.log10(max(n_tr, 1)),
        "single_dominance": dominance,
        "log_rp": np.log10(rp),
        "harmonic_dll": harm,
        "depth_consistency": depth_consistency(t, f, e, sig),
        "centroid_sigma": centroid_sigma(lc, sig),
    }
    sec_any_depth, sec_any_snr, sec_any_phase = max_secondary(t, f, P, t0, dur)
    extras = {"secondary_any_snr": sec_any_snr, "secondary_any_phase": sec_any_phase,
              "secondary_any_ratio": float(sec_any_depth / max(robust_depth, 1e-7)) if np.isfinite(sec_any_depth) else np.nan,
              "rp_earth": float(rp), "n_transits": n_tr, "depth_ppm": max(robust_depth, 0) * 1e6,
              "box_depth_ppm": depth * 1e6, "duration_hr": dur * 24,
              "outlier_ratio": float(depth / max(robust_depth, sig.depth_err, 1e-7)),
              "oot_ratio": oot_variability(t, f, P, t0, dur), "dur_ratio": float(dur_ratio)}
    return feats, extras


def rule_flags(feats, extras):
    """Hard physical red flags, reported next to the ML score."""
    flags = []
    if extras["rp_earth"] > config.MAX_PLANET_RADIUS_EARTH:
        flags.append("too_large")
    if feats["odd_even_sigma"] > 3:
        flags.append("odd_even")
    if feats["secondary_snr"] > 3:
        flags.append("secondary")
    elif (np.isfinite(extras.get("secondary_any_snr", np.nan)) and extras["secondary_any_snr"] > 6
          and extras.get("secondary_any_ratio", 0) > 0.1):
        flags.append("secondary")          # eccentric binary: second eclipse away from phase 0.5
    if extras["n_transits"] < config.MIN_TRANSITS:
        flags.append("few_transits")
    if feats["single_dominance"] > 0.6:
        flags.append("single_event")
    if feats["shape"] < 0.35:
        flags.append("v_shaped")
    if extras.get("outlier_ratio", 1) > 3:
        flags.append("outlier_driven")
    if np.isfinite(feats.get("centroid_sigma", np.nan)) and feats["centroid_sigma"] > 4:
        flags.append("centroid_shift")
    oot = extras.get("oot_ratio", np.nan)
    if (np.isfinite(oot) and oot > 0.5) or feats["secondary_snr"] < -4:
        flags.append("variable_star")       # smooth brightness wave, not a transit
    if extras.get("dur_ratio", 1.0) < 0.33:
        flags.append("too_short")           # far shorter than this star allows: grazing binary or noise
    return flags


def passes_detection(sig, extras):
    return (sig.sde >= config.MIN_SDE and sig.snr >= config.MIN_SNR
            and extras["n_transits"] >= config.MIN_TRANSITS and sig.depth > 0)
