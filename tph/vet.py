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
]


def expected_duration(period, rstar=1.0, mstar=1.0):
    """Central-transit duration (days) for a circular orbit around the given star."""
    a_au = (mstar * (period / 365.25) ** 2) ** (1.0 / 3.0)
    x = min(1.0, rstar * R_SUN_AU / a_au)
    return period / np.pi * np.arcsin(x)


def _sigma(v):
    val, err = v
    return float(val / err) if err > 0 and np.isfinite(err) else 0.0


def compute_features(t, f, e, sig, rstar=1.0, mstar=1.0):
    st = sig.stats
    P, t0, dur = sig.period, sig.t0, sig.duration
    depth = max(sig.depth, 1e-7)

    d_odd, e_odd = st["depth_odd"]
    d_even, e_even = st["depth_even"]
    oe = abs(d_odd - d_even) / np.sqrt(e_odd ** 2 + e_even ** 2) if (e_odd > 0 and e_even > 0) else 0.0

    sec = _sigma(st["depth_phased"])     # dip at phase 0.5 -> secondary eclipse
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
    }
    extras = {"rp_earth": float(rp), "n_transits": n_tr, "depth_ppm": max(robust_depth, 0) * 1e6,
              "box_depth_ppm": depth * 1e6, "duration_hr": dur * 24,
              "outlier_ratio": float(depth / max(robust_depth, sig.depth_err, 1e-7))}
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
    if extras["n_transits"] < config.MIN_TRANSITS:
        flags.append("few_transits")
    if feats["single_dominance"] > 0.6:
        flags.append("single_event")
    if feats["shape"] < 0.35:
        flags.append("v_shaped")
    if extras.get("outlier_ratio", 1) > 3:
        flags.append("outlier_driven")
    return flags


def passes_detection(sig, extras):
    return (sig.sde >= config.MIN_SDE and sig.snr >= config.MIN_SNR
            and extras["n_transits"] >= config.MIN_TRANSITS and sig.depth > 0)
