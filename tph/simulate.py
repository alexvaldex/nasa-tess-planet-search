"""Realistic synthetic TESS light curves for training and testing the vetter.

Injects limb-darkened planet transits, eclipsing binaries (EBs), diluted background
EBs, and instrumental systematics into noisy, variable, gappy TESS-like time series.
"""
import numpy as np

from .vet import R_SUN_AU, R_SUN_EARTH

CLASSES = ["planet", "eb", "beb", "systematic"]


def overlap_fraction(z, k):
    """Fraction of a unit-radius star covered by a disk of radius k at separation z."""
    z = np.asarray(z, float)
    out = np.zeros_like(z)
    full = z <= 1 - k
    out[full] = k ** 2
    part = (z > abs(1 - k)) & (z < 1 + k)
    zp = z[part]
    a1 = k ** 2 * np.arccos(np.clip((zp ** 2 + k ** 2 - 1) / (2 * zp * k), -1, 1))
    a2 = np.arccos(np.clip((zp ** 2 + 1 - k ** 2) / (2 * zp), -1, 1))
    a3 = 0.5 * np.sqrt(np.clip((-zp + k + 1) * (zp + k - 1) * (zp - k + 1) * (zp + k + 1), 0, None))
    out[part] = (a1 + a2 - a3) / np.pi
    if k > 1:
        out[z <= k - 1] = 1.0
    return out


def limb_dark(r, u1=0.4, u2=0.25):
    mu = np.sqrt(np.clip(1 - np.clip(r, 0, 1) ** 2, 0, 1))
    return (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2) / (1 - u1 / 3 - u2 / 6)


def eclipse_signal(t, period, t0, k, a_rs, b, secondary_ratio=0.0):
    """Flux drop for a body of radius ratio k on a circular orbit.
    secondary_ratio > 0 adds an occultation at phase 0.5 (EBs)."""
    inc = np.arccos(np.clip(b / a_rs, 0, 1))
    phi = 2 * np.pi * (t - t0) / period
    z = a_rs * np.sqrt(np.sin(phi) ** 2 + (np.cos(inc) * np.cos(phi)) ** 2)
    front = np.cos(phi) > 0
    drop = np.zeros_like(t)
    frac = overlap_fraction(z, k)
    r_eff = np.clip(z, 0, 1)
    drop[front] = frac[front] * limb_dark(r_eff[front])
    if secondary_ratio > 0:
        back = ~front
        drop[back] = secondary_ratio * frac[back] / max(k ** 2, 1e-9) * k ** 2
    return drop


def time_grid(rng, n_sectors=None, cadence_min=10.0):
    n_sectors = n_sectors or rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
    ts = []
    start = 0.0
    for _ in range(n_sectors):
        t = np.arange(start, start + 27.4, cadence_min / 1440)
        mid = start + 13.7
        keep = np.abs(t - mid) > 0.5                         # orbit downlink gap
        for _ in range(rng.integers(0, 3)):                  # scattered-light gaps
            g0 = rng.uniform(start, start + 27)
            keep &= ~((t > g0) & (t < g0 + rng.uniform(0.2, 1.5)))
        ts.append(t[keep])
        start += 27.4   # production uses a contiguous block of sectors
    return np.concatenate(ts)


def noise_and_variability(rng, t):
    sigma = 10 ** rng.uniform(np.log10(300e-6), np.log10(4000e-6))
    flux = 1 + rng.normal(0, sigma, t.size)
    # correlated (red) noise
    walk = np.cumsum(rng.normal(0, 1, t.size))
    k = 36
    red = np.convolve(walk - walk.mean(), np.ones(k) / k, mode="same")
    red = red / (np.std(red) + 1e-12) * sigma * rng.uniform(0, 0.6)
    flux += red
    # stellar rotation / pulsation
    for _ in range(rng.integers(0, 3)):
        p = 10 ** rng.uniform(np.log10(0.8), np.log10(20))
        flux += rng.uniform(0, 5e-3) * np.sin(2 * np.pi * t / p + rng.uniform(0, 6.3))
    return flux, sigma


def random_star(rng):
    r = rng.uniform(0.5, 1.5)
    m = np.clip(r ** 1.25 * rng.normal(1, 0.08), 0.3, 1.6)
    return r, m


def a_over_rs(period, rstar, mstar):
    a_au = (mstar * (period / 365.25) ** 2) ** (1 / 3)
    return a_au / (rstar * R_SUN_AU)


def simulate(rng, kind):
    t = time_grid(rng)
    flux, sigma = noise_and_variability(rng, t)
    rstar, mstar = random_star(rng)
    truth = {"kind": kind, "rstar": rstar, "mstar": mstar, "sigma": sigma}
    if kind == "systematic":
        for _ in range(rng.integers(0, 4)):
            tc = rng.choice(t)
            w = rng.uniform(0.05, 0.4)
            flux -= rng.uniform(1, 6) * sigma * np.exp(-0.5 * ((t - tc) / w) ** 2)
        truth["period"] = np.nan
    else:
        P = 10 ** rng.uniform(np.log10(0.6), np.log10(14))
        t0 = rng.uniform(t.min(), t.min() + P)
        ars = a_over_rs(P, rstar, mstar)
        if kind == "planet":
            rp = 10 ** rng.uniform(np.log10(1.0), np.log10(20))
            k = rp / (rstar * R_SUN_EARTH)
            b = rng.uniform(0, 0.95)
            drop = eclipse_signal(t, P, t0, k, ars, b)
            truth.update(rp=rp)
        else:
            k = rng.uniform(0.3, 1.0)
            b = rng.uniform(0, 1.0 + k * 0.8)            # often grazing -> V-shaped
            ars = max(ars * rng.uniform(0.6, 1.2), 1.5 + k)
            ratio = rng.uniform(0.05, 0.9)               # secondary/primary surface brightness
            drop = eclipse_signal(t, P, t0, min(k, 0.99), ars, b, secondary_ratio=ratio * k ** 2)
            drop /= 1 + ratio * k ** 2
            if kind == "beb":
                drop *= 10 ** rng.uniform(-2.3, -0.7)    # diluted by a brighter foreground star
        flux = flux * (1 - drop)
        truth["in_transit"] = drop > 1e-6
        truth.update(period=P, t0=t0)
    err = np.full_like(t, sigma)
    return t, flux, err, truth
