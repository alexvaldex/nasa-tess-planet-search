"""One-page diagnostic figure per candidate (the kind astronomers use to vet by eye)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#1b1f24"
ACCENT = "#e4572e"
MUTED = "#9aa4ae"


def _fold(t, period, t0):
    return ((t - t0 + 0.5 * period) % period) - 0.5 * period


def _binned(x, y, nbins):
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    idx = np.digitize(x, edges) - 1
    xb, yb = [], []
    for i in range(nbins):
        m = idx == i
        if m.sum() >= 3:
            xb.append(x[m].mean()); yb.append(np.median(y[m]))
    return np.array(xb), np.array(yb)


def diagnostic(path, star, t, flat, raw, trend, sig, feats, extras, score, status, flags, note=""):
    P, t0, dur = sig.period, sig.t0, sig.duration
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.25], hspace=0.55, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, raw, ".", ms=1.2, color=MUTED)
    ax.plot(t, trend, "-", lw=0.8, color=INK)
    ax.set_title("Raw TESS light curve with stellar-variability trend", loc="left", fontsize=10)
    ax.set_ylabel("Rel. flux")

    ax = fig.add_subplot(gs[1, :])
    ax.plot(t, flat, ".", ms=1.2, color=MUTED)
    n0, n1 = np.ceil((t.min() - t0) / P), np.floor((t.max() - t0) / P)
    for n in np.arange(n0, n1 + 1):
        ax.axvline(t0 + n * P, color=ACCENT, lw=0.6, alpha=0.6)
    ax.set_title("Detrended light curve, predicted transits marked", loc="left", fontsize=10)
    ax.set_xlabel("Time (BTJD = BJD - 2457000)"); ax.set_ylabel("Rel. flux")

    ax = fig.add_subplot(gs[2, 0])
    if sig.periods is not None:
        ax.plot(sig.periods, sig.power, lw=0.5, color=INK)
        ax.axvline(P, color=ACCENT, lw=1, alpha=0.5)
        ax.set_xscale("log")
    ax.set_title(f"BLS periodogram (SDE {sig.sde:.1f})", loc="left", fontsize=10)
    ax.set_xlabel("Period (days)"); ax.set_ylabel("SDE")

    ax = fig.add_subplot(gs[2, 1])
    ph = _fold(t, P, t0) * 24
    w = max(3 * dur * 24, 3)
    m = np.abs(ph) < w
    ax.plot(ph[m], flat[m], ".", ms=2, color=MUTED)
    if m.sum() > 20:
        xb, yb = _binned(ph[m], flat[m], 40)
        ax.plot(xb, yb, "o", ms=3.5, color=ACCENT)
    ax.set_title(f"Phase-folded transit (depth {extras['depth_ppm']:.0f} ppm)", loc="left", fontsize=10)
    ax.set_xlabel("Hours from mid-transit"); ax.set_ylabel("Rel. flux")

    ax = fig.add_subplot(gs[2, 2])
    ph2 = _fold(t, 2 * P, t0) * 24
    for off, col, lab in ((0, ACCENT, "odd"), (P * 24, INK, "even")):
        x = ph2 - off
        x = (x + P * 24) % (2 * P * 24) - P * 24
        mm = np.abs(x) < w
        if mm.sum() > 20:
            xb, yb = _binned(x[mm], flat[mm], 30)
            ax.plot(xb, yb, "o-", ms=3, lw=0.8, color=col, label=lab)
    phs = _fold(t, P, t0 + P / 2) * 24
    ms_ = np.abs(phs) < w
    if ms_.sum() > 20:
        xb, yb = _binned(phs[ms_], flat[ms_], 30)
        ax.plot(xb, yb, "s", ms=2.5, color="#3a86ff", label="phase 0.5")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"Odd/even ({feats['odd_even_sigma']:.1f}σ) & secondary ({feats['secondary_snr']:.1f}σ)",
                 loc="left", fontsize=10)
    ax.set_xlabel("Hours from event")

    flag_txt = ", ".join(flags) if flags else "none"
    fig.suptitle(
        f"TIC {star['tic']}   |   P = {P:.4f} d   |   Rp ≈ {extras['rp_earth']:.1f} $R_\\oplus$   |   "
        f"AI planet score {score:.2f}   |   {status.replace('_', ' ')}",
        x=0.06, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.06, 0.935,
             f"Tmag {star.get('tmag', np.nan):.1f} · Teff {star.get('teff', np.nan):.0f} K · "
             f"R* {star.get('rstar', np.nan):.2f} $R_\\odot$ · sectors {star.get('sectors', '')} · "
             f"{extras['n_transits']} transits · duration {extras['duration_hr']:.1f} h · "
             f"SNR {sig.snr:.1f} · flags: {flag_txt}" + (f" · {note}" if note else ""),
             fontsize=9, color="#555")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
