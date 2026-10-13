"""Offline tests: no network needed. Run with `pytest -q`."""
import numpy as np
import pandas as pd

from tph import catalogs, detrend, search, simulate, vet
from tph.pipeline import analyze


def _planet_lc(seed=1, rp=4.0, period=3.21, sigma=6e-4):
    rng = np.random.default_rng(seed)
    t = simulate.time_grid(rng, n_sectors=1)
    f = 1 + rng.normal(0, sigma, t.size)
    k = rp / 109.1
    f *= 1 - simulate.eclipse_signal(t, period, t.min() + 0.7, k, simulate.a_over_rs(period, 1, 1), 0.2)
    return t, f, np.full_like(t, sigma)


def test_recovers_injected_planet():
    t, f, e = _planet_lc()
    t, f, e, _, _ = detrend.prepare(t, f, e)
    s = search.bls_search(t, f, e)
    assert abs(s.period / 3.21 - 1) < 0.002
    feats, extras = vet.compute_features(t, f, e, s)
    assert vet.passes_detection(s, extras)
    assert 3.0 < extras["rp_earth"] < 5.0
    assert "odd_even" not in vet.rule_flags(feats, extras)


def test_eclipsing_binary_is_flagged():
    rng = np.random.default_rng(4)
    t = simulate.time_grid(rng, n_sectors=1)
    f = 1 + rng.normal(0, 5e-4, t.size)
    f *= 1 - simulate.eclipse_signal(t, 2.5, t.min() + 0.4, 0.7, 6.0, 0.3, secondary_ratio=0.25)
    t, f, e, _, _ = detrend.prepare(t, f, np.full_like(t, 5e-4))
    s = search.bls_search(t, f, e)
    feats, extras = vet.compute_features(t, f, e, s)
    flags = vet.rule_flags(feats, extras)
    assert {"too_large", "secondary", "odd_even"} & set(flags)


def test_analyze_end_to_end(tmp_path, monkeypatch):
    from tph import config
    monkeypatch.setattr(config, "PLOTS", tmp_path)
    from tph import replay
    monkeypatch.setattr(replay, "STARS_DIR", tmp_path)
    t, f, e = _planet_lc(seed=2)
    star = {"tic": 123, "rstar": 1.0, "mstar": 1.0, "ra": 10.0, "dec": -40.0, "tmag": 11, "teff": 5700}
    screened, cands = analyze(star, {"t": t, "f": f, "e": e, "sectors": [1], "authors": ["SPOC"]})
    assert cands and abs(cands[0]["period"] / 3.21 - 1) < 0.002


def test_known_object_matching(monkeypatch):
    tois = pd.DataFrame({"tic": [111], "toi": ["999.01"], "disp": ["PC"], "period": [3.0],
                         "epoch": [0], "dur_hr": [2], "depth_ppm": [1000], "ra": [10.0], "dec": [-40.0]})
    monkeypatch.setattr(catalogs, "load_tois", lambda: tois)
    monkeypatch.setattr(catalogs, "load_confirmed",
                        lambda: pd.DataFrame(columns=["tic", "name", "period", "ra", "dec"]))
    monkeypatch.setattr(catalogs, "load_tess_ebs", lambda: {222})
    monkeypatch.setattr(catalogs, "load_ctois",
                        lambda: pd.DataFrame(columns=["tic", "name", "period", "ra", "dec"]))
    k = catalogs.KnownObjects()
    assert k.classify(111, 10, -40, 6.0)[0] == "known_toi"          # 2x harmonic
    assert k.classify(222, 50, 50, 1.0)[0] == "known_eb"
    assert k.classify(333, 10.005, -40.0, 3.0)[0] == "blend_of_known"  # 18" away
    assert k.classify(444, 100, 20, 3.0)[0] == "new"


def test_common_period_veto():
    from tph import postprocess
    c = pd.DataFrame({"tic": [1, 2, 3, 4], "sectors": ["70 71"] * 4, "period": [8.386, 8.39, 8.383, 3.1],
                      "status": ["likely_eb", "candidate", "low_score", "strong_candidate"], "flags": ["", "", "", ""]})
    out, changed = postprocess.veto_common_periods(c)
    assert set(out.status[:3]) == {"systematic"} and out.status[3] == "strong_candidate"
    assert changed == {1, 2, 3}


def test_centroid_shift_detected():
    t, f, e = _planet_lc(seed=3)
    tt, ff, ee, _, _ = detrend.prepare(t, f, e)
    s = search.bls_search(tt, ff, ee)
    rng = np.random.default_rng(0)
    ph = ((t - s.t0 + 0.5 * s.period) % s.period) - 0.5 * s.period
    shift = np.where(np.abs(ph) < s.duration / 2, 0.02, 0.0)       # image moves 0.02 px in transit
    base = {"t": t, "sec": np.ones_like(t)}
    moved = dict(base, cx=100 + shift + rng.normal(0, 0.003, t.size), cy=50 + rng.normal(0, 0.003, t.size))
    still = dict(base, cx=100 + rng.normal(0, 0.003, t.size), cy=50 + rng.normal(0, 0.003, t.size))
    assert vet.centroid_sigma(moved, s) > 10
    assert vet.centroid_sigma(still, s) < 4
    feats, _ = vet.compute_features(tt, ff, ee, s)
    assert np.isfinite(feats["depth_consistency"]) and 10 ** feats["depth_consistency"] < 3


def test_secondary_anywhere_in_orbit():
    """Eccentric binary: second equal dip at phase 0.62 must be caught; a planet must not trip it."""
    rng = np.random.default_rng(5)
    t = simulate.time_grid(rng, n_sectors=2)
    f = 1 + rng.normal(0, 8e-4, t.size)
    P, t0 = 9.3, t.min() + 1.1
    for off in (0.0, 0.62):
        f *= 1 - simulate.eclipse_signal(t, P, t0 + off * P, 0.12, 20.0, 0.2)
    tt, ff, ee, _, _ = detrend.prepare(t, f, np.full_like(t, 8e-4))
    s = search.bls_search(tt, ff, ee)
    feats, extras = vet.compute_features(tt, ff, ee, s)
    assert "secondary" in vet.rule_flags(feats, extras)
    t2, f2, e2 = _planet_lc(seed=4)
    t2, f2, e2, _, _ = detrend.prepare(t2, f2, e2)
    s2 = search.bls_search(t2, f2, e2)
    feats2, extras2 = vet.compute_features(t2, f2, e2, s2)
    assert "secondary" not in vet.rule_flags(feats2, extras2)
