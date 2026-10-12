"""Command line entry point.

    python -m tph hunt --n 300 --hours 4        # screen fresh stars (what the bot runs)
    python -m tph train-synthetic --n 2400      # train the vetter on simulated data
    python -m tph train-real --n 300            # retrain on real TESS planets / false positives
    python -m tph dashboard                     # rebuild docs/ site and results summary
    python -m tph demo                          # offline end-to-end demo on a simulated star
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import config

_KNOWN = None


def _init_worker(known):
    global _KNOWN
    _KNOWN = known


def _hunt_one(star):
    from .pipeline import process_star
    t = time.time()
    screened, cands = process_star(star, _KNOWN)
    screened["seconds"] = round(time.time() - t, 1)
    return screened, cands


def _append(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def _write_status(state, run_start, n_total, n_done, feed, n_cands, note=""):
    """Small JSON the Mission Control page polls to show what the bot is doing."""
    import datetime as dt
    config.DOCS.mkdir(exist_ok=True)
    (config.DOCS / "status.json").write_text(json.dumps({
        "state": state, "note": note,
        "run_started": dt.datetime.fromtimestamp(run_start, dt.timezone.utc).isoformat(timespec="seconds"),
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "targets": n_total, "done": n_done, "candidates_this_run": n_cands,
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "repo": os.environ.get("GITHUB_REPOSITORY"),
        "feed": feed[-25:],
    }, indent=1))


def _checkpoint_push(msg):
    """Inside GitHub Actions, push partial results so the public site updates mid-run."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    import subprocess
    who = ("-c user.name=tess-hunter-bot "
           "-c user.email=41898282+github-actions[bot]@users.noreply.github.com")
    cmd = ("git add results docs models data/*.csv 2>/dev/null; "
           f"git diff --cached --quiet || (git {who} commit -qm \"{msg}\" && "
           f"(git {who} pull -q --rebase -X theirs && git push -q) || true)")
    subprocess.run(cmd, shell=True, cwd=config.ROOT)


def cmd_hunt(a):
    from .catalogs import KnownObjects, select_targets
    from .dashboard import build
    start = time.time()
    rng = np.random.default_rng(a.seed if a.seed is not None else int(time.time()))
    _write_status("starting", start, 0, 0, [], 0, "loading catalogs and choosing stars")
    known = KnownObjects()
    from . import model
    m = model.load()   # rebuild once here (if needed) so workers just read the file
    print(f"[hunt] vetter: {m['metrics'].get('source') if m else 'none'}")
    done = set(pd.read_csv(config.SCREENED_CSV)["tic"]) if config.SCREENED_CSV.exists() else set()
    print(f"[hunt] {len(done)} stars already screened (never repeated); selecting {a.n} new targets")
    targets = select_targets(rng, a.n, exclude=done)
    print(f"[hunt] {len(targets)} targets selected in {time.time() - start:.0f}s")
    _write_status("running", start, len(targets), 0, [], 0)
    _checkpoint_push("bot: run started")
    budget = a.hours * 3600
    buf_s, buf_c, n_done, n_cands, feed = [], [], 0, 0, []
    last_push = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=(known,)) as ex:
        futs = {ex.submit(_hunt_one, s): s for s in targets}
        for fut in as_completed(futs):
            star = futs[fut]
            try:
                screened, cands = fut.result()
            except Exception as exc:
                screened, cands = {"tic": star["tic"], "status": "worker_error",
                                   "note": str(exc)[:120]}, []
            buf_s.append(screened); buf_c.extend(cands); n_done += 1
            n_cands += sum(c["status"] in ("strong_candidate", "candidate") for c in cands)
            feed.append({"tic": int(star["tic"]), "status": screened.get("status"),
                         "ra": star.get("ra"), "dec": star.get("dec"), "tmag": star.get("tmag"),
                         "sectors": screened.get("sectors", ""), "period": screened.get("best_period"),
                         "sde": screened.get("best_sde"), "seconds": screened.get("seconds")})
            tag = ", ".join(f"{c['id']}:{c['status']}" for c in cands)
            print(f"  [{n_done}/{len(targets)}] TIC {screened['tic']} {screened['status']} {tag}", flush=True)
            if len(buf_s) >= 10:
                _append(config.SCREENED_CSV, buf_s); _append(config.CANDIDATES_CSV, buf_c)
                buf_s, buf_c = [], []
                _write_status("running", start, len(targets), n_done, feed, n_cands)
            if time.time() - last_push > a.push_minutes * 60:
                _append(config.SCREENED_CSV, buf_s); _append(config.CANDIDATES_CSV, buf_c)
                buf_s, buf_c = [], []
                build()
                _write_status("running", start, len(targets), n_done, feed, n_cands)
                _checkpoint_push(f"bot: progress {n_done}/{len(targets)}")
                last_push = time.time()
            if time.time() - start > budget:
                print("[hunt] time budget reached, stopping early")
                for f in futs:
                    f.cancel()
                break
    _append(config.SCREENED_CSV, buf_s); _append(config.CANDIDATES_CSV, buf_c)
    _append(config.RESULTS / "runs.csv", [{
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "started_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime(start)),
        "minutes": round((time.time() - start) / 60, 1), "stars": n_done,
        "new_candidates": n_cands}])
    _write_status("idle", start, len(targets), n_done, feed, n_cands, "next run in under 6 hours")
    build()
    print(f"[hunt] finished {n_done} stars in {(time.time() - start) / 60:.1f} min")


def cmd_train_synthetic(a):
    from . import model
    path = config.ROOT / "data" / "synthetic_train.csv"
    if a.n > 0:
        print(f"[train] simulating {a.n} light curves")
        model.build_synthetic(a.n, workers=a.workers, out=path)
    df = pd.read_csv(path)
    metrics, _ = model.train(df, source="synthetic")
    print(json.dumps(metrics, indent=2))


def _real_one(row):
    """Run the blind search on a known TOI and see if we recover its catalog period."""
    from .fetch import get_lightcurve
    from . import detrend, search, vet
    try:
        lc = get_lightcurve(int(row["tic"]))
        if lc is None:
            return None
        t, f, e, _, _ = detrend.prepare(lc["t"], lc["f"], lc["e"])
        sigs = search.iterative_search(t, f, e)  # list of signals
        out = {"tic": int(row["tic"]), "toi": row["toi"], "disp": row["disp"],
               "cat_period": row["period"], "recovered": False}
        for s in sigs:
            r = s.period / row["period"]
            if any(abs(r / h - 1) < 0.01 for h in (1, 2, 0.5)):
                out["recovered"] = True
            if abs(r - 1) < 0.01:
                keep = s.keep
                feats, extras = vet.compute_features(t[keep], f[keep], e[keep], s,
                                                     row["rstar"], row["mstar"])
                out.update(feats, found_period=s.period, sde=s.sde, snr=s.snr,
                           passes=vet.passes_detection(s, extras), **extras)
                break
        return out
    except Exception as exc:
        print(f"  TOI {row['toi']} failed: {exc}")
        return None


def cmd_train_real(a):
    from astroquery.mast import Catalogs
    from . import model, vet
    from .catalogs import load_tois
    tois = load_tois().dropna(subset=["period"])
    tois = tois[(tois.period > config.PERIOD_MIN) & (tois.period < config.PERIOD_MAX)]
    pl = tois[tois.disp.isin(["CP", "KP"])].sample(frac=1, random_state=1).head(a.n)
    fp = tois[tois.disp.isin(["FP", "FA"])].sample(frac=1, random_state=1).head(a.n)
    sample = pd.concat([pl, fp]).drop_duplicates("tic").reset_index(drop=True)
    print(f"[train-real] {len(pl)} planets + {len(fp)} false positives")
    params = []
    for i in range(0, len(sample), 200):
        ids = [int(x) for x in sample.tic.iloc[i:i + 200]]
        try:
            t = Catalogs.query_criteria(catalog="Tic", ID=ids).to_pandas()
            params.append(t[["ID", "rad", "mass"]])
        except Exception as exc:
            print(f"  TIC param query failed: {exc}")
    if params:
        p = pd.concat(params).rename(columns={"ID": "tic", "rad": "rstar", "mass": "mstar"})
        p["tic"] = p.tic.astype(np.int64)
        sample = sample.merge(p.drop_duplicates("tic"), on="tic", how="left")
    else:
        sample["rstar"], sample["mstar"] = 1.0, 1.0
    sample["rstar"] = sample.rstar.fillna(1.0)
    sample["mstar"] = sample.mstar.fillna(sample.rstar)
    rows = []
    start = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_real_one, r) for r in sample.to_dict("records")]
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r is not None:
                rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(futs)} processed", flush=True)
                pd.DataFrame(rows).to_csv(config.ROOT / "data" / "real_train.csv", index=False)
            if time.time() - start > a.hours * 3600:
                for f in futs:
                    f.cancel()
                break
    real = pd.DataFrame(rows)
    real.to_csv(config.ROOT / "data" / "real_train.csv", index=False)
    is_pl = real.disp.isin(["CP", "KP"])
    validation = {
        "date": time.strftime("%Y-%m-%d"),
        "known_planets_tested": int(is_pl.sum()),
        "known_planets_recovered": int(real[is_pl].recovered.sum()),
        "recovery_rate": round(float(real[is_pl].recovered.mean()), 4) if is_pl.any() else None,
        "false_positives_tested": int((~is_pl).sum()),
    }
    feat = real.dropna(subset=vet.FEATURES).copy()
    feat["label"] = feat.disp.isin(["CP", "KP"]).astype(int)
    feat["kind"] = np.where(feat.label == 1, "real_planet", "real_false_positive")
    syn = pd.read_csv(config.ROOT / "data" / "synthetic_train.csv")
    both = pd.concat([syn, feat], ignore_index=True)
    metrics, p = model.train(both, source=f"synthetic+real ({len(feat)} real TOIs)")
    real_mask = both.kind.str.startswith("real").to_numpy()
    if real_mask.sum() > 20 and len(set(both.label[real_mask])) == 2:
        from sklearn.metrics import roc_auc_score
        yr, pr = both.label[real_mask].to_numpy(), p[real_mask]
        metrics["real_only"] = {
            "n": int(real_mask.sum()),
            "roc_auc": round(float(roc_auc_score(yr, pr)), 4),
            "planets_scored_above_0.5": round(float((pr[yr == 1] >= 0.5).mean()), 4),
            "false_positives_rejected": round(float((pr[yr == 0] < 0.5).mean()), 4)}
        (config.MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
        import joblib
        m = joblib.load(config.MODEL_PATH); m["metrics"] = metrics; joblib.dump(m, config.MODEL_PATH)
    validation["model"] = metrics
    config.VALIDATION_JSON.parent.mkdir(exist_ok=True)
    config.VALIDATION_JSON.write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))
    from .dashboard import build
    build()


def cmd_dashboard(a):
    from .dashboard import build
    build()


def cmd_demo(a):
    """Offline demo: hide a 2.5 Earth-radius planet in a simulated 2-sector light curve
    and run the full pipeline on it."""
    from . import simulate
    from .pipeline import analyze
    rng = np.random.default_rng(a.seed or 7)
    t = simulate.time_grid(rng, n_sectors=2)
    flux, sigma = simulate.noise_and_variability(rng, t)
    rstar, mstar, rp, period = 0.85, 0.88, a.rp, a.period
    k = rp / (rstar * 109.1)
    drop = simulate.eclipse_signal(t, period, t.min() + 1.3, k,
                                   simulate.a_over_rs(period, rstar, mstar), b=0.3)
    flux = flux * (1 - drop)
    star = {"tic": 0, "tmag": 11.0, "teff": 5300, "rstar": rstar, "mstar": mstar,
            "ra": 84.29, "dec": -35.17}
    lc = {"t": t, "f": flux, "e": np.full_like(t, sigma), "sectors": ["sim1", "sim2"],
          "authors": ["simulated"]}
    screened, cands = analyze(star, lc)
    from .replay import STARS_DIR
    if (STARS_DIR / "0.json").exists():
        (STARS_DIR / "0.json").replace(STARS_DIR / "demo.json")   # sample for Mission Control
    print(f"injected: P={period} d, Rp={rp} Re, depth={k**2*1e6:.0f} ppm, noise={sigma*1e6:.0f} ppm")
    for c in cands:
        print("recovered:", {kk: c[kk] for kk in ("id", "status", "score", "period", "rp_earth", "snr", "flags", "plot")})


def main():
    ap = argparse.ArgumentParser(prog="tph", description="TESS Planet Hunter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = max(2, (os.cpu_count() or 2))
    p = sub.add_parser("hunt"); p.add_argument("--n", type=int, default=300)
    p.add_argument("--hours", type=float, default=4.0); p.add_argument("--seed", type=int)
    p.add_argument("--workers", type=int, default=w)
    p.add_argument("--push-minutes", type=float, default=15.0, dest="push_minutes")
    p.set_defaults(fn=cmd_hunt)
    p = sub.add_parser("train-synthetic"); p.add_argument("--n", type=int, default=2400)
    p.add_argument("--workers", type=int, default=w); p.set_defaults(fn=cmd_train_synthetic)
    p = sub.add_parser("train-real"); p.add_argument("--n", type=int, default=300)
    p.add_argument("--hours", type=float, default=4.5)
    p.add_argument("--workers", type=int, default=w); p.set_defaults(fn=cmd_train_real)
    p = sub.add_parser("dashboard"); p.set_defaults(fn=cmd_dashboard)
    p = sub.add_parser("demo"); p.add_argument("--seed", type=int)
    p.add_argument("--rp", type=float, default=2.5); p.add_argument("--period", type=float, default=4.37)
    p.set_defaults(fn=cmd_demo)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
