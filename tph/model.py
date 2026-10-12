"""The AI vetter: a gradient-boosted decision-tree classifier that scores each detected
signal with a probability of being a planet (vs. eclipsing binary / systematic / noise)."""
import json
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config, detrend, search, simulate, vet

_MODEL = None


def _simulate_one(seed):
    try:
        return _simulate_one_inner(seed)
    except Exception as exc:
        print(f"  sim {seed} failed: {exc}")
        return None


def _simulate_one_inner(seed):
    rng = np.random.default_rng(seed)
    kind = rng.choice(simulate.CLASSES, p=[0.45, 0.2, 0.2, 0.15])
    t, f, e, truth = simulate.simulate(rng, kind)
    lc = None
    if rng.random() < 0.6:   # ~60% of real stars have SPOC centroids; the rest are QLP-only
        noise = 10 ** rng.uniform(-3.5, -2.3)          # pixels
        cx, cy = rng.normal(0, noise, t.size), rng.normal(0, noise, t.size)
        if kind == "beb" and "in_transit" in truth:     # the light centre moves toward the binary
            amp = noise * rng.uniform(0.3, 3.0)
            cx = cx + amp * truth["in_transit"]; cy = cy + 0.5 * amp * truth["in_transit"]
        lc = {"t": t, "cx": cx, "cy": cy, "sec": np.floor(t / 27.4)}
    t, f, e, _, _ = detrend.prepare(t, f, e)
    sig = search.bls_search(t, f, e)
    if sig is None or sig.sde < config.MIN_SDE * 0.8:
        return None
    # Planet stellar params as a catalog would give them (10% scatter).
    r = truth["rstar"] * rng.normal(1, 0.1)
    m = truth["mstar"] * rng.normal(1, 0.1)
    feats, extras = vet.compute_features(t, f, e, sig, r, m, lc=lc)
    recovered = kind == "planet" and abs(sig.period / truth["period"] - 1) < 0.01
    row = dict(feats, label=int(recovered), kind=kind, seed=int(seed),
               true_period=truth.get("period", np.nan), found_period=sig.period,
               rp_true=truth.get("rp", np.nan), **extras)
    return row


def build_synthetic(n, workers=2, start_seed=0, out=None):
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(_simulate_one, range(start_seed, start_seed + n), chunksize=8)):
            if row is not None:
                rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"  simulated {i + 1}/{n}  usable={len(rows)}", flush=True)
                if out:
                    pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    if out:
        df.to_csv(out, index=False)
    return df


def train(df, source="synthetic"):
    X, y = df[vet.FEATURES].to_numpy(float), df["label"].to_numpy(int)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, class_weight="balanced",
                                         random_state=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = p >= config.CANDIDATE_SCORE
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary")
    metrics = {
        "source": source, "n_train": int(len(y)), "n_planets": int(y.sum()),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "avg_precision": round(float(average_precision_score(y, p)), 4),
        "precision@0.5": round(float(prec), 4), "recall@0.5": round(float(rec), 4),
        "f1@0.5": round(float(f1), 4),
    }
    if "kind" in df:
        metrics["false_positive_rate_by_class"] = {
            k: round(float((p[(df.kind == k).to_numpy() & (y == 0)] >= 0.5).mean()), 4)
            for k in df.kind.unique() if ((df.kind == k) & (y == 0)).any()}
    clf.fit(X, y)
    config.MODELS.mkdir(exist_ok=True)
    import sklearn
    joblib.dump({"model": clf, "features": vet.FEATURES, "metrics": metrics,
                 "sklearn": sklearn.__version__}, config.MODEL_PATH)
    (config.MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics, p


def training_frame():
    """Synthetic examples plus any real TOI examples collected by train-real."""
    data = config.ROOT / "data"
    frames = []
    if (data / "synthetic_train.csv").exists():
        frames.append(pd.read_csv(data / "synthetic_train.csv"))
    if (data / "real_train.csv").exists():
        real = pd.read_csv(data / "real_train.csv").dropna(subset=vet.FEATURES)
        if len(real):
            real["label"] = real.disp.isin(["CP", "KP"]).astype(int)
            real["kind"] = np.where(real.label == 1, "real_planet", "real_false_positive")
            frames.append(real)
    return pd.concat(frames, ignore_index=True) if frames else None


def load():
    """Load the saved model; if it was pickled by a different scikit-learn version,
    rebuild it from the committed training data (takes seconds)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import sklearn
    m = None
    if config.MODEL_PATH.exists():
        try:
            m = joblib.load(config.MODEL_PATH)
            if m.get("sklearn") != sklearn.__version__:
                m = None
        except Exception:
            m = None
    if m is None:
        df = training_frame()
        if df is None:
            return None
        n_real = int(df.kind.astype(str).str.startswith("real").sum()) if "kind" in df else 0
        print(f"  [model] rebuilding vetter for scikit-learn {sklearn.__version__}")
        train(df, source="synthetic" + (f"+real ({n_real} real TOIs)" if n_real else ""))
        m = joblib.load(config.MODEL_PATH)
    _MODEL = m
    return _MODEL


def score(feats):
    m = load()
    if m is None:
        return float("nan")
    x = np.array([[feats[k] for k in m["features"]]], float)
    return float(m["model"].predict_proba(x)[0, 1])


HARD_FLAGS = {"too_large", "secondary", "odd_even", "centroid_shift", "single_event", "outlier_driven"}


def pipeline_metrics(df):
    """Cross-validated accuracy of the *whole* vetting stage (AI score + red-flag rules)
    on real TESS objects: how many confirmed planets survive, how many known false
    positives are thrown out."""
    X, y = df[vet.FEATURES].to_numpy(float), df["label"].to_numpy(int)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, class_weight="balanced", random_state=0)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1),
                          method="predict_proba")[:, 1]
    real = df["kind"].astype(str).str.startswith("real").to_numpy()
    if real.sum() < 20:
        return None
    r = df[real]
    kept = []
    for (_, row), prob in zip(r.iterrows(), p[real]):
        feats = {k: row[k] for k in vet.FEATURES}
        extras = {"rp_earth": row.get("rp_earth", 1.0), "n_transits": row.get("n_transits", 5),
                  "outlier_ratio": row.get("outlier_ratio", 1.0)}
        kept.append(prob >= config.CANDIDATE_SCORE and not (HARD_FLAGS & set(vet.rule_flags(feats, extras))))
    kept, pl = np.array(kept), r["label"].to_numpy() == 1
    return {"real_planets": int(pl.sum()), "real_false_positives": int((~pl).sum()),
            "planets_kept": round(float(kept[pl].mean()), 4),
            "false_positives_rejected": round(float(1 - kept[~pl].mean()), 4)}
