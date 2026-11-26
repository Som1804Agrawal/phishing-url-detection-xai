"""Train and compare several ML classifiers for phishing URL detection.

Usage (synthetic demo):
    python src/train_evaluate.py --data data/sample_phishing_synthetic.csv
Usage (real Kaggle data, URL + label columns, optionally DNS/WHOIS columns):
    python src/train_evaluate.py --data data/phishing_urls.csv --url-col url --label-col label --positive-label 1
If the CSV has NO url column, every numeric column except the label is used as a feature.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from features import ALL_FEATURES, DNS_FEATURES, FEATURE_GROUP, WHOIS_FEATURES, build_lexical_frame, infer_group


LABEL_CANDIDATES = ["label", "Label", "class", "Class", "target", "Target", "phishing", "Phishing", "result", "Result", "status", "Status", "type"]


def load_data(args):
    df = pd.read_csv(args.data)
    if args.label_col not in df.columns:
        found = [c for c in LABEL_CANDIDATES if c in df.columns]
        if not found:
            raise SystemExit(f"Label column '{args.label_col}' not found. Columns are: {list(df.columns)[:15]} ... Use --label-col.")
        print(f"  label column '{args.label_col}' not found; using '{found[0]}'")
        args.label_col = found[0]
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        print(f"  removed {before - len(df):,} duplicate rows")
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=args.seed)
    y = (df[args.label_col] == args.positive_label).astype(int).values
    if args.url_col in df.columns:
        X = build_lexical_frame(df[args.url_col]).reset_index(drop=True)
        for col in DNS_FEATURES + WHOIS_FEATURES:
            if col in df.columns:
                X[col] = df[col].values
        X = X[[c for c in ALL_FEATURES if c in X.columns]]
        groups = {f: FEATURE_GROUP[f] for f in X.columns}
    else:  # dataset ships ready-made numeric features (e.g. Kaggle final_dataset.csv with 74 features)
        X = df.drop(columns=[args.label_col]).select_dtypes("number").reset_index(drop=True)
        override = json.loads(Path(args.groups_json).read_text()) if args.groups_json else {}
        groups = {f: override.get(f, infer_group(f)) for f in X.columns}
        counts = pd.Series(groups).value_counts().to_dict()
        print(f"  feature families inferred from column names: {counts}  (check against your dataset description; use --groups-json to fix)")
    return X, y, groups


def make_models(seed, tune):
    models = {
        "Logistic Regression": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=2000, random_state=seed))]),
        "Random Forest": Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed))]),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=seed),
        # Linear SVM + probability calibration: an RBF SVC is impractical on ~580k rows
        "SVM (Linear)": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                  ("clf", CalibratedClassifierCV(LinearSVC(dual=False, C=1.0, random_state=seed), cv=3))]),
    }
    try:  # optional extra model
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1,
                                          eval_metric="logloss", random_state=seed)
    except ImportError:
        pass
    grids = {
        "Random Forest": {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [None, 15, 25], "clf__min_samples_leaf": [1, 2, 4]},
        "Gradient Boosting": {"learning_rate": [0.05, 0.1, 0.2], "max_depth": [None, 6, 10], "max_iter": [200, 300]},
    }
    return models, grids if tune else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--url-col", default="url")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive-label", default=1, help="value of label-col that means PHISHING")
    ap.add_argument("--sample", type=int, default=0, help="optionally subsample N rows (speed)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv-folds", type=int, default=3, help="0 disables cross-validation")
    ap.add_argument("--tune", action="store_true", help="randomized hyper-parameter search (RF, GB)")
    ap.add_argument("--groups-json", default="", help="optional JSON {feature: Lexical|DNS|WHOIS} for datasets with ready-made features")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()
    try:
        args.positive_label = type(pd.read_csv(args.data, nrows=1)[args.label_col].iloc[0])(args.positive_label)
    except Exception:
        pass

    out, figs = Path(args.out), Path(args.out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(exist_ok=True)

    X, y, groups = load_data(args)
    print(f"Data: {X.shape[0]} rows, {X.shape[1]} features, phishing share {y.mean():.2%}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.seed)

    models, grids = make_models(args.seed, args.tune)
    rows, fitted, probas, cms, reports = [], {}, {}, {}, []
    for name, model in models.items():
        t0 = time.time()
        if name in grids:
            search = RandomizedSearchCV(model, grids[name], n_iter=6, cv=3, scoring="f1", n_jobs=-1, random_state=args.seed)
            search.fit(Xtr, ytr)
            model = search.best_estimator_
            print(f"  tuned {name}: {search.best_params_}")
        else:
            model.fit(Xtr, ytr)
        train_time = time.time() - t0
        cv_f1 = np.nan
        if args.cv_folds > 1:
            cv_f1 = cross_val_score(model, Xtr, ytr, cv=StratifiedKFold(args.cv_folds, shuffle=True, random_state=args.seed),
                                    scoring="f1", n_jobs=1).mean()
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fitted[name], probas[name] = model, proba
        cms[name] = confusion_matrix(yte, pred)
        reports.append(f"=== {name} ===\n{classification_report(yte, pred, target_names=['Legitimate', 'Phishing'], digits=4)}")
        tn, fp, fn, tp = cms[name].ravel()
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred),
                     "Recall": recall_score(yte, pred), "F1-score": f1_score(yte, pred),
                     "ROC-AUC": roc_auc_score(yte, proba), f"CV F1 ({args.cv_folds}-fold)": cv_f1,
                     "Train time (s)": train_time,
                     "Success rate (%)": 100 * (tp + tn) / (tp + tn + fp + fn), "Failure rate (%)": 100 * (fp + fn) / (tp + tn + fp + fn),
                     "Phishing detected (%)": 100 * tp / (tp + fn), "Phishing missed (%)": 100 * fn / (tp + fn),
                     "Legit passed (%)": 100 * tn / (tn + fp), "False alarms (%)": 100 * fp / (tn + fp)})
        print(f"  {name:20s} acc={rows[-1]['Accuracy']:.4f} f1={rows[-1]['F1-score']:.4f} ({train_time:.1f}s)")

    res = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
    res.round(4).to_csv(out / "metrics.csv", index=False)
    sf_cols = ["Model", "Success rate (%)", "Failure rate (%)", "Phishing detected (%)", "Phishing missed (%)", "Legit passed (%)", "False alarms (%)"]
    res[sf_cols].round(2).to_csv(out / "success_failure.csv", index=False)
    (out / "classification_reports.txt").write_text("\n".join(reports))
    pd.DataFrame({n: cm.ravel() for n, cm in cms.items()}, index=["TN", "FP", "FN", "TP"]).T.to_csv(out / "confusion_matrices.csv")

    # ---- figures ----
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    axes = np.atleast_1d(axes)
    for ax, (name, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["Legit", "Phish"]); ax.set_yticks([0, 1], ["Legit", "Phish"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "confusion_matrices.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yte, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(figs / "roc_curves.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    m = res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]]
    m.plot.bar(ax=ax, ylim=(max(0, m.values.min() - 0.05), 1.0), width=0.8)
    ax.set_ylabel("Score"); ax.set_title("Model comparison (test set)"); plt.xticks(rotation=20, ha="right")
    ax.legend(fontsize=7, ncol=4, loc="lower center"); fig.tight_layout(); fig.savefig(figs / "metrics_comparison.png", dpi=200); plt.close(fig)

    # success / failure figure
    sf = res.set_index("Model")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    names = list(sf.index)[::-1]
    a1.barh(names, sf.loc[names, "Success rate (%)"], color="#3B6FB6", label="Success (correct)")
    a1.barh(names, sf.loc[names, "Failure rate (%)"], left=sf.loc[names, "Success rate (%)"], color="#C0504D", label="Failure (wrong)")
    for i, n in enumerate(names):
        a1.text(1, i, f"{sf.loc[n, 'Success rate (%)']:.2f}%", va="center", color="white", fontsize=8)
        a1.text(99.5, i, f"{sf.loc[n, 'Failure rate (%)']:.2f}%", va="center", ha="right", color="white", fontsize=7)
    a1.set_xlim(0, 100); a1.set_xlabel("% of test URLs"); a1.set_title("Overall success vs failure", fontsize=10); a1.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    w = 0.2; x = np.arange(len(names))
    for k, (c, col, lab) in enumerate([("Phishing detected (%)", "#3B6FB6", "Phishing detected"), ("Phishing missed (%)", "#C0504D", "Phishing missed (FNR)"),
                                       ("Legit passed (%)", "#6FA37B", "Legit passed"), ("False alarms (%)", "#E8873A", "False alarms (FPR)")]):
        a2.bar(x + (k - 1.5) * w, sf.loc[names, c], w, label=lab, color=col)
    a2.set_xticks(x, names, rotation=15, ha="right", fontsize=8); a2.set_ylabel("%"); a2.set_title("Per-class success and failure", fontsize=10)
    a2.legend(fontsize=7, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.08))
    fig.tight_layout(); fig.savefig(figs / "success_failure.png", dpi=200); plt.close(fig)

    # ---- save best model + everything explain.py needs ----
    best = res.iloc[0]["Model"]
    joblib.dump(fitted[best], Path(args.models_dir) / "best_model.joblib")
    test = Xte.copy(); test["__label__"] = yte
    test.to_csv(out / "test_split.csv", index=False)
    meta = {"best_model": best, "features": list(X.columns), "groups": groups,
            "train_medians": {c: (None if np.isnan(Xtr[c].median()) else float(Xtr[c].median())) for c in X.columns},
            "n_rows": int(len(X)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "phishing_share": float(y.mean()),
            "data_file": str(args.data)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nBest model: {best}  -> saved to {args.models_dir}/best_model.joblib")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
