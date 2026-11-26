"""Explainable-AI analysis of the best model (run after train_evaluate.py).

Always available (scikit-learn only):
  * permutation feature importance (global)  * feature-GROUP importance (Lexical / DNS / WHOIS)
  * partial dependence plots                 * occlusion-based local explanations
Used automatically if installed:  SHAP (global + local)  and  LIME (local).
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

warnings.filterwarnings("ignore")
GROUP_COLORS = {"Lexical": "#2a6f97", "DNS": "#e07a1f", "WHOIS": "#3a9d5d", "Dataset-provided": "#777777"}


def prob(model, X):
    return model.predict_proba(X)[:, 1]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def occlusion_attribution(model, row: pd.Series, medians: dict):
    """Local explanation (model-agnostic, no extra packages).

    For each feature, replace its value by the training median and measure how much the model's
    log-odds of 'phishing' drops. Positive value = this feature's actual value pushes the URL
    towards 'phishing'. Returns (P(phishing), Series sorted by absolute effect).
    """
    base = prob(model, row.to_frame().T.astype(float))[0]
    contrib = {}
    for f in row.index:
        alt = row.copy()
        alt[f] = medians[f] if medians[f] is not None else np.nan
        contrib[f] = logit(base) - logit(prob(model, alt.to_frame().T.astype(float))[0])
    return base, pd.Series(contrib).sort_values(key=np.abs, ascending=False)


def plot_local(contrib: pd.Series, base: float, title: str, groups: dict, path: Path, k=10):
    top = contrib.head(k)[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.barh(top.index, top.values, color=[GROUP_COLORS.get(groups[f], "#777") for f in top.index])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Effect on log-odds of phishing  (+ = towards phishing)")
    ax.set_title(f"{title}\nP(phishing) = {base:.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", default="models/best_model.joblib")
    ap.add_argument("--n", type=int, default=2000, help="test rows used for permutation importance / SHAP")
    ap.add_argument("--shap-n", type=int, default=500, help="rows used for the SHAP summary (SHAP is slow on big forests)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    res, figs = Path(a.results), Path(a.results) / "figures"
    model = joblib.load(a.model)
    meta = json.loads((res / "meta.json").read_text())
    feats, groups, med = meta["features"], meta["groups"], meta["train_medians"]
    test = pd.read_csv(res / "test_split.csv")
    y = test.pop("__label__").values
    X = test[feats].astype(float)
    idx = np.random.default_rng(a.seed).choice(len(X), min(a.n, len(X)), replace=False)
    Xs, ys = X.iloc[idx], y[idx]
    print("Explaining:", meta["best_model"])

    # 1) permutation importance (global)
    pi = permutation_importance(model, Xs, ys, scoring="f1", n_repeats=5, random_state=a.seed, n_jobs=1)
    imp = pd.DataFrame({"feature": feats, "group": [groups[f] for f in feats], "importance": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("importance", ascending=False)
    imp.round(5).to_csv(res / "permutation_importance.csv", index=False)
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top.feature, top.importance, xerr=top["std"], color=[GROUP_COLORS.get(g, "#777") for g in top.group])
    ax.set_xlabel("Drop in F1 when feature is shuffled"); ax.set_title("Global importance (top 15)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for g, c in GROUP_COLORS.items() if g in set(top.group)]
    ax.legend(handles, [g for g in GROUP_COLORS if g in set(top.group)], fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(figs / "global_importance.png", dpi=200); plt.close(fig)

    # 2) feature-group importance
    gi = imp.assign(importance=imp.importance.clip(lower=0)).groupby("group").importance.sum()
    gi = (gi / gi.sum() * 100).sort_values(ascending=False)
    gi.round(2).to_csv(res / "group_importance.csv", header=["share_percent"])
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.bar(gi.index, gi.values, color=[GROUP_COLORS.get(g, "#777") for g in gi.index])
    for i, v in enumerate(gi.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Share of total importance (%)"); ax.set_title("Importance by feature family")
    fig.tight_layout(); fig.savefig(figs / "group_importance.png", dpi=200); plt.close(fig)

    # 3) partial dependence of top-4 features
    try:
        fig, ax = plt.subplots(2, 2, figsize=(6.5, 5))
        PartialDependenceDisplay.from_estimator(model, Xs, list(imp.feature.head(4)), ax=ax.ravel(), kind="average")
        fig.suptitle("Partial dependence of P(phishing)", fontsize=10); fig.tight_layout()
        fig.savefig(figs / "partial_dependence.png", dpi=200); plt.close(fig)
    except Exception as e:
        print("  partial dependence skipped:", e)

    # 4) local explanations (occlusion) for one phishing and one legitimate URL
    proba = prob(model, Xs)
    # pick correctly-classified, not-fully-saturated examples so the explanation is informative
    picks = {"phishing": int(np.argmin(np.where(ys == 1, np.abs(proba - 0.9), 9))),
             "legitimate": int(np.argmin(np.where(ys == 0, np.abs(proba - 0.1), 9)))}
    local_rows = []
    for kind, i in picks.items():
        base, contrib = occlusion_attribution(model, Xs.iloc[i], med)
        plot_local(contrib, base, f"Local explanation - example {kind} URL", groups, figs / f"local_{kind}.png")
        local_rows += [{"case": kind, "feature": f, "value": Xs.iloc[i][f], "effect": c} for f, c in contrib.head(10).items()]
    pd.DataFrame(local_rows).round(4).to_csv(res / "local_explanations.csv", index=False)

    # 5) optional SHAP
    try:
        import shap
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        Xt = model[:-1].transform(Xs) if hasattr(model, "steps") and len(model.steps) > 1 else Xs
        Xt = pd.DataFrame(Xt, columns=feats).iloc[:a.shap_n]
        sv = shap.TreeExplainer(est).shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else np.asarray(sv)
        if sv.ndim == 3:          # newer SHAP versions return (rows, features, classes)
            sv = sv[:, :, 1]      # class 1 = phishing
        plt.figure()
        shap.summary_plot(sv, Xt, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(figs / "shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()
        print("  SHAP summary saved")
    except Exception as e:
        print(f"  SHAP not run ({type(e).__name__}); install `shap` and use a tree-based best model to enable it.")

    # 6) optional LIME
    try:
        from lime.lime_tabular import LimeTabularExplainer
        Xf = Xs.fillna(pd.Series(med))
        ex = LimeTabularExplainer(Xf.values, feature_names=feats, class_names=["legitimate", "phishing"], discretize_continuous=True)
        for kind, i in picks.items():
            e = ex.explain_instance(Xf.iloc[i].values, lambda d: model.predict_proba(pd.DataFrame(d, columns=feats)), num_features=10, num_samples=1000)
            e.save_to_file(str(res / f"lime_{kind}.html"))
        print("  LIME explanations saved")
    except Exception as e:
        print(f"  LIME not run ({type(e).__name__}); install `lime` to enable it.")

    print("\nTop 10 features:\n", imp.head(10)[["feature", "group", "importance"]].round(4).to_string(index=False))
    print("\nImportance by family (%):\n", gi.round(1).to_string())


if __name__ == "__main__":
    main()
