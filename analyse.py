#!/usr/bin/env python3
"""Astraeus morning analysis: python analyse.py /workspace/astraeus/log.csv

Outputs (into ./analysis_out):
  - summary.txt          overall + per-outcome failure rates, P1/P2/P3 re-weighted
  - rates_by_param.png   failure rate vs each disturbance component (binned)
  - prior_compare.png    failure probability under each prior (with bootstrap CIs)
  - rank_shift.txt       failure-mode ranking under each prior + Spearman rho

Importance weights: sampling q = P1, so w1 = 1; w2 = p2/q; w3 = p3/q
(P3 tight-kernel — see priors.py; its near-zero mass on sun_e is a FINDING).
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import priors

FAIL_OUTCOMES = ["timeout", "tip_over", "stuck"]
PARAMS = ["k_soil", "theta_r", "r_terrain", "sun_e", "sun_psi"]


def weights(df):
    w1 = np.ones(len(df))
    w2 = np.zeros(len(df)); w3 = np.zeros(len(df))
    for i, r in df.iterrows():
        x = {k: r[k] for k in PARAMS}
        q = priors.pdf_P1(x)
        if q <= 0:   # batch-tier values sit outside P1 range? still fine: q>0 needed
            w2[i] = w3[i] = 0.0
            continue
        w2[i] = priors.pdf_P2(x) / q
        w3[i] = priors.pdf_P3(x) / q
    for w in (w2, w3):
        s = w.sum()
        if s > 0:
            w /= s / len(w)   # self-normalise to mean 1
    return {"P1": w1, "P2": w2, "P3": w3}


def wrate(fail, w):
    return float((fail * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


def boot_ci(fail, w, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(fail))
    vals = [wrate(fail[j], w[j]) for j in
            (rng.choice(idx, size=len(idx), replace=True) for _ in range(n))]
    return np.nanpercentile(vals, [2.5, 97.5])


def main(path):
    out = "analysis_out"; os.makedirs(out, exist_ok=True)
    df = pd.read_csv(path).reset_index(drop=True)
    df["fail"] = df["outcome"].isin(FAIL_OUTCOMES).astype(int)
    W = weights(df)
    fail = df["fail"].to_numpy()

    lines = [f"episodes: {len(df)}   (batches: {df['batch_id'].nunique()})",
             f"outcomes: {df['outcome'].value_counts().to_dict()}", ""]
    for k, w in W.items():
        lo, hi = boot_ci(fail, w)
        lines.append(f"failure prob under {k}: {wrate(fail, w):.3f}  "
                     f"[95% CI {lo:.3f}, {hi:.3f}]")
    lines.append("")

    # failure-mode ranking per prior (+ Spearman between rankings)
    modes = sorted(df.loc[df.fail == 1, "outcome"].unique())
    ranks = {}
    lines.append("failure-mode weighted shares per prior:")
    for k, w in W.items():
        shares = {m: wrate((df["outcome"] == m).to_numpy().astype(int), w)
                  for m in modes}
        ranks[k] = pd.Series(shares).rank(ascending=False)
        lines.append(f"  {k}: " + ", ".join(f"{m}={v:.3f}" for m, v in shares.items()))
    if len(modes) >= 2:
        from scipy.stats import spearmanr
        for a, b in [("P1", "P2"), ("P1", "P3"), ("P2", "P3")]:
            rho = spearmanr(ranks[a], ranks[b]).statistic
            lines.append(f"Spearman rank corr {a} vs {b}: {rho:.3f}")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines))
    print("\n".join(lines))

    # failure rate vs each parameter, binned
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(4 * len(PARAMS), 3.2))
    for ax, p in zip(np.atleast_1d(axes), PARAMS):
        if df[p].nunique() <= 3:   # batch-tier discrete levels
            g = df.groupby(p)["fail"].mean()
            ax.bar([str(round(v, 3)) for v in g.index], g.values, color="0.2")
        else:
            bins = np.linspace(df[p].min(), df[p].max(), 7)
            g = df.groupby(pd.cut(df[p], bins), observed=True)["fail"].mean()
            ax.bar(range(len(g)), g.values, color="0.2")
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels([f"{iv.mid:.2f}" for iv in g.index], rotation=45, fontsize=7)
        ax.set_title(p, fontsize=9); ax.set_ylim(0, 1)
    axes[0].set_ylabel("failure rate")
    fig.tight_layout(); fig.savefig(os.path.join(out, "rates_by_param.png"), dpi=160)

    # prior comparison bar
    fig2, ax = plt.subplots(figsize=(4, 3.2))
    ks = list(W); vals = [wrate(fail, W[k]) for k in ks]
    cis = [boot_ci(fail, W[k]) for k in ks]
    ax.bar(ks, vals, color=["0.6", "0.35", "0.1"])
    for i, (v, (lo, hi)) in enumerate(zip(vals, cis)):
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color="k", capsize=4)
    ax.set_ylabel("failure probability"); ax.set_ylim(0, 1)
    fig2.tight_layout(); fig2.savefig(os.path.join(out, "prior_compare.png"), dpi=160)
    print(f"\nwrote {out}/summary.txt, rates_by_param.png, prior_compare.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/workspace/astraeus/log.csv")
