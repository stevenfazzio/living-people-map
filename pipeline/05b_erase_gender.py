"""Stage 05b: gender-erased variants of the lead embeddings (no API cost).

Two linear erasures of Wikidata gender (P21) from the base embeddings, both
fit on male/female rows (>=99% of people; other classes are too small to
estimate) and applied to ALL rows:

  centroid -- project out the unit difference-of-means direction
              (mu_female - mu_male). First-moment removal only; the
              international-sibling-cities ablation found this the best
              quality/diversity balance for neighborhood tasks.
  leace    -- concept_erasure.LeaceEraser: the least-squares-minimal affine
              map after which NO linear probe can beat chance. Stronger
              guarantee, more perturbation ("leace over-spread" in the
              cities ablation).

Erasure is linear-only: leads still say "actress"/"played Hermione", so
nonlinear signal survives -- the erased maps' gender colormap is the readout.

A held-out logistic probe (AUC) on raw vs erased embeddings is printed as
the before/after check.

Input:  embeddings_{ROSTER}_lead.npz, people_wikidata_{ROSTER}.parquet
Output: embeddings_{ROSTER}_lead_nogender_centroid.npz, ..._leace.npz
"""

import numpy as np
import pandas as pd
import torch
from concept_erasure import LeaceEraser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config
from util import save_npz_atomic


def probe_auc(x: np.ndarray, y: np.ndarray, method: str, seed: int = 42) -> float:
    """Split-FIRST protocol: fit the eraser on train only, transform both
    splits, then probe. Fitting an eraser on the full set before splitting
    yields systematically anti-predictive held-out AUC (any noise direction
    with a residual class gap in train has the opposite gap in test, since
    the full-set gap is exactly zero) -- a metric artifact, not erasure."""
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)
    if method == "centroid":
        d = x_tr[y_tr == 1].mean(axis=0) - x_tr[y_tr == 0].mean(axis=0)
        d /= np.linalg.norm(d)
        x_tr = x_tr - np.outer(x_tr @ d, d)
        x_te = x_te - np.outer(x_te @ d, d)
    elif method == "leace":
        eraser = LeaceEraser.fit(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
        x_tr = eraser(torch.from_numpy(x_tr)).numpy()
        x_te = eraser(torch.from_numpy(x_te)).numpy()
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(x_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1]))


def main() -> None:
    base = np.load(config.DATA_DIR / f"embeddings_{config.ROSTER_VARIANT}_lead.npz", allow_pickle=False)
    emb, pageids, base_sig = base["emb"].astype(np.float32), base["pageid"], str(base["sig"])

    wikidata = pd.read_parquet(config.WIKIDATA_PARQUET, columns=["pageid", "gender"])
    gender = pd.DataFrame({"pageid": pageids}).merge(wikidata, on="pageid", how="left")["gender"]
    is_f = (gender == "female").to_numpy()
    is_m = (gender == "male").to_numpy()
    mask = is_f | is_m
    y = is_f[mask].astype(np.int64)
    print(f"{mask.sum():,}/{len(emb):,} people are male/female-labeled ({is_f.sum():,} female)")

    # centroid: project out the difference-of-means direction, uniformly.
    d = emb[is_f].mean(axis=0) - emb[is_m].mean(axis=0)
    d /= np.linalg.norm(d)
    emb_centroid = emb - np.outer(emb @ d, d)

    # LEACE: fit on labeled rows, apply to everyone.
    eraser = LeaceEraser.fit(torch.from_numpy(emb[mask]), torch.from_numpy(y))
    emb_leace = eraser(torch.from_numpy(emb)).numpy().astype(np.float32)

    print("\nHeld-out probe AUC, split-first protocol (0.5 = fully erased):")
    for method in ("raw", "centroid", "leace"):
        print(f"  {method:9s} {probe_auc(emb[mask], y, method):.4f}", flush=True)

    for method, x in [("centroid", emb_centroid), ("leace", emb_leace)]:
        out = config.DATA_DIR / f"embeddings_{config.ROSTER_VARIANT}_lead_nogender_{method}.npz"
        save_npz_atomic(out, emb=x, pageid=pageids, sig=np.str_(f"{base_sig}_nogender_{method}"))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
