"""Before/after metrics for the gender-erasure experiment (global roster).

For each embedding variant with UMAP coords on disk, reports mean k-NN gender
purity in the 2D map (fraction of a person's 15 nearest neighbors sharing
their gender, among male/female-labeled people). Chance level = the purity
of a gender-blind map, which for an imbalanced 2-class mix is
p_m^2 + p_f^2 normalized -- reported alongside.

Run: uv run python analysis/gender_erasure_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

K = 15
VARIANTS = ["lead", "lead_nogender_centroid", "lead_nogender_leace"]


def knn_purity(coords: np.ndarray, labels: np.ndarray) -> float:
    nn = NearestNeighbors(n_neighbors=K + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    neighbor_labels = labels[idx[:, 1:]]  # drop self
    return float((neighbor_labels == labels[:, None]).mean())


def main() -> None:
    wikidata = pd.read_parquet(config.WIKIDATA_PARQUET, columns=["pageid", "gender"])
    print(f"k-NN gender purity in 2D (k={K}), {config.ROSTER_VARIANT} roster:")
    for variant in VARIANTS:
        path = config.DATA_DIR / f"umap_coords_{config.ROSTER_VARIANT}_{variant}.npz"
        if not path.exists():
            print(f"  {variant:24s} (no coords yet)")
            continue
        data = np.load(path, allow_pickle=False)
        gender = pd.DataFrame({"pageid": data["pageid"]}).merge(wikidata, on="pageid", how="left")["gender"]
        mask = gender.isin(["male", "female"]).to_numpy()
        labels = gender[mask].to_numpy()
        purity = knn_purity(data["coords"][mask], labels)
        p = pd.Series(labels).value_counts(normalize=True)
        chance = float((p**2).sum())
        print(f"  {variant:24s} purity {purity:.3f}   (gender-blind baseline {chance:.3f})")


if __name__ == "__main__":
    main()
