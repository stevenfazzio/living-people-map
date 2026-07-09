"""Stage 06: reduce embeddings to 2D with UMAP.

Parameters deliberately match the sibling projects (movie-madness-map,
steam-atlas) for cross-map shape consistency: n_neighbors=15, min_dist=0.05,
metric=cosine, random_state=42.

Input:  data/embeddings_lead.npz
Output: data/umap_coords_lead.npz [coords (N,2) float32, pageid (N,) int64]
"""

import numpy as np
import umap

import config
from util import save_npz_atomic


def main() -> None:
    data = np.load(config.embeddings_npz(), allow_pickle=False)
    emb, pageids = data["emb"], data["pageid"]
    print(f"UMAP on {emb.shape}")
    np.random.seed(42)
    reducer = umap.UMAP(**config.UMAP_KWARGS)
    coords = reducer.fit_transform(emb).astype(np.float32)
    save_npz_atomic(config.umap_coords_npz(), coords=coords, pageid=pageids)
    print(f"Wrote {config.umap_coords_npz()} {coords.shape}")


if __name__ == "__main__":
    main()
