"""Stage 05: embed page leads with Cohere embed-v4.0.

Semantic text = the raw lead (it opens with the person's name, so identity is
in-text); people with missing/short leads fall back to "{title}. {description}".
1024-dim float embeddings, input_type="clustering" (family convention -- do not
"fix" the mismatch with Toponymy's internal embedder, see movie-madness-map).

Checkpoints a partial matrix every 20 batches so an interrupted run resumes;
the checkpoint carries a signature (model/dim/count/first+last pageid) and is
discarded if the input no longer matches.

Input:  data/people_texts.parquet
Output: data/embeddings_lead.npz [emb (N,1024) float32, pageid (N,) int64, sig]
"""

import os
import time

import cohere
import numpy as np
import pandas as pd

import config
from util import save_npz_atomic

PROGRESS_NPZ = config.DATA_DIR / f".embed_{config.EMBED_VARIANT}_progress.npz"
CHECKPOINT_EVERY = 20  # batches


def build_embed_text(row) -> str:
    lead = (row["lead"] or "").strip()
    if len(lead) >= 40:
        return lead
    description = (row["description"] or "").strip()
    return f"{row['title']}. {description}" if description else row["title"]


def signature(pageids: np.ndarray) -> str:
    return f"{config.EMBED_MODEL}_{config.EMBED_DIM}_{len(pageids)}_{pageids[0]}_{pageids[-1]}_{config.EMBED_VARIANT}"


def load_progress(sig: str, n: int) -> tuple[np.ndarray, int]:
    if PROGRESS_NPZ.exists():
        saved = np.load(PROGRESS_NPZ, allow_pickle=False)
        if str(saved["sig"]) == sig:
            done = int(saved["done"])
            print(f"Resuming from {done:,}/{n:,}")
            emb = np.zeros((n, config.EMBED_DIM), dtype=np.float32)
            emb[:done] = saved["emb"][:done]
            return emb, done
        print("Progress signature mismatch; starting fresh.")
    return np.zeros((n, config.EMBED_DIM), dtype=np.float32), 0


def embed_batch(client: cohere.ClientV2, texts: list[str], max_retries: int = 6) -> np.ndarray:
    for attempt in range(max_retries):
        try:
            resp = client.embed(
                texts=texts,
                model=config.EMBED_MODEL,
                input_type="clustering",
                output_dimension=config.EMBED_DIM,
                embedding_types=["float"],
            )
            return np.asarray(resp.embeddings.float_, dtype=np.float32)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = min(2**attempt * 5, 60)
            print(f"Embed batch failed ({exc}); retry {attempt + 1}/{max_retries} in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main() -> None:
    people = pd.read_parquet(config.TEXTS_PARQUET)
    texts = people.apply(build_embed_text, axis=1).tolist()
    pageids = people["pageid"].to_numpy(dtype=np.int64)
    sig = signature(pageids)
    n = len(texts)

    client = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])
    emb, done = load_progress(sig, n)

    n_batches_done = 0
    while done < n:
        batch = texts[done : done + config.EMBED_BATCH]
        emb[done : done + len(batch)] = embed_batch(client, batch)
        done += len(batch)
        n_batches_done += 1
        if n_batches_done % CHECKPOINT_EVERY == 0:
            save_npz_atomic(PROGRESS_NPZ, emb=emb, done=np.int64(done), sig=np.str_(sig))
            print(f"Embedded {done:,}/{n:,}", flush=True)

    save_npz_atomic(config.embeddings_npz(), emb=emb, pageid=pageids, sig=np.str_(sig))
    print(f"Wrote {config.embeddings_npz()} {emb.shape}")
    if PROGRESS_NPZ.exists():
        PROGRESS_NPZ.unlink()


if __name__ == "__main__":
    main()
