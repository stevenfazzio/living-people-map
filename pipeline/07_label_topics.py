"""Stage 07: hierarchical region labels via Toponymy + Claude Sonnet.

Toponymy is map labeling for the embedding space: the default clusterer finds
density regions in the 2D UMAP coords (clusterable_vectors), names them with
the LLM using the high-dim embeddings + documents, and emits per-person labels
at multiple zoom levels ("Unlabelled" rows are the unnamed gaps of the map at
that scale -- keep them). Layer 0 of topic_names_/cluster_layers_ is the
FINEST; DataMapPlot wants coarsest first, so we reverse when saving.

Documents are title + description + lead (raw source text, not summaries --
steam-atlas ablation lesson), capped at TOPONYMY_TEXT_CHARS.

Inputs:  data/people_texts.parquet, embeddings_lead.npz, umap_coords_lead.npz
Output:  data/labels_lead.parquet [pageid, label_layer_0 (coarsest), ...]
"""

import os

import nest_asyncio  # Toponymy makes re-entrant asyncio.run() calls
import numpy as np
import pandas as pd
from toponymy import Toponymy, ToponymyClusterer
from toponymy.embedding_wrappers import CohereEmbedder
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()

import config  # noqa: E402
from util import write_parquet_atomic  # noqa: E402


def build_documents(df: pd.DataFrame) -> list[str]:
    docs = []
    for _, row in df.iterrows():
        parts = [row["title"]]
        if row.get("description"):
            parts.append(str(row["description"]))
        text = " -- ".join(parts)
        if row.get("lead"):
            text = f"{text}\n{row['lead']}"
        docs.append(text[: config.TOPONYMY_TEXT_CHARS])
    return docs


def main() -> None:
    people = pd.read_parquet(config.TEXTS_PARQUET)
    emb_data = np.load(config.embeddings_npz(), allow_pickle=False)
    coords_data = np.load(config.umap_coords_npz(), allow_pickle=False)
    embeddings, coords = emb_data["emb"], coords_data["coords"]
    assert (emb_data["pageid"] == people["pageid"].to_numpy()).all(), "embeddings/texts row order mismatch"
    assert (coords_data["pageid"] == people["pageid"].to_numpy()).all(), "coords/texts row order mismatch"
    print(f"Loaded {len(people):,} people, embeddings {embeddings.shape}, coords {coords.shape}")

    documents = build_documents(people)

    llm = AsyncAnthropicNamer(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=config.NAMER_MODEL,
        max_concurrent_requests=config.NAMER_CONCURRENCY,
    )
    embedder = CohereEmbedder(api_key=os.environ["CO_API_KEY"], model=config.EMBED_MODEL)
    clusterer = ToponymyClusterer(min_clusters=4)

    np.random.seed(42)
    clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)

    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="Wikipedia biography leads of living people",
        corpus_description="collection of the 25,000 most famous living people on English Wikipedia",
        exemplar_delimiters=['    * """', '"""\n'],
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    topic_model.fit(objects=documents, embedding_vectors=embeddings, clusterable_vectors=coords)

    n_layers = len(topic_model.cluster_layers_)
    if n_layers == 0:
        raise ValueError("Toponymy produced 0 cluster layers")
    print(f"Toponymy produced {n_layers} cluster layer(s)")

    labels = {"pageid": people["pageid"].reset_index(drop=True)}
    for i, layer in enumerate(reversed(topic_model.cluster_layers_)):  # coarsest first for DataMapPlot
        labels[f"label_layer_{i}"] = layer.topic_name_vector
        named = pd.Series(layer.topic_name_vector)
        print(f"label_layer_{i}: {named.nunique()} names, {(named == 'Unlabelled').mean():.0%} unlabelled")

    write_parquet_atomic(pd.DataFrame(labels), config.labels_parquet())
    print(f"Wrote {config.labels_parquet()}")


if __name__ == "__main__":
    main()
