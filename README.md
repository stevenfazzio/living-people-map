# living-people-map

A semantic map of the most famous living people, built from English Wikipedia.
Fame proxy: median monthly enwiki pageviews over a trailing 12-month window.
Sibling project of movie-madness-map, steam-atlas, and jeopardy-wikipedia-map.

**Status: work in progress** — data acquisition stages only, so far.

## Pipeline

Run stages in order with `uv run python pipeline/NN_*.py`. All knobs live in
`pipeline/config.py` (no CLI args). Every stage is resumable; `data/` is
gitignored and treated as expensive to regenerate.

| Stage | Script | Does | Output |
|---|---|---|---|
| 00 | `00_fetch_roster.py` | Enumerate Category:Living people (+QIDs) via action API | `data/roster.parquet` |
| 01 | `01_fetch_pageviews.py` | Download 12 monthly pageview dumps, filter to enwiki | `data/pageviews/pageviews_YYYYMM.parquet` |
| 01b | `01b_fetch_redirects.py` | Build redirect map from page/redirect SQL dumps | `data/redirects.parquet` |
| 02 | `02_rank_people.py` | Resolve redirects, join, rank by median monthly views | `data/people_ranked.parquet`, `data/people_top100k.parquet` |
| 03 | `03_fetch_texts.py` | Leads + short descriptions + thumbnails (top `MAP_N`) | `data/people_texts.parquet` |
| 04 | `04_fetch_wikidata.py` | Occupation / citizenship / birth year / gender facets | `data/people_wikidata.parquet` |
| 05 | `05_embed.py` | Cohere embed-v4.0 lead embeddings (1024-dim) | `data/embeddings_lead.npz` |
| 06 | `06_reduce_umap.py` | UMAP to 2D (family params, seeded) | `data/umap_coords_lead.npz` |
| 07 | `07_label_topics.py` | Toponymy hierarchical region names (Claude Sonnet) | `data/labels_lead.parquet` |
| 08 | `08_visualize.py` | DataMapPlot interactive HTML | `data/living_people_map_lead.html` → `docs/index.html` |

View locally: `python3 -m http.server 8742 --bind 127.0.0.1 -d docs` →
http://127.0.0.1:8742/index.html (never open via `file://`).

## Setup

```
make install   # uv sync --extra dev
make test
```

No API keys needed for stages 00–02 (see `.env.example` for later stages).

## Data sources & attribution

- [English Wikipedia](https://en.wikipedia.org/) article metadata and text,
  CC BY-SA 4.0.
- [Wikimedia pageview_complete dumps](https://dumps.wikimedia.org/other/pageview_complete/)
  (user traffic only).
