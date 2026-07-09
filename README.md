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
| 01d | `01d_fetch_sitelinks.py` | Per-language titles for roster people (wb_items_per_site) | `data/sitelinks.parquet` |
| 01e | `01e_fetch_pageviews_global.py` | All-language monthly views per person | `data/pageviews_global/global_YYYYMM.parquet` |
| 02 | `02_rank_people.py` | Resolve redirects, join, rank by median monthly enwiki views | `data/people_ranked.parquet`, `data/people_top100k.parquet` |
| 02b | `02b_rank_global.py` | Rank by all-language median; language facets | `data/people_ranked_global.parquet`, `data/people_top25k_global.parquet` |
| 03 | `03_fetch_texts.py` | Leads + short descriptions + thumbnails (top `MAP_N`) | `data/people_texts.parquet` |
| 04 | `04_fetch_wikidata.py` | Occupation / citizenship / birth year / gender facets | `data/people_wikidata.parquet` |
| 05 | `05_embed.py` | Cohere embed-v4.0 lead embeddings (1024-dim) | `data/embeddings_lead.npz` |
| 06 | `06_reduce_umap.py` | UMAP to 2D (family params, seeded) | `data/umap_coords_lead.npz` |
| 07 | `07_label_topics.py` | Toponymy hierarchical region names (Claude Sonnet) | `data/labels_lead.parquet` |
| 08 | `08_visualize.py` | DataMapPlot interactive HTML | `docs/index.html` (enwiki) / `docs/global.html` (global) |

Stages 03–08 run once per `ROSTER_VARIANT` ("enwiki" or "global" in
`pipeline/config.py`); fetch checkpoints are shared, so the second variant
only fetches people the first didn't cover.

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
