# living-people-map

A semantic map of the most famous living people, built from Wikipedia. Fame
proxy: median monthly pageviews over a trailing 12-month window (2025-07 ..
2026-06). Sibling project of movie-madness-map, steam-atlas, and
jeopardy-wikipedia-map.

**Live: <https://stevenfazzio.github.io/living-people-map/>**

## The maps

25,000 people, positioned by the semantic similarity of their Wikipedia lead
paragraphs — Cohere `embed-v4.0` → UMAP → [Toponymy](https://github.com/TutteInstitute/toponymy)
region names → DataMapPlot.

| Map | Roster | Embeddings |
|---|---|---|
| [(landing page)](https://stevenfazzio.github.io/living-people-map/) | Global | **gender-erased** (centroid) |
| [global_nogender_leace](https://stevenfazzio.github.io/living-people-map/global_nogender_leace.html) | Global | **gender-erased** (LEACE) |
| [global](https://stevenfazzio.github.io/living-people-map/global.html) | Global | unmodified |
| [enwiki](https://stevenfazzio.github.io/living-people-map/enwiki.html) | English Wikipedia | unmodified |

The **global** roster ranks people by attention across all 346 Wikipedia
language editions; the **enwiki** roster ranks by English Wikipedia views
alone (an anglosphere lens). Their top-25k rosters overlap 76.5%.

## Gender erasure

On the unmodified map, gender is the single strongest organizer of the
layout: 15-nearest-neighbor gender purity in 2D is **0.904**, against a
**0.548** gender-blind baseline. Stage 05b removes gender from the embeddings
linearly, two ways, before the UMAP step — difference-of-means centroid
projection, and [LEACE](https://github.com/EleutherAI/concept-erasure).

Both drop a held-out linear gender probe from **AUC 1.00 to ~0.49** (chance)
and map purity from 0.904 to **~0.65**. For a binary concept LEACE is rank-1,
so the centroid direction is essentially the entire linearly-decodable
subspace — which is why the two erased maps agree so closely.

The residual 0.65 is nonlinear signal that a linear erasure cannot reach:
leads still say "actress", still use gendered pronouns, still describe
gender-skewed roles. The region names de-gender on their own, though — a
single "American Actors" continent replaces the actress/actor split.

Reproduce the numbers with `uv run python analysis/gender_erasure_metrics.py`.

> Probe-protocol gotcha, in case it saves you an afternoon: fit the eraser on
> the **train split only**. Fitting on the full set before splitting yields
> systematically *anti*-predictive held-out AUC, because the noise directions'
> class gaps must cancel between splits.

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
| 03 | `03_fetch_texts.py` | Leads + short descriptions + thumbnails (top `MAP_N`) | `data/people_texts_*.parquet` |
| 04 | `04_fetch_wikidata.py` | Occupation / citizenship / birth year / gender facets | `data/people_wikidata_*.parquet` |
| 05 | `05_embed.py` | Cohere embed-v4.0 lead embeddings (1024-dim) | `data/embeddings_*_lead.npz` |
| 05b | `05b_erase_gender.py` | Linear gender erasure (centroid + LEACE), no API cost | `data/embeddings_*_lead_nogender_*.npz` |
| 06 | `06_reduce_umap.py` | UMAP to 2D (family params, seeded) | `data/umap_coords_*.npz` |
| 07 | `07_label_topics.py` | Toponymy hierarchical region names (Claude Sonnet) | `data/labels_*.parquet` |
| 08 | `08_visualize.py` | DataMapPlot interactive HTML | `docs/*.html` (see below) |

Stages 03–08 run once per (`ROSTER_VARIANT`, `EMBED_VARIANT`) pair in
`pipeline/config.py`; fetch checkpoints are shared, so a second variant only
fetches people the first didn't cover. Stage 08 names its output after the
variant pair, except that the published landing page — `config.SITE_LANDING`,
currently global + centroid-erased — owns `docs/index.html`:

| `ROSTER_VARIANT` | `EMBED_VARIANT` | Output |
|---|---|---|
| `global` | `lead_nogender_centroid` | `docs/index.html` |
| `global` | `lead_nogender_leace` | `docs/global_nogender_leace.html` |
| `global` | `lead` | `docs/global.html` |
| `enwiki` | `lead` | `docs/enwiki.html` |

`docs/` is served directly by GitHub Pages, so the committed HTML *is* the
published site.

View locally: `python3 -m http.server 8742 --bind 127.0.0.1 -d docs` →
<http://127.0.0.1:8742/> (never open via `file://`).

## Setup

```
make install   # uv sync --extra dev
make test
```

No API keys needed for stages 00–02b; stages 05 and 07 need Cohere and
Anthropic keys respectively (see `.env.example`).

## Data sources & attribution

- [English Wikipedia](https://en.wikipedia.org/) article metadata and text,
  CC BY-SA 4.0.
- [Wikimedia pageview_complete dumps](https://dumps.wikimedia.org/other/pageview_complete/)
  (user traffic only).
- [Wikidata](https://www.wikidata.org/) facets (occupation, citizenship,
  birth year, gender), CC0.
