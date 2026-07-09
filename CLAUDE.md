# living-people-map — project notes

A datamap of the most famous living people, from English Wikipedia. Sibling of
movie-madness-map, steam-atlas, and jeopardy-wikipedia-map — follow the family
conventions (numbered pipeline stages, `pipeline/config.py` as the single
source of knobs, no CLI args, uv + ruff line-length 120, parquet for tables /
npz for vectors, atomic writes, `data/` gitignored and expensive).

## Working rules

- Smoke-test by lowering config knobs (`MAX_ROSTER_REQUESTS`, trimming
  `PAGEVIEW_MONTHS`), never by adding CLI flags.
- Never overwrite files in `data/` in place — `util.write_parquet_atomic`.
- Long runs: background them with output to `logs/` and check the log.
- Wikimedia etiquette: every request goes through `util.make_session()`
  (descriptive User-Agent with contact info — placeholder UAs get 429'd),
  `maxlag=5` on action-API calls, modest sleeps between requests.

## Decisions (dated)

- **2026-07-08 — fame metric**: rank by **median** of 12 monthly enwiki
  pageview totals (2025-07..2026-06), not the sum — robust to one-off news
  spikes. Sum/max/n_months kept as metadata; max/median is a fun "volatility"
  signal for a later colormap. Articles created mid-window are penalized by
  the zero-filled median; `median_present` recorded as context.
- **2026-07-08 — enwiki pageviews, not global**: we embed enwiki text, so we
  rank by enwiki views (anglosphere lens), not QRank/all-project views.
- **2026-07-08 — roster = Category:Living people** (~1.15M pages), fetched
  with `generator=categorymembers` + `prop=pageprops` so title, pageid, and
  Wikidata QID come in one crawl (~2,300 requests).
- **2026-07-08 — pageviews from monthly dumps**, not the per-article REST API
  (1M+ articles would mean 1M+ API calls). `pageview_complete/monthly/...-user.bz2`
  is human-only traffic, ~6 GB/month; we keep only the filtered per-month
  parquet (~8M titles) and delete the bz2.
- **2026-07-08 — N is TBD**: pull top 100k, choose N from the rank/views
  distribution + spot-checking names at candidate cutoffs.
- **2026-07-09 — full redirect resolution (stage 01b)**: the naive title join
  zeroed out Vijay (Tamil megastar) because his article was renamed
  "Vijay (actor)" → "C. Joseph Vijay" mid-window, splitting his view history
  across titles. Stage 01b builds source→target from the `page` + `redirect`
  SQL dumps (~11M ns0 redirects); stage 02 resolves every dump title through
  it before keying. This also fixes the nickname-redirect undercount
  ("The Rock" etc.), which is no longer an accepted limitation.

## Wikipedia data gotchas (mostly inherited from jeopardy-wikipedia-map)

- **`title_key`** (underscores→spaces, collapse whitespace, lowercase) is the
  only safe join key across the action API, pageview dumps, and anything else
  Wikimedia. Dump titles are underscored; API titles use spaces.
- The pageview dump's **page_id column is unreliable** — never join on it.
- Dump titles containing double quotes are **CSV-style quoted with backslash
  escapes** (`util.parse_dump_title`). Affects real people ("Weird Al" Yankovic).
- Dump rows are per access method (desktop / mobile-web / mobile-app) — sum
  the three rows per title.
- Redirect views land on the redirect title, not the target — and a mid-window
  article RENAME leaves the old months' views on the old title. Stage 02
  therefore resolves every dump title through the stage-01b redirect map
  before joining; title_key case-folding remains as a safety net.
- Roster snapshot time ≠ pageview window: someone who died recently is out of
  the category (correct — this is a *living* people map) even though they
  have huge views in the window.
- **DataMapPlot label layers go FINEST-FIRST** (`create_interactive_plot`
  docstring; `hierarchical_collision_priority` gives later = coarser layers
  the collision win). steam-atlas's stage 09 comment "DataMapPlot expects
  coarsest first" is WRONG for datamapplot 0.7.3 — passing coarsest-first
  inverts zoom gating and collision priority so fine labels dominate the
  overview zoom. Our parquet stores label_layer_0 = coarsest (natural for
  reading); stage 08 reverses at the call site.
- **np.savez appends `.npz` to any filename that lacks it** — an atomic-write
  temp file named `*.npz.tmp` gets the real data written to `*.npz.tmp.npz`
  while the rename moves an empty file into place. `util.save_npz_atomic`
  uses a `.tmp.npz` suffix and verifies the written keys before renaming
  (regression test in tests/test_util.py).
