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
- **2026-07-09 — global (all-language) roster variant**: QRank would have
  been the easy multilingual fame metric but is abandoned (file frozen
  2024-03-16 despite the site claiming updates), so stages 01d/01e/02b build
  it from the same 12 monthly dumps: wb_items_per_site sitelinks map every
  roster QID to its title in all 346 language editions, and per-(person,
  language, month) views feed the same median metric. `ROSTER_VARIANT` in
  config switches stages 03+ between rosters; artifacts are suffixed, maps
  publish to docs/index.html (enwiki) and docs/global.html. Non-English
  redirect/rename views are NOT resolved (enwiki's are) — documented
  undercount. Top-25k overlap between the two rosters: 76.5%.
- **2026-07-09 — gender-erasure variants (stage 05b)**: gender was the
  map's strongest organizer (2D 15-NN gender purity 0.904 vs 0.548
  gender-blind baseline). Centroid (difference-of-means projection) and
  LEACE erasures both drop a held-out linear probe from AUC 1.00 to ~0.49
  and map purity to ~0.65 — for a binary concept LEACE is rank-1, so the
  centroid direction ≈ the whole linearly-decodable subspace. Residual 0.65
  purity = nonlinear text signal ("actress", pronouns, roles). Toponymy
  names de-gendered on their own (one "American Actors" continent replaces
  the actress/actor split). Probe protocol gotcha: fit the eraser on the
  TRAIN split only — fitting on the full set before splitting yields
  systematically anti-predictive held-out AUC (noise directions' class gaps
  must cancel between splits).
- **2026-07-09 — stage 04 read TOP_PARQUET instead of map_roster_parquet()**
  after the variant split, so the global map's first render had enwiki
  people's facets (5,872 newcomers all "Unknown"). Fixed + refetched; when
  adding a variant knob, grep EVERY stage for the paths it switches.
- **2026-07-09 — no case-folding in the pageview join**: the global metric
  disagreeing with the enwiki metric exposed that title_key's lowercasing
  let people absorb unrelated same-name-different-case pages (TeQuila ←
  "Tequila" the drink). Only 5 of the enwiki top-25k were fake; fixed by
  joining exact canonical titles.
- **2026-07-09 — full redirect resolution (stage 01b)**: the naive title join
  zeroed out Vijay (Tamil megastar) because his article was renamed
  "Vijay (actor)" → "C. Joseph Vijay" mid-window, splitting his view history
  across titles. Stage 01b builds source→target from the `page` + `redirect`
  SQL dumps (~11M ns0 redirects); stage 02 resolves every dump title through
  it before keying. This also fixes the nickname-redirect undercount
  ("The Rock" etc.), which is no longer an accepted limitation.
- **2026-08-04 — published public; the erased map leads**: repo is public at
  github.com/stevenfazzio/living-people-map, Pages serves `docs/` from main.
  `docs/` is published as-is, so one variant has to own `index.html`:
  `config.SITE_LANDING` = (global, lead_nogender_centroid) claims it and the
  enwiki base map moved to `docs/enwiki.html`. Centroid over LEACE because
  the two are near-identical here (rank-1 concept) and difference-of-means is
  the simpler method to explain to a visitor. Change `SITE_LANDING` — not the
  filenames — to re-point the landing page, or stage 08 will recreate the old
  layout on the next render.
- **2026-08-04 — only two maps are published; history purged**: rendered maps
  are ~8 MB each and four of them were adding ~34 MB of blobs per re-render
  (82 MB across history by the second one). `docs/` now tracks only
  `index.html` (centroid-erased) and `global.html` — kept deliberately as the
  *control*, since the erasure claim is unverifiable by eye without the
  un-erased map beside it. `enwiki.html` and `global_nogender_leace.html` were
  dropped (LEACE lands within noise of centroid; enwiki is a roster side-note)
  and purged from history with git-filter-repo + force-push — safe only
  because the repo was hours old with 0 clones/forks. **That window is closed
  now: never rewrite this history again, untrack going forward instead.**
  .git went 70 MB → 26 MB. Note the purge is *path*-based, so it is not total:
  in c8892a5 and 73ce556 the enwiki map lived at `docs/index.html` (the rename
  to `enwiki.html` came later), and ~16 MB of it still sits there under that
  old path. Stripping it would need a blob-id filter and would leave the
  initial commit's "+ first rendered map" message pointing at nothing, so it
  was left alone deliberately.
  `.gitignore` allowlists `docs/index.html` + `docs/global.html` against
  `docs/*.html`, so rendering another variant can't accidentally be committed.
  Nothing was lost — `data/living_people_map_*.html` holds every variant and
  stage 08 re-renders any of them in ~20 s.
- **2026-08-04 — datamapplot pinned to a git rev, not PyPI**: PyPI's latest is
  0.7.3 (2026-05-31) and two merges we need postdate it with no release since,
  so `[tool.uv.sources]` pins main's HEAD `5cf47aa5`. That SHA *is* the #206
  merge commit, so nothing unreviewed rides along. #206 (tap-to-inspect) is
  the one that matters: stage 08 sets `on_click`, so before it a phone tap
  jumped straight to Wikipedia and the hover card was unreachable on touch —
  a real defect for a site mostly opened on phones. #197 gives
  `MAP_SCROLL_ZOOM_SPEED` (0.05; the 0.01 default is sluggish over 25k
  points). **Gotcha: main still self-reports `version = "0.7.3"`, so
  `datamapplot.__version__` cannot tell you whether you're on the release or
  the pin — trust the rev in `pyproject.toml`.** Second gotcha:
  `render_html` is wrapped by the config decorator, which does not set
  `__wrapped__`, so `inspect.signature` shows `*args, **kwargs` and makes new
  params look MISSING — grep the installed source instead. Drop the pin when
  0.7.4 ships.
- **2026-08-04 — Pages URL is stevenfazzio.com, NOT github.io**: the account
  has a user-level custom domain, so project sites serve from
  `https://stevenfazzio.com/<repo>/` (same as steam-atlas, jeopardy-map,
  movie-madness-map). `gh api repos/:owner/:repo/pages --jq .html_url` is the
  authority; don't assume the github.io form when writing README links.

## Wikipedia data gotchas (mostly inherited from jeopardy-wikipedia-map)

- **Join on EXACT canonical titles** (underscores→spaces only) after
  redirect resolution. Never case-fold: distinct pages can differ only by
  case, and lowercasing let rapper "TeQuila" absorb "Tequila" the drink's
  views (inflating him to enwiki rank ~7k; caught 2026-07-09 when the global
  metric disagreed). Case-variant redirects are real rows in the redirect
  table, so the redirect map already covers them.
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
