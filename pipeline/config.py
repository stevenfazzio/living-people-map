"""Central configuration for the living-people-map pipeline.

All knobs live here -- no CLI args (family convention shared with
movie-madness-map / steam-atlas / jeopardy-wikipedia-map). For smoke tests,
temporarily set MAX_ROSTER_REQUESTS or trim PAGEVIEW_MONTHS; don't add flags.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = "living-people-map"
PROJECT_TAGLINE = "A semantic map of the most famous living people"

# Wikimedia REQUIRES a descriptive User-Agent with real contact info --
# requests with a generic/placeholder UA get rate-limited (429) quickly.
USER_AGENT = f"{PROJECT_NAME}/0.1 (https://github.com/stevenfazzio/{PROJECT_NAME}; fazzios@gmail.com)"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PAGEVIEWS_DIR = DATA_DIR / "pageviews"

# --- Stage 00: roster (Category:Living people) ---
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
ROSTER_CATEGORY = "Category:Living people"  # ~1.15M pages as of 2026-07
ROSTER_BATCH_SIZE = 500  # gcmlimit; 500 is the non-bot maximum
ROSTER_SLEEP_S = 0.15  # pause between API requests (~2,300 total)
MAX_ROSTER_REQUESTS = None  # set to a small int for smoke tests
ROSTER_PAGES_JSONL = DATA_DIR / ".roster_pages.jsonl"  # append-only crawl checkpoint
ROSTER_PARQUET = DATA_DIR / "roster.parquet"

# --- Stage 01: monthly pageview dumps ---
# Trailing 12 complete months as of 2026-07. Each -user.bz2 (human traffic
# only, bots excluded) is ~6 GB; we keep only the filtered per-month parquet.
PAGEVIEW_MONTHS = [
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
]
PAGEVIEW_URL_TEMPLATE = (
    "https://dumps.wikimedia.org/other/pageview_complete/monthly/{year}/{year}-{month}/pageviews-{year}{month}-user.bz2"
)
PAGEVIEW_WORKERS = 2  # months processed concurrently (subprocesses do the heavy work)
KEEP_RAW_DUMPS = False  # delete each .bz2 after its parquet is written


def pageview_month_parquet(month: str) -> Path:
    """data/pageviews/pageviews_YYYYMM.parquet -- the per-month resume unit."""
    return PAGEVIEWS_DIR / f"pageviews_{month.replace('-', '')}.parquet"


# --- Stage 01b: redirect map (fixes renamed-article splits + alias undercounts) ---
REDIRECTS_PARQUET = DATA_DIR / "redirects.parquet"

# --- Stage 02: ranking ---
TOP_K = 100_000
RANKED_PARQUET = DATA_DIR / "people_ranked.parquet"  # full roster + fame metrics
TOP_PARQUET = DATA_DIR / f"people_top{TOP_K // 1000}k.parquet"  # TOP_K slice for downstream stages

# --- Stage 01d: Wikidata sitelinks (QID -> per-language titles) ---
WB_ITEMS_PER_SITE_URL = "https://dumps.wikimedia.org/wikidatawiki/latest/wikidatawiki-latest-wb_items_per_site.sql.gz"
SITELINKS_PARQUET = DATA_DIR / "sitelinks.parquet"  # roster people only, Wikipedia sites only

# --- Stage 01e: global (all-language) pageviews for roster people ---
GLOBAL_PAGEVIEWS_DIR = DATA_DIR / "pageviews_global"


def global_month_parquet(month: str) -> Path:
    return GLOBAL_PAGEVIEWS_DIR / f"global_{month.replace('-', '')}.parquet"


# --- Stages 03+: the map subset ---
MAP_N = 25_000  # chosen 2026-07-09 from the choose-N analysis (data/analysis/choose_n.html)

# --- Stage 02b: global ranking ---
GLOBAL_RANKED_PARQUET = DATA_DIR / "people_ranked_global.parquet"
GLOBAL_TOP_PARQUET = DATA_DIR / f"people_top{MAP_N // 1000}k_global.parquet"

# Which fame metric selects (and ranks) the MAP_N people downstream:
#   "enwiki" -- median monthly English Wikipedia views (stage 02)
#   "global" -- median monthly views across ALL Wikipedias (stage 02b)
# Stage 03+ artifacts are suffixed with this, so both maps coexist.
ROSTER_VARIANT = "global"


def map_roster_parquet() -> Path:
    return TOP_PARQUET if ROSTER_VARIANT == "enwiki" else GLOBAL_TOP_PARQUET


# --- Stage 03: leads, descriptions, thumbnails ---
# JSONL checkpoints are shared across roster variants (keyed by pageid), so a
# second variant only fetches people it hasn't seen; the parquets are per-variant.
TEXTS_PARQUET = DATA_DIR / f"people_texts_{ROSTER_VARIANT}.parquet"
EXTRACTS_JSONL = DATA_DIR / ".texts_extracts.jsonl"  # resumable checkpoints
PROPS_JSONL = DATA_DIR / ".texts_props.jsonl"
EXTRACTS_BATCH = 20  # exlimit max for exintro requests
PROPS_BATCH = 50  # pageimages/description batch (non-bot max)
THUMB_SIZE = 320  # px, hover-card thumbnail width
TEXT_SLEEP_S = 0.1

# --- Stage 04: Wikidata facets ---
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_PARQUET = DATA_DIR / f"people_wikidata_{ROSTER_VARIANT}.parquet"
WIKIDATA_JSONL = DATA_DIR / ".wikidata_claims.jsonl"
WB_BATCH = 50  # wbgetentities non-bot max

# --- Stage 05: embeddings ---
# "lead" is the base Cohere embedding; "lead_nogender_centroid" and
# "lead_nogender_leace" are stage-05b linear transforms of it (no API cost).
# Future experiment: "full" (whole articles).
EMBED_VARIANT = "lead_nogender_leace"
EMBED_MODEL = "embed-v4.0"
EMBED_DIM = 1024
EMBED_BATCH = 96


def embeddings_npz(variant: str = EMBED_VARIANT):
    return DATA_DIR / f"embeddings_{ROSTER_VARIANT}_{variant}.npz"


def umap_coords_npz(variant: str = EMBED_VARIANT):
    return DATA_DIR / f"umap_coords_{ROSTER_VARIANT}_{variant}.npz"


def labels_parquet(variant: str = EMBED_VARIANT):
    return DATA_DIR / f"labels_{ROSTER_VARIANT}_{variant}.parquet"


# --- Stage 06: UMAP (params deliberately match the sibling projects) ---
UMAP_KWARGS = dict(n_components=2, n_neighbors=15, min_dist=0.05, metric="cosine", random_state=42)

# --- Stage 07: Toponymy region labels ---
NAMER_MODEL = "claude-sonnet-4-6"
NAMER_CONCURRENCY = 24
TOPONYMY_TEXT_CHARS = 2000  # title + lead, truncated (steam-atlas lesson: semantic text, not summaries)

# --- Stage 08: visualize ---
MAP_HTML = DATA_DIR / f"living_people_map_{ROSTER_VARIANT}_{EMBED_VARIANT}.html"
DOCS_DIR = REPO_ROOT / "docs"
# docs/ is published as-is by GitHub Pages, so one variant has to own index.html.
# The published site leads with the gender-erased centroid map (2026-08-04); every
# other variant keeps its descriptive filename.
SITE_LANDING = ("global", "lead_nogender_centroid")
_docs_suffix = "" if EMBED_VARIANT == "lead" else "_" + EMBED_VARIANT.removeprefix("lead_")
_docs_name = "index" if (ROSTER_VARIANT, EMBED_VARIANT) == SITE_LANDING else f"{ROSTER_VARIANT}{_docs_suffix}"
DOCS_HTML = DOCS_DIR / f"{_docs_name}.html"
MAP_SUBTITLE = (
    f"The {MAP_N:,} most famous living people, by English Wikipedia attention"
    if ROSTER_VARIANT == "enwiki"
    else f"The {MAP_N:,} most famous living people, by attention across all Wikipedia languages"
)
if EMBED_VARIANT != "lead":
    MAP_SUBTITLE += " · gender-erased embeddings"
