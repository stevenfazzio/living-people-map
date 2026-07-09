"""Stage 01b: build the enwiki redirect map (source title -> target title).

Why this exists: pageview dumps record views under the title AS REQUESTED that
month. Two consequences for ranking:
  1. Renamed articles split their history across old and new titles -- e.g.
     "Vijay (actor)" -> "C. Joseph Vijay" mid-window zeroed out a top-500
     celebrity in the naive join.
  2. Popular nickname/alias redirects ("The Rock") never reach the target.
Resolving every dump title through the redirect table before joining fixes both.

Sources (streamed, parsed with a tuple regex, deleted after unless KEEP_RAW_DUMPS):
  enwiki-latest-page.sql.gz     (~2.2 GB) -> page_id -> (ns0 title, is_redirect)
  enwiki-latest-redirect.sql.gz (~160 MB) -> rd_from page_id -> ns0 target title

Output: data/redirects.parquet [source_title, target_title] (underscored, ns0->ns0)
"""

import gzip
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

import config
from util import write_parquet_atomic

# page: (page_id, page_namespace, 'page_title', page_is_redirect, ...)
PAGE_ROW_RE = re.compile(r"\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)',([01]),")
# redirect: (rd_from, rd_namespace, 'rd_title', 'rd_interwiki', 'rd_fragment')
REDIRECT_ROW_RE = re.compile(r"\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)'")
SQL_ESCAPE_RE = re.compile(r"\\(.)")


def unescape_sql(s: str) -> str:
    return SQL_ESCAPE_RE.sub(r"\1", s)


def download(name: str) -> Path:
    url = f"https://dumps.wikimedia.org/enwiki/latest/{name}"
    dest = config.DATA_DIR / "dumps" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"{name}: already downloaded")
        return dest
    print(f"Downloading {url}", flush=True)
    t0 = time.time()
    subprocess.run(
        [
            "curl",
            "-fsSL",
            "--retry",
            "5",
            "--retry-delay",
            "15",
            "-C",
            "-",
            "-A",
            config.USER_AGENT,
            "-o",
            str(dest),
            url,
        ],
        check=True,
    )
    print(f"Downloaded {name}: {dest.stat().st_size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min", flush=True)
    return dest


def parse_page_sql(path: Path) -> tuple[dict[int, str], set[int]]:
    """Return (ns0 page_id -> title, page_ids that are redirects)."""
    id_to_title: dict[int, str] = {}
    redirect_ids: set[int] = set()
    n_lines = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue
            n_lines += 1
            if n_lines % 500 == 0:
                print(f"page.sql: {n_lines} INSERT lines, {len(id_to_title):,} ns0 pages", flush=True)
            for m in PAGE_ROW_RE.finditer(line):
                if m.group(2) != "0":
                    continue
                page_id = int(m.group(1))
                id_to_title[page_id] = unescape_sql(m.group(3))
                if m.group(4) == "1":
                    redirect_ids.add(page_id)
    print(f"page.sql: {len(id_to_title):,} ns0 pages, {len(redirect_ids):,} of them redirects", flush=True)
    return id_to_title, redirect_ids


def parse_redirect_sql(path: Path) -> dict[int, str]:
    """Return rd_from page_id -> ns0 target title."""
    targets: dict[int, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue
            for m in REDIRECT_ROW_RE.finditer(line):
                if m.group(2) != "0":
                    continue
                targets[int(m.group(1))] = unescape_sql(m.group(3))
    print(f"redirect.sql: {len(targets):,} ns0-target redirects", flush=True)
    return targets


def main() -> None:
    page_path = download("enwiki-latest-page.sql.gz")
    redirect_path = download("enwiki-latest-redirect.sql.gz")

    id_to_title, redirect_ids = parse_page_sql(page_path)
    targets = parse_redirect_sql(redirect_path)

    rows = []
    for page_id in redirect_ids:
        source = id_to_title.get(page_id)
        target = targets.get(page_id)
        if source and target and source != target:
            rows.append((source, target))
    df = pd.DataFrame(rows, columns=["source_title", "target_title"])
    df = df.drop_duplicates(subset="source_title")
    print(f"Redirect map: {len(df):,} source->target pairs")
    write_parquet_atomic(df, config.REDIRECTS_PARQUET)
    print(f"Wrote {config.REDIRECTS_PARQUET}")

    if not config.KEEP_RAW_DUMPS:
        page_path.unlink()
        redirect_path.unlink()


if __name__ == "__main__":
    main()
