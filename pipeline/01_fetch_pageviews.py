"""Stage 01: download monthly pageview_complete dumps, filter to en.wikipedia.

For each month in config.PAGEVIEW_MONTHS:
  1. curl the ~6 GB pageviews-YYYYMM-user.bz2 (resumable: curl -C -; the
     -user variant is human traffic only, bots/spiders already excluded)
  2. stream bzip2 -dc | grep '^en\\.wikipedia ' and sum views per title,
     collapsing the desktop / mobile-web / mobile-app rows
  3. write data/pageviews/pageviews_YYYYMM.parquet [title, views]
  4. delete the .bz2 (config.KEEP_RAW_DUMPS)

The per-month parquet is the resume unit: months whose parquet exists are
skipped. PAGEVIEW_WORKERS months run concurrently -- threads are fine because
curl/bzip2/grep subprocesses do the heavy lifting.

Dump line format: {domain} {title} {page_id} {access_type} {monthly_total} {daily_encoding}
Titles are underscored; titles containing double quotes are CSV-style quoted
(handled in util.parse_dump_title, applied per unique title at the end).
The page_id field is unreliable (jeopardy-wikipedia-map lesson) -- never join on it.
"""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import config
from util import parse_dump_title, write_parquet_atomic


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url}", flush=True)
    t0 = time.time()
    cmd = [
        "curl",
        "-fsSL",
        "--retry",
        "5",
        "--retry-delay",
        "15",
        "-C",
        "-",  # resume partial downloads
        "--speed-limit",
        "10000",
        "--speed-time",
        "120",  # abort if <10 KB/s for 2 min (then curl retries)
        "-A",
        config.USER_AGENT,
        "-o",
        str(dest),
        url,
    ]
    subprocess.run(cmd, check=True)
    gb = dest.stat().st_size / 1e9
    print(f"Downloaded {dest.name}: {gb:.1f} GB in {(time.time() - t0) / 60:.1f} min", flush=True)


def filter_and_aggregate(bz2_path: Path, month: str) -> dict[bytes, int]:
    """Stream-decompress, keep en.wikipedia rows, sum views per raw title.

    Keys stay bytes in the hot loop (~50-80M rows); decode once per unique
    title afterwards.
    """
    counts: dict[bytes, int] = {}
    bunzip = subprocess.Popen(["bzip2", "-dc", str(bz2_path)], stdout=subprocess.PIPE)
    grep = subprocess.Popen(["grep", "^en\\.wikipedia "], stdin=bunzip.stdout, stdout=subprocess.PIPE)
    bunzip.stdout.close()  # let bzip2 get SIGPIPE if grep dies
    n_rows = 0
    assert grep.stdout is not None
    for line in grep.stdout:
        parts = line.rstrip(b"\n").split(b" ")
        if len(parts) < 5:
            continue
        try:
            views = int(parts[4])
        except ValueError:
            continue
        title = parts[1]
        counts[title] = counts.get(title, 0) + views
        n_rows += 1
        if n_rows % 10_000_000 == 0:
            print(f"{month}: {n_rows / 1e6:.0f}M en.wikipedia rows processed", flush=True)
    grep.wait()
    bunzip.wait()
    if bunzip.returncode != 0:
        raise RuntimeError(f"{month}: bzip2 exited {bunzip.returncode}")
    if grep.returncode != 0:  # 1 = zero matches, which is also fatal for us
        raise RuntimeError(f"{month}: grep exited {grep.returncode}")
    print(f"{month}: {n_rows:,} rows -> {len(counts):,} distinct titles", flush=True)
    return counts


def process_month(month: str) -> str:
    out_path = config.pageview_month_parquet(month)
    if out_path.exists():
        return f"{month}: parquet exists, skipping"
    year, mm = month.split("-")
    url = config.PAGEVIEW_URL_TEMPLATE.format(year=year, month=mm)
    bz2_path = config.PAGEVIEWS_DIR / f"pageviews-{year}{mm}-user.bz2"
    config.PAGEVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    download(url, bz2_path)
    counts = filter_and_aggregate(bz2_path, month)
    df = pd.DataFrame(
        {
            "title": [parse_dump_title(t.decode("utf-8", errors="replace")) for t in counts.keys()],
            "views": list(counts.values()),
        }
    )
    df = df.groupby("title", as_index=False)["views"].sum()  # unquoting can merge title variants
    df = df.sort_values("views", ascending=False).reset_index(drop=True)
    write_parquet_atomic(df, out_path)
    if not config.KEEP_RAW_DUMPS:
        bz2_path.unlink()
    return f"{month}: {len(df):,} titles, {df['views'].sum():,} total views"


def main() -> None:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=config.PAGEVIEW_WORKERS) as pool:
        for result in pool.map(process_month, config.PAGEVIEW_MONTHS):
            print(result, flush=True)
    print(f"All {len(config.PAGEVIEW_MONTHS)} months done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
