"""Stage 01e: all-language monthly pageviews for roster people.

Same monthly pageview_complete dumps as stage 01, but instead of keeping only
en.wikipedia rows, every Wikipedia language edition's rows are matched against
the stage-01d sitelink map ((site, title) -> QID) and credited to the person.
enwiki redirect titles are folded into the lookup (stage 01b map) so the
English component matches stage 02's redirect-resolved metric; other
languages count canonical titles only -- a documented undercount.

Per month the ~6 GB -user.bz2 is downloaded again (stage 01 deleted them),
filtered via bzip2|grep to Wikipedia rows (~150-250M/month), aggregated per
(person, language), and deleted. The per-month parquet is the resume unit.

Output: data/pageviews_global/global_YYYYMM.parquet [qid_int, lang, views]
"""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import config
from util import parse_dump_title, write_parquet_atomic

# pageview dump wiki codes use hyphens; Wikidata site ids use underscores,
# plus a few legacy ids that don't follow the rule.
SPECIAL_SITE_IDS = {"be-tarask": "be_x_oldwiki", "gsw": "alswiki", "lzh": "zh_classicalwiki", "nan": "zh_min_nanwiki"}


def build_lookup() -> dict[bytes, int]:
    """(site + ' ' + underscored_title) -> numeric QID, for all roster sitelinks
    plus enwiki redirect sources pointing at roster people."""
    sitelinks = pd.read_parquet(config.SITELINKS_PARQUET)
    lookup: dict[bytes, int] = {}
    for qid, site, title in zip(sitelinks["qid"], sitelinks["site"], sitelinks["title"]):
        lookup[f"{site} {title}".encode()] = int(qid[1:])
    print(f"Lookup: {len(lookup):,} sitelinks", flush=True)

    redirects = pd.read_parquet(config.REDIRECTS_PARQUET)
    enwiki_qid = {key[7:]: qid for key, qid in lookup.items() if key.startswith(b"enwiki ")}  # title bytes -> qid
    n_redirects = 0
    for source, target in zip(redirects["source_title"], redirects["target_title"]):
        qid = enwiki_qid.get(target.encode())
        if qid is not None:
            lookup.setdefault(b"enwiki " + source.encode(), qid)
            n_redirects += 1
    print(f"Lookup: +{n_redirects:,} enwiki redirect titles -> {len(lookup):,} total", flush=True)
    return lookup


def download(url: str, dest: Path) -> None:
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
            "--speed-limit",
            "10000",
            "--speed-time",
            "120",
            "-A",
            config.USER_AGENT,
            "-o",
            str(dest),
            url,
        ],
        check=True,
    )
    gb = dest.stat().st_size / 1e9
    print(f"Downloaded {dest.name}: {gb:.1f} GB in {(time.time() - t0) / 60:.1f} min", flush=True)


def site_for_wiki_code(lang: str) -> str:
    return SPECIAL_SITE_IDS.get(lang, lang.replace("-", "_") + "wiki")


def filter_and_aggregate(bz2_path: Path, month: str, lookup: dict[bytes, int]) -> dict[tuple[int, str], int]:
    counts: dict[tuple[int, str], int] = {}
    site_cache: dict[bytes, tuple[bytes, str]] = {}  # wiki code -> (site prefix bytes, lang str)
    bunzip = subprocess.Popen(["bzip2", "-dc", str(bz2_path)], stdout=subprocess.PIPE)
    grep = subprocess.Popen(["grep", "-E", "^[a-z-]+\\.wikipedia "], stdin=bunzip.stdout, stdout=subprocess.PIPE)
    bunzip.stdout.close()
    n_rows, n_matched = 0, 0
    assert grep.stdout is not None
    for line in grep.stdout:
        parts = line.rstrip(b"\n").split(b" ")
        if len(parts) < 5:
            continue
        try:
            views = int(parts[4])
        except ValueError:
            continue
        n_rows += 1
        if n_rows % 20_000_000 == 0:
            print(f"{month}: {n_rows / 1e6:.0f}M rows, {n_matched / 1e6:.1f}M matched", flush=True)
        wiki = parts[0]
        cached = site_cache.get(wiki)
        if cached is None:
            lang = wiki[: -len(b".wikipedia")].decode()
            cached = (site_for_wiki_code(lang).encode() + b" ", lang)
            site_cache[wiki] = cached
        site_prefix, lang = cached
        title = parts[1]
        if title.startswith(b'"'):
            title = parse_dump_title(title.decode("utf-8", errors="replace")).encode()
        qid = lookup.get(site_prefix + title)
        if qid is None:
            continue
        n_matched += 1
        key = (qid, lang)
        counts[key] = counts.get(key, 0) + views
    grep.wait()
    bunzip.wait()
    if bunzip.returncode != 0:
        raise RuntimeError(f"{month}: bzip2 exited {bunzip.returncode}")
    if grep.returncode != 0:
        raise RuntimeError(f"{month}: grep exited {grep.returncode}")
    print(
        f"{month}: {n_rows:,} wikipedia rows -> {n_matched:,} matched -> {len(counts):,} (person,lang) pairs",
        flush=True,
    )
    return counts


def make_process_month(lookup: dict[bytes, int]):
    def process_month(month: str) -> str:
        out_path = config.global_month_parquet(month)
        if out_path.exists():
            return f"{month}: parquet exists, skipping"
        year, mm = month.split("-")
        url = config.PAGEVIEW_URL_TEMPLATE.format(year=year, month=mm)
        bz2_path = config.GLOBAL_PAGEVIEWS_DIR / f"pageviews-{year}{mm}-user.bz2"
        config.GLOBAL_PAGEVIEWS_DIR.mkdir(parents=True, exist_ok=True)
        download(url, bz2_path)
        counts = filter_and_aggregate(bz2_path, month, lookup)
        df = pd.DataFrame(
            {
                "qid_int": [k[0] for k in counts.keys()],
                "lang": [k[1] for k in counts.keys()],
                "views": list(counts.values()),
            }
        )
        write_parquet_atomic(df, out_path)
        if not config.KEEP_RAW_DUMPS:
            bz2_path.unlink()
        return f"{month}: {len(df):,} (person,lang) rows, {df['views'].sum():,} views"

    return process_month


def main() -> None:
    t0 = time.time()
    lookup = build_lookup()
    with ThreadPoolExecutor(max_workers=config.PAGEVIEW_WORKERS) as pool:
        for result in pool.map(make_process_month(lookup), config.PAGEVIEW_MONTHS):
            print(result, flush=True)
    print(f"All {len(config.PAGEVIEW_MONTHS)} months done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
