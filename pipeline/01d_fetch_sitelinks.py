"""Stage 01d: per-language Wikipedia titles for every roster person.

Streams the wb_items_per_site SQL dump (~2 GB gz, every Wikidata sitelink)
and keeps rows whose item is in our roster and whose site is a Wikipedia
language edition ({lang}wiki, excluding Commons/Meta/etc. special sites).
This is the (site, title) -> QID lookup that lets stage 01e credit pageviews
in ANY language to the right person.

Titles are stored underscored (the pageview dumps' form); wb_items_per_site
stores them with spaces.

Output: data/sitelinks.parquet [qid, site, title]  (~10-15M rows)
"""

import gzip
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

import config
from util import write_parquet_atomic

# (ips_row_id, ips_item_id, 'ips_site_id', 'ips_site_page')
ROW_RE = re.compile(r"\((\d+),(\d+),'((?:[^'\\]|\\.)*)','((?:[^'\\]|\\.)*)'\)")
SQL_ESCAPE_RE = re.compile(r"\\(.)")
SITE_RE = re.compile(r"^[a-z][a-z0-9_]*wiki$")
NON_LANGUAGE_WIKIS = {
    "commonswiki",
    "specieswiki",
    "wikidatawiki",
    "metawiki",
    "mediawikiwiki",
    "sourceswiki",
    "foundationwiki",
    "outreachwiki",
    "incubatorwiki",
    "wikifunctionswiki",
    "wikimaniawiki",
    "loginwiki",
    "votewiki",
    "testwiki",
    "test2wiki",
    "testwikidatawiki",
}


def download() -> Path:
    dest = config.DATA_DIR / "dumps" / "wikidatawiki-latest-wb_items_per_site.sql.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"{dest.name}: already downloaded")
        return dest
    print(f"Downloading {config.WB_ITEMS_PER_SITE_URL}", flush=True)
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
            config.WB_ITEMS_PER_SITE_URL,
        ],
        check=True,
    )
    print(f"Downloaded {dest.stat().st_size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min", flush=True)
    return dest


def main() -> None:
    roster = pd.read_parquet(config.ROSTER_PARQUET, columns=["qid"])
    roster_ids = {int(q[1:]) for q in roster["qid"].dropna()}
    print(f"Roster: {len(roster_ids):,} QIDs")

    dump_path = download()

    rows = []
    n_lines = 0
    with gzip.open(dump_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue
            n_lines += 1
            if n_lines % 500 == 0:
                print(f"{n_lines} INSERT lines, {len(rows):,} roster sitelinks", flush=True)
            for m in ROW_RE.finditer(line):
                item_id = int(m.group(2))
                if item_id not in roster_ids:
                    continue
                site = m.group(3)
                if not SITE_RE.match(site) or site in NON_LANGUAGE_WIKIS:
                    continue
                title = SQL_ESCAPE_RE.sub(r"\1", m.group(4)).replace(" ", "_")
                rows.append((f"Q{item_id}", site, title))

    df = pd.DataFrame(rows, columns=["qid", "site", "title"])
    print(f"Sitelinks: {len(df):,} rows, {df['site'].nunique()} sites, {df['qid'].nunique():,} people")
    print(f"People with an enwiki sitelink: {(df['site'] == 'enwiki').sum():,}")
    write_parquet_atomic(df, config.SITELINKS_PARQUET)
    print(f"Wrote {config.SITELINKS_PARQUET}")

    if not config.KEEP_RAW_DUMPS:
        dump_path.unlink()


if __name__ == "__main__":
    main()
