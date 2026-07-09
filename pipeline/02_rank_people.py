"""Stage 02: join roster to monthly pageviews, compute fame metrics, rank.

Every dump title is first resolved through the redirect map (stage 01b):
views recorded under a redirect -- alias titles like "The Rock", and OLD
titles of renamed articles -- are credited to the canonical target. Without
this, a mid-window rename zeroes a person out entirely (the "Vijay (actor)"
-> "C. Joseph Vijay" incident). After resolution, the join key is
util.title_key (underscores->spaces, collapse whitespace, lowercase), since
the roster (API titles with spaces) and the dumps (underscored) disagree on
representation.

Metrics per person (a month with no row in the dump = 0 views):
  median_views    -- median of the monthly totals; THE ranking metric
                     (robust to one-off news spikes, unlike the sum)
  total_views, max_views
  n_months        -- months with >=1 view (~months the article existed)
  median_present  -- median over only those months (context for articles
                     created mid-window, which median_views penalizes)

Outputs:
  data/people_ranked.parquet  -- full roster + metrics + per-month columns
  data/people_top100k.parquet -- top TOP_K rows by median_views
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config
from util import title_key, write_parquet_atomic


def month_col(month: str) -> str:
    return f"views_{month.replace('-', '')}"


def load_roster(redirect_sources: set[str]) -> pd.DataFrame:
    roster = pd.read_parquet(config.ROSTER_PARQUET)
    # The category includes pages that are themselves redirects (people merged
    # into event/list articles, or renamed with the old categorized title left
    # behind). They aren't biographies; drop them.
    stale = roster["title"].str.replace(" ", "_").isin(redirect_sources)
    if stale.any():
        print(f"Dropping {int(stale.sum()):,} roster rows that are redirect pages, not articles.")
        roster = roster[~stale].reset_index(drop=True)
    roster["title_key"] = roster["title"].map(title_key)
    collisions = roster["title_key"].duplicated(keep=False)
    if collisions.any():
        n = int(collisions.sum())
        print(f"WARNING: {n} roster rows share a title_key with another row; views will double-count for these.")
        print(roster.loc[collisions, "title"].head(20).to_list())
    return roster


def main() -> None:
    # Dump-side work runs in pyarrow (C speed; a python title_key .map over
    # 30-57M titles/month would take hours). Normalization skips title_key's
    # whitespace collapse -- API titles never have doubled spaces, and a dump
    # title with doubled underscores simply won't match (negligible).
    redirects = pd.read_parquet(config.REDIRECTS_PARQUET)

    roster = load_roster(set(redirects["source_title"]))
    print(f"Roster: {len(roster):,} people")

    months_available = [m for m in config.PAGEVIEW_MONTHS if config.pageview_month_parquet(m).exists()]
    missing = set(config.PAGEVIEW_MONTHS) - set(months_available)
    if missing:
        print(f"WARNING: ranking on {len(months_available)}/12 months; missing {sorted(missing)}")

    redirect_src = pa.array(redirects["source_title"])
    redirect_tgt_keys = pc.utf8_lower(
        pc.replace_substring(pa.array(redirects["target_title"]), pattern="_", replacement=" ")
    )
    print(f"Redirect map: {len(redirects):,} entries")

    roster_keys = pa.array(roster["title_key"].unique())
    for month in months_available:
        table = pq.read_table(config.pageview_month_parquet(month), columns=["title", "views"])
        titles = table["title"].combine_chunks()
        keys_direct = pc.utf8_lower(pc.replace_substring(titles, pattern="_", replacement=" "))
        redirect_idx = pc.index_in(titles, value_set=redirect_src)
        keys = pc.if_else(pc.is_valid(redirect_idx), pc.take(redirect_tgt_keys, redirect_idx), keys_direct)
        mask = pc.is_in(keys, value_set=roster_keys)
        matched = pd.DataFrame(
            {
                "title_key": keys.filter(mask).to_pandas(),
                "views": table["views"].combine_chunks().filter(mask).to_pandas(),
            }
        )
        per_key = matched.groupby("title_key")["views"].sum()
        roster[month_col(month)] = roster["title_key"].map(per_key).fillna(0).astype("int64")
        print(f"{month}: matched {(roster[month_col(month)] > 0).sum():,} people", flush=True)

    matrix = roster[[month_col(m) for m in months_available]].to_numpy()
    roster["median_views"] = np.median(matrix, axis=1)
    roster["total_views"] = matrix.sum(axis=1)
    roster["max_views"] = matrix.max(axis=1)
    roster["n_months"] = (matrix > 0).sum(axis=1)
    masked = np.ma.masked_equal(matrix, 0)
    roster["median_present"] = np.ma.median(masked, axis=1).filled(0)

    roster = roster.sort_values(["median_views", "total_views"], ascending=False, kind="mergesort").reset_index(
        drop=True
    )
    roster["rank"] = np.arange(1, len(roster) + 1)

    print("\nTop 20 by median monthly views:")
    print(roster[["rank", "title", "median_views", "total_views", "n_months"]].head(20).to_string(index=False))
    print(f"\nPeople with zero views in every month: {(roster['total_views'] == 0).sum():,}")

    write_parquet_atomic(roster, config.RANKED_PARQUET)
    write_parquet_atomic(roster.head(config.TOP_K).reset_index(drop=True), config.TOP_PARQUET)
    print(f"Wrote {config.RANKED_PARQUET} and {config.TOP_PARQUET}")


if __name__ == "__main__":
    main()
