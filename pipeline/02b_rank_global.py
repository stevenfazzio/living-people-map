"""Stage 02b: rank roster people by all-language Wikipedia attention.

Aggregates the stage-01e per-(person, language, month) views into the same
fame metrics as stage 02 -- median of 12 zero-filled monthly totals as THE
ranking metric -- but summed across every Wikipedia language. Also derives
two language facets over the full window:
  top_lang       -- language edition contributing the most views
  english_share  -- en views / all views

Outputs:
  data/people_ranked_global.parquet  -- everyone with >=1 view anywhere
  data/people_top25k_global.parquet  -- head(MAP_N), stage-03-ready columns
"""

import numpy as np
import pandas as pd

import config
from util import write_parquet_atomic


def main() -> None:
    months = [m for m in config.PAGEVIEW_MONTHS if config.global_month_parquet(m).exists()]
    missing = set(config.PAGEVIEW_MONTHS) - set(months)
    if missing:
        print(f"WARNING: ranking on {len(months)}/12 months; missing {sorted(missing)}")

    monthly_totals: dict[str, pd.Series] = {}
    lang_acc: pd.Series | None = None
    for month in months:
        df = pd.read_parquet(config.global_month_parquet(month))
        monthly_totals[month] = df.groupby("qid_int")["views"].sum()
        by_lang = df.groupby(["qid_int", "lang"])["views"].sum()
        lang_acc = by_lang if lang_acc is None else lang_acc.add(by_lang, fill_value=0)
        print(f"{month}: {df['qid_int'].nunique():,} people, {df['views'].sum():,} views", flush=True)

    people = pd.DataFrame({"qid_int": sorted(set().union(*[s.index for s in monthly_totals.values()]))})
    month_cols = []
    for month in months:
        col = f"views_{month.replace('-', '')}"
        month_cols.append(col)
        people[col] = people["qid_int"].map(monthly_totals[month]).fillna(0).astype("int64")

    matrix = people[month_cols].to_numpy()
    people["median_views"] = np.median(matrix, axis=1)
    people["total_views"] = matrix.sum(axis=1)
    people["max_views"] = matrix.max(axis=1)
    people["n_months"] = (matrix > 0).sum(axis=1)

    lang_totals = lang_acc.reset_index().rename(columns={"views": "lang_views"})
    top_lang = lang_totals.loc[lang_totals.groupby("qid_int")["lang_views"].idxmax(), ["qid_int", "lang"]]
    people["top_lang"] = people["qid_int"].map(top_lang.set_index("qid_int")["lang"])
    en_views = lang_totals[lang_totals["lang"] == "en"].set_index("qid_int")["lang_views"]
    people["english_share"] = (people["qid_int"].map(en_views).fillna(0) / people["total_views"].clip(lower=1)).astype(
        "float32"
    )

    people["qid"] = "Q" + people["qid_int"].astype(str)
    enwiki = pd.read_parquet(config.RANKED_PARQUET, columns=["qid", "pageid", "title", "rank", "median_views"]).rename(
        columns={"rank": "enwiki_rank", "median_views": "enwiki_median"}
    )
    people = people.merge(enwiki, on="qid", how="inner")  # drops sitelink targets not in roster (shouldn't exist)

    people = people.sort_values(["median_views", "total_views"], ascending=False, kind="mergesort").reset_index(
        drop=True
    )
    people["rank"] = np.arange(1, len(people) + 1)

    print(f"\nGlobal ranking: {len(people):,} people with views")
    print("\nTop 20 by global median monthly views:")
    print(
        people[["rank", "title", "median_views", "enwiki_rank", "top_lang", "english_share"]]
        .head(20)
        .to_string(index=False)
    )

    write_parquet_atomic(people, config.GLOBAL_RANKED_PARQUET)
    write_parquet_atomic(people.head(config.MAP_N).reset_index(drop=True), config.GLOBAL_TOP_PARQUET)
    print(f"Wrote {config.GLOBAL_RANKED_PARQUET} and {config.GLOBAL_TOP_PARQUET}")


if __name__ == "__main__":
    main()
