"""Compare the enwiki top-25k roster with the global (all-language) top-25k.

Prints: overlap, the biggest newcomers (globally famous, anglosphere-quiet),
the biggest leavers (enwiki-famous, globally quieter), dominant-language
distribution of the global roster, and english_share summary.

Run after stage 02b: uv run python analysis/compare_rosters.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

pd.set_option("display.width", 160)


def main() -> None:
    n = config.MAP_N
    enwiki = pd.read_parquet(config.TOP_PARQUET, columns=["qid", "title", "rank"]).head(n)
    glob = pd.read_parquet(config.GLOBAL_RANKED_PARQUET)
    glob_top = glob.head(n)

    overlap = set(enwiki["qid"]) & set(glob_top["qid"])
    print(f"Top-{n // 1000}k overlap: {len(overlap):,} people ({len(overlap) / n:.1%})")
    print(f"Newcomers in global roster: {n - len(overlap):,}\n")

    newcomers = glob_top[~glob_top["qid"].isin(set(enwiki["qid"]))]
    print("=== Top 25 newcomers (global rank | enwiki rank | dominant lang) ===")
    cols = ["rank", "title", "enwiki_rank", "top_lang", "english_share", "median_views"]
    print(newcomers[cols].head(25).to_string(index=False))

    leavers = glob.merge(enwiki[["qid", "rank"]].rename(columns={"rank": "enwiki_top_rank"}), on="qid")
    leavers = leavers[leavers["rank"] > n].sort_values("enwiki_top_rank")
    print(f"\n=== Top 25 leavers (enwiki-famous, below global top-{n // 1000}k) ===")
    print(leavers[["enwiki_top_rank", "title", "rank", "top_lang", "english_share"]].head(25).to_string(index=False))

    print("\n=== Dominant attention language, global top-25k ===")
    print(glob_top["top_lang"].value_counts().head(20).to_string())

    print("\n=== english_share distribution, global top-25k ===")
    print(glob_top["english_share"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())


if __name__ == "__main__":
    main()
