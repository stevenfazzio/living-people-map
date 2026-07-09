"""Stage 03: fetch page leads, short descriptions, and thumbnails for the map subset.

Two batched passes over the enwiki action API (no per-page REST calls):
  A. prop=extracts (exintro, explaintext)          -- 20 pages/request -> lead
  B. prop=pageimages|description (pithumbsize=320) -- 50 pages/request -> thumb, blurb

The lead is the SEMANTIC text (embedding input); the description ("American
actress") and thumbnail feed the hover card. Both passes checkpoint to JSONL
keyed by pageid, so reruns only fetch what's missing.

Input:  data/people_top100k.parquet, head(MAP_N)
Output: data/people_texts.parquet [pageid, title, qid, lead, description, thumb_url]
"""

import json
import time
from pathlib import Path

import pandas as pd

import config
from util import get_with_retry, make_session, write_parquet_atomic


def load_done(jsonl_path: Path) -> dict[int, dict]:
    done: dict[int, dict] = {}
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                done[rec["pageid"]] = rec
    return done


def fetch_batched(people: pd.DataFrame, jsonl_path: Path, batch_size: int, extra_params: dict, label: str) -> dict:
    """Generic batched title fetch with JSONL checkpointing, keyed by pageid."""
    done = load_done(jsonl_path)
    todo = people[~people["pageid"].isin(done)]
    print(f"{label}: {len(done):,} cached, {len(todo):,} to fetch")
    session = make_session()
    with open(jsonl_path, "a") as out:
        for start in range(0, len(todo), batch_size):
            batch = todo.iloc[start : start + batch_size]
            params = {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "titles": "|".join(batch["title"]),
                "maxlag": "5",
                **extra_params,
            }
            resp = get_with_retry(session, config.WIKI_API_URL, params=params)
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"{label}: API error {data['error']}")
            pages = {p["pageid"]: p for p in data.get("query", {}).get("pages", []) if "pageid" in p}
            for pageid in batch["pageid"]:
                page = pages.get(pageid, {})
                rec = {
                    "pageid": int(pageid),
                    "extract": page.get("extract"),
                    "description": page.get("description"),
                    "thumb_url": page.get("thumbnail", {}).get("source"),
                }
                out.write(json.dumps(rec) + "\n")
                done[rec["pageid"]] = rec
            out.flush()
            if (start // batch_size) % 50 == 0:
                print(f"{label}: {len(done):,}/{len(people):,}", flush=True)
            time.sleep(config.TEXT_SLEEP_S)
    return done


def main() -> None:
    people = pd.read_parquet(config.TOP_PARQUET).head(config.MAP_N)
    people = people[["pageid", "title", "qid", "rank", "median_views", "total_views", "max_views"]]
    print(f"Fetching texts for top {len(people):,} people")

    extracts = fetch_batched(
        people,
        config.EXTRACTS_JSONL,
        config.EXTRACTS_BATCH,
        {"prop": "extracts", "exintro": "1", "explaintext": "1", "exlimit": "max"},
        "extracts",
    )
    props = fetch_batched(
        people,
        config.PROPS_JSONL,
        config.PROPS_BATCH,
        {"prop": "pageimages|description", "piprop": "thumbnail", "pithumbsize": str(config.THUMB_SIZE)},
        "props",
    )

    people["lead"] = people["pageid"].map(lambda p: (extracts.get(p) or {}).get("extract"))
    people["description"] = people["pageid"].map(lambda p: (props.get(p) or {}).get("description"))
    people["thumb_url"] = people["pageid"].map(lambda p: (props.get(p) or {}).get("thumb_url"))

    n_no_lead = int(people["lead"].isna().sum() + (people["lead"].fillna("").str.len() < 40).sum())
    print(f"Missing/short leads: {n_no_lead:,}")
    print(f"Missing descriptions: {int(people['description'].isna().sum()):,}")
    print(f"Missing thumbnails: {int(people['thumb_url'].isna().sum()):,}")

    write_parquet_atomic(people, config.TEXTS_PARQUET)
    print(f"Wrote {config.TEXTS_PARQUET} ({len(people):,} rows)")


if __name__ == "__main__":
    main()
