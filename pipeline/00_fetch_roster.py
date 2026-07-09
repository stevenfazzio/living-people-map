"""Stage 00: enumerate Category:Living people with Wikidata QIDs.

Crawls the enwiki action API with generator=categorymembers + prop=pageprops,
so one pass yields title, pageid, AND the wikibase_item QID per page. Raw
batches are appended to a JSONL checkpoint (data/.roster_pages.jsonl) with
their continuation token, so an interrupted crawl resumes for free. The final
parquet is assembled from the JSONL only once the crawl reports completion.

Output: data/roster.parquet [title, pageid, qid]
~1.15M pages / 500 per request = ~2,300 requests = ~15-20 min.
"""

import json
import time

import pandas as pd

import config
from util import get_with_retry, make_session, write_parquet_atomic


def api_params(cont: dict | None) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "categorymembers",
        "gcmtitle": config.ROSTER_CATEGORY,
        "gcmtype": "page",
        "gcmnamespace": "0",
        "gcmlimit": str(config.ROSTER_BATCH_SIZE),
        "prop": "pageprops",
        "ppprop": "wikibase_item",
        "maxlag": "5",
    }
    if cont:
        params.update(cont)
    return params


def load_checkpoint() -> tuple[int, dict | None, bool]:
    """Scan the JSONL checkpoint; return (n_batches, continue_params, finished)."""
    if not config.ROSTER_PAGES_JSONL.exists():
        return 0, None, False
    n_batches, cont, finished = 0, None, False
    with open(config.ROSTER_PAGES_JSONL) as f:
        for line in f:
            record = json.loads(line)
            n_batches += 1
            cont = record["continue"]
            finished = cont is None
    return n_batches, cont, finished


def crawl() -> bool:
    """Fetch category member batches until exhausted. Returns True if complete."""
    n_batches, cont, finished = load_checkpoint()
    if finished:
        print(f"Crawl already complete ({n_batches} batches).")
        return True
    if n_batches:
        print(f"Resuming after batch {n_batches}.")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    with open(config.ROSTER_PAGES_JSONL, "a") as out:
        while True:
            resp = get_with_retry(session, config.WIKI_API_URL, params=api_params(cont))
            data = resp.json()
            if "error" in data:
                if data["error"]["code"] == "maxlag":
                    print("Server lagged (maxlag); sleeping 5s")
                    time.sleep(5)
                    continue
                raise RuntimeError(f"API error: {data['error']}")
            pages = data.get("query", {}).get("pages", [])
            cont = data.get("continue")
            batch = [
                {
                    "title": p["title"],
                    "pageid": p["pageid"],
                    "qid": p.get("pageprops", {}).get("wikibase_item"),
                }
                for p in pages
            ]
            out.write(json.dumps({"pages": batch, "continue": cont}) + "\n")
            out.flush()
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"Fetched {n_batches} batches (~{n_batches * config.ROSTER_BATCH_SIZE:,} pages)")
            if cont is None:
                print(f"Crawl complete: {n_batches} batches.")
                return True
            if config.MAX_ROSTER_REQUESTS and n_batches >= config.MAX_ROSTER_REQUESTS:
                print(f"Stopping at MAX_ROSTER_REQUESTS={config.MAX_ROSTER_REQUESTS} (smoke test); crawl NOT complete.")
                return False
            time.sleep(config.ROSTER_SLEEP_S)


def assemble() -> None:
    rows = []
    with open(config.ROSTER_PAGES_JSONL) as f:
        for line in f:
            rows.extend(json.loads(line)["pages"])
    df = pd.DataFrame(rows)
    n_raw = len(df)
    df = df.drop_duplicates(subset="pageid").reset_index(drop=True)
    if n_raw != len(df):
        print(f"Dropped {n_raw - len(df)} duplicate pageids from continuation overlap.")
    df["pageid"] = df["pageid"].astype("int64")
    n_missing_qid = int(df["qid"].isna().sum())
    print(f"Roster: {len(df):,} people, {n_missing_qid:,} without a QID ({n_missing_qid / len(df):.2%})")
    write_parquet_atomic(df, config.ROSTER_PARQUET)
    print(f"Wrote {config.ROSTER_PARQUET}")


def main() -> None:
    if crawl():
        assemble()
    else:
        print("Skipping parquet assembly (crawl incomplete). Inspect the JSONL checkpoint instead.")


if __name__ == "__main__":
    main()
