"""Stage 04: fetch Wikidata facets for the map subset.

wbgetentities in batches of 50 QIDs (props=claims), then a second pass to
resolve value QIDs (occupations, citizenships, genders) to English labels.
Facets: P106 occupation, P27 citizenship, P569 birth year, P21 gender.
Deprecated-rank claims are skipped; preferred-rank claims win over normal.

Input:  data/people_top100k.parquet head(MAP_N)  (qid column from stage 00)
Output: data/people_wikidata.parquet
        [qid, pageid, birth_year, gender, citizenships, occupations]
"""

import json
import time

import pandas as pd

import config
from util import get_with_retry, make_session, write_parquet_atomic

CLAIM_PROPS = {"P106": "occupation_qids", "P27": "citizenship_qids", "P21": "gender_qids"}


def best_claims(claims: list[dict]) -> list[dict]:
    kept = [c for c in claims if c.get("rank") != "deprecated"]
    preferred = [c for c in kept if c.get("rank") == "preferred"]
    return preferred or kept


def item_ids(claims: list[dict]) -> list[str]:
    out = []
    for c in best_claims(claims):
        value = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and "id" in value:
            out.append(value["id"])
    return out


def birth_year(claims: list[dict]) -> int | None:
    for c in best_claims(claims):
        value = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and value.get("time", "").startswith("+") and value.get("precision", 0) >= 9:
            return int(value["time"][1:5])
    return None


def fetch_entities(session, qids: list[str], props: str) -> dict[str, dict]:
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": "|".join(qids),
        "props": props,
        "maxlag": "5",
    }
    if props == "labels":
        params["languages"] = "en"
    resp = get_with_retry(session, config.WIKIDATA_API_URL, params=params)
    data = resp.json()
    if "error" in data:
        if data["error"].get("code") == "maxlag":
            time.sleep(5)
            return fetch_entities(session, qids, props)
        raise RuntimeError(f"wbgetentities error: {data['error']}")
    return data.get("entities", {})


def fetch_claims(people: pd.DataFrame) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if config.WIKIDATA_JSONL.exists():
        with open(config.WIKIDATA_JSONL) as f:
            for line in f:
                rec = json.loads(line)
                done[rec["qid"]] = rec
    todo = [q for q in people["qid"] if q and q not in done]
    print(f"claims: {len(done):,} cached, {len(todo):,} to fetch")
    session = make_session()
    with open(config.WIKIDATA_JSONL, "a") as out:
        for start in range(0, len(todo), config.WB_BATCH):
            batch = todo[start : start + config.WB_BATCH]
            entities = fetch_entities(session, batch, "claims")
            for qid in batch:
                claims = entities.get(qid, {}).get("claims", {})
                rec = {"qid": qid, "birth_year": birth_year(claims.get("P569", []))}
                for prop, col in CLAIM_PROPS.items():
                    rec[col] = item_ids(claims.get(prop, []))
                out.write(json.dumps(rec) + "\n")
                done[qid] = rec
            out.flush()
            if (start // config.WB_BATCH) % 50 == 0:
                print(f"claims: {len(done):,}/{len(people):,}", flush=True)
            time.sleep(config.TEXT_SLEEP_S)
    return done


def fetch_labels(value_qids: list[str]) -> dict[str, str]:
    session = make_session()
    labels: dict[str, str] = {}
    print(f"labels: {len(value_qids):,} distinct value QIDs")
    for start in range(0, len(value_qids), config.WB_BATCH):
        batch = value_qids[start : start + config.WB_BATCH]
        entities = fetch_entities(session, batch, "labels")
        for qid in batch:
            labels[qid] = entities.get(qid, {}).get("labels", {}).get("en", {}).get("value", qid)
        time.sleep(config.TEXT_SLEEP_S)
    return labels


def main() -> None:
    people = pd.read_parquet(config.map_roster_parquet(), columns=["pageid", "qid"]).head(config.MAP_N)
    n_missing_qid = int(people["qid"].isna().sum())
    if n_missing_qid:
        print(f"{n_missing_qid} people have no QID; facets will be null for them")

    claims = fetch_claims(people.dropna(subset=["qid"]))

    distinct = sorted({q for rec in claims.values() for col in CLAIM_PROPS.values() for q in rec[col]})
    labels = fetch_labels(distinct)

    def build_row(qid):
        rec = claims.get(qid)
        if not rec:
            return pd.Series({"birth_year": None, "gender": None, "citizenships": [], "occupations": []})
        return pd.Series(
            {
                "birth_year": rec["birth_year"],
                "gender": labels.get(rec["gender_qids"][0]) if rec["gender_qids"] else None,
                "citizenships": [labels.get(q, q) for q in rec["citizenship_qids"]],
                "occupations": [labels.get(q, q) for q in rec["occupation_qids"]],
            }
        )

    people = pd.concat([people, people["qid"].apply(build_row)], axis=1)
    people["birth_year"] = people["birth_year"].astype("Int64")
    print(f"birth_year coverage: {people['birth_year'].notna().mean():.1%}")
    print(f"gender coverage: {people['gender'].notna().mean():.1%}")
    print(f"occupation coverage: {(people['occupations'].str.len() > 0).mean():.1%}")

    write_parquet_atomic(people, config.WIKIDATA_PARQUET)
    print(f"Wrote {config.WIKIDATA_PARQUET} ({len(people):,} rows)")


if __name__ == "__main__":
    main()
