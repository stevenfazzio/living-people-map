"""Shared helpers: HTTP session/retry, atomic parquet writes, title normalization."""

import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests

from config import USER_AGENT


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def get_with_retry(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    max_retries: int = 6,
    timeout: int = 60,
) -> requests.Response:
    """GET with exponential backoff; honors Retry-After on 429."""
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 30))
                print(f"429 rate limited; sleeping {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries - 1:
                raise
            wait = min(2**attempt * 5, 60)
            print(f"Request failed ({exc}); retry {attempt + 1}/{max_retries} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"exhausted retries for {url}")


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp file in the same directory, verify row count, atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False)
        n_verify = len(pd.read_parquet(tmp_path, columns=[df.columns[0]]))
        assert n_verify == len(df), f"row count mismatch writing {path}: {n_verify} vs {len(df)}"
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def save_npz_atomic(path: Path, **arrays) -> None:
    """np.savez_compressed to a temp file in the same directory, verify, atomic rename.

    The temp name MUST end in .npz: np.savez silently appends .npz to any other
    name, writing to a stray file while the rename moves an empty one into place.
    """
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        verify = np.load(tmp, allow_pickle=False)
        assert sorted(verify.files) == sorted(arrays), f"npz key mismatch writing {path}"
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def title_key(title: str) -> str:
    """Lowercased normalization (underscores to spaces, collapse whitespace).

    NO LONGER used for the pageview join: distinct pages can differ only by
    case, so case-folding lets a person absorb an unrelated page's views
    (rapper "TeQuila" absorbed "Tequila" the drink). Stage 02 joins exact
    canonical titles after stage-01b redirect resolution instead. Kept for
    fuzzy/diagnostic use only."""
    return " ".join(title.replace("_", " ").split()).lower()


def parse_dump_title(raw: str) -> str:
    """Undo the CSV-style quoting pageview_complete applies to titles that
    contain double quotes (e.g. "Weird Al" Yankovic)."""
    if len(raw) >= 2 and raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return raw
