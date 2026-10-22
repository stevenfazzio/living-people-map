import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "fetch_wikidata", Path(__file__).resolve().parent.parent / "pipeline" / "04_fetch_wikidata.py"
)
fetch_wikidata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_wikidata)


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


def test_wikidata_reads_send_no_maxlag(monkeypatch):
    sent = []

    def fake_get(session, url, params):
        sent.append(params)
        return FakeResponse({"entities": {"Q42": {"id": "Q42"}}})

    monkeypatch.setattr(fetch_wikidata, "get_with_retry", fake_get)
    assert fetch_wikidata.fetch_entities(None, ["Q42"], "claims") == {"Q42": {"id": "Q42"}}
    assert "maxlag" not in sent[0]


def test_api_errors_raise_instead_of_retrying_forever(monkeypatch):
    calls = []

    def fake_get(session, url, params):
        calls.append(params)
        return FakeResponse({"error": {"code": "maxlag", "info": "Waiting for wdqs1012: 15.5 seconds lagged."}})

    monkeypatch.setattr(fetch_wikidata, "get_with_retry", fake_get)
    with pytest.raises(RuntimeError, match="maxlag"):
        fetch_wikidata.fetch_entities(None, ["Q42"], "claims")
    assert len(calls) == 1
