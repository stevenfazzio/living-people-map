import numpy as np

from util import parse_dump_title, save_npz_atomic, title_key


def test_save_npz_atomic_writes_real_data(tmp_path):
    # regression: np.savez appends .npz to non-.npz temp names, leaving the
    # real data in a stray file and renaming an empty one into place
    out = tmp_path / "arrays.npz"
    save_npz_atomic(out, a=np.arange(5), b=np.ones((2, 2)))
    loaded = np.load(out)
    assert sorted(loaded.files) == ["a", "b"]
    assert loaded["a"].sum() == 10
    assert not list(tmp_path.glob("tmp*"))  # no strays left behind


def test_title_key_underscores_and_case():
    assert title_key("LeBron_James") == "lebron james"
    assert title_key("LeBron James") == "lebron james"
    assert title_key("Lebron  James ") == "lebron james"


def test_title_key_matches_api_and_dump_forms():
    # API form (spaces) and dump form (underscores) must agree
    assert title_key('"Weird Al" Yankovic') == title_key('"Weird_Al"_Yankovic')


def test_parse_dump_title_plain():
    assert parse_dump_title("LeBron_James") == "LeBron_James"


def test_parse_dump_title_quoted():
    # pageview_complete CSV-quotes titles containing double quotes
    assert parse_dump_title('"\\"Weird_Al\\"_Yankovic"') == '"Weird_Al"_Yankovic'


def test_parse_dump_title_quoted_backslash():
    assert parse_dump_title('"a\\\\b"') == "a\\b"


def test_parse_dump_title_lone_quote_char():
    # a bare quote is not a quoted field
    assert parse_dump_title('"') == '"'
