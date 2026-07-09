import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "fetch_redirects", Path(__file__).resolve().parent.parent / "pipeline" / "01b_fetch_redirects.py"
)
fetch_redirects = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_redirects)

PAGE_LINE = (
    "INSERT INTO `page` VALUES "
    "(10,0,'AccessibleComputing',1,0,0.33,'20260101000000','20260101000000',123,111,'wikitext',NULL),"
    "(12,0,'Anarchism',0,0,0.78,'20260101000000','20260101000000',456,222,'wikitext',NULL),"
    "(13,1,'Talk_page',0,0,0.12,'20260101000000','20260101000000',789,333,'wikitext',NULL),"
    "(14,0,'O\\'Brien_(surname)',0,0,0.5,'20260101000000','20260101000000',321,444,'wikitext',NULL);"
)

REDIRECT_LINE = (
    "INSERT INTO `redirect` VALUES "
    "(10,0,'Computer_accessibility','',''),"
    "(99,4,'Project_page','',''),"
    "(15,0,'O\\'Brien','','');"
)


def test_page_row_regex():
    rows = fetch_redirects.PAGE_ROW_RE.findall(PAGE_LINE)
    assert ("10", "0", "AccessibleComputing", "1") in rows
    assert ("12", "0", "Anarchism", "0") in rows
    assert ("13", "1", "Talk_page", "0") in rows  # ns filtering happens later
    titles = [fetch_redirects.unescape_sql(r[2]) for r in rows]
    assert "O'Brien_(surname)" in titles


def test_redirect_row_regex():
    rows = fetch_redirects.REDIRECT_ROW_RE.findall(REDIRECT_LINE)
    assert ("10", "0", "Computer_accessibility") in rows
    assert ("99", "4", "Project_page") in rows
    assert fetch_redirects.unescape_sql(rows[2][2]) == "O'Brien"


def test_unescape_sql():
    assert fetch_redirects.unescape_sql("O\\'Brien") == "O'Brien"
    assert fetch_redirects.unescape_sql('a\\"b\\\\c') == 'a"b\\c'
