import pytest

from cats.orca.legacy_catalog import _parse_ifct_files
from ingest.spectral import _merge


def test_parse_three_fragment_ifct_data_with_major_states(tmp_path):
    (tmp_path / "IFCTdata.txt").write_text(
        "state hole(1) ele(1) hole(2) ele(2) hole(3) ele(3) redis(1) 1->2 1<-2 1->3 1<-3 2->3 2<-3\n"
        "1 0.7 0.8 0.1 0.1 0.2 0.1 0.6594 0.0205 0.0078 0.0878 0.1917 0.0010 0.0060\n",
        encoding="utf-8",
    )
    (tmp_path / "IFCTmajor.txt").write_text(
        "state f nm\n1 0.0048 596.4: Redis(1) 65.9 % 1->3 8.8 % 1<-3 19.2 %\n",
        encoding="utf-8",
    )

    states, terms, _, warnings = _parse_ifct_files("calc", tmp_path, tmp_path, -1)

    assert warnings == []
    assert states == [{"calculation_id": "calc", "source": "IFCTmajor.txt", "state": 1, "f": 0.0048, "nm": 596.4, "CT(e)": None, "CT(%)": None}]
    assert next(row for row in terms if row["term"] == "1->3")["pct"] == pytest.approx(8.78)
    assert len(terms) == 7


def test_empty_followup_does_not_erase_retained_ct_states():
    assert _merge({"ct_states": [{"state": 1}]}, {"ct_states": []})["ct_states"] == [
        {"state": 1}
    ]
