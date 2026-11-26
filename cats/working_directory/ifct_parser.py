from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List

import pandas as pd


_ROW_RE = re.compile(
    r"^\s*(?P<state>\d+)\s+"
    r"(?P<f>[+-]?\d+(?:\.\d+)?)\s+"
    r"(?P<nm>[+-]?\d+(?:\.\d+)?):\s+"
    r"(?P<rest>.+?)\s*$"
)

_TERM_RE = re.compile(
    r"(?P<label>Redis\(\d+\)|\d+->\d+|\d+<-\d+)\s+"
    r"(?P<pct>[+-]?\d+(?:\.\d+)?)\s*%"
)


def parse_ifct(
    file: Union[str, Path],
    *,
    wide: bool = False,
    top_n: Optional[int] = None,
    min_pct: Optional[float] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Parse an IFCT contribution listing into pandas DataFrames.

    Parameters
    ----------
    file
        Path to the text file.
    wide
        If False (default), return (states_df, terms_df) where terms_df is long-form:
            terms_df columns = [state, term, pct]
        If True, return a single merged DataFrame:
            states_df merged with wide-form term columns (one column per term).
    top_n
        When wide=True (or when building wide terms), keep only top N terms per state (by pct).
    min_pct
        Filter terms to pct >= min_pct before wide pivot (and also applies to returned terms_df).

    Returns
    -------
    If wide=False:
        (states_df, terms_df)
    If wide=True:
        merged_df
    """
    file = Path(file)
    lines = file.read_text(encoding="utf-8", errors="replace").splitlines()

    state_rows: List[Dict[str, Any]] = []
    term_rows: List[Dict[str, Any]] = []

    for line in lines:
        m = _ROW_RE.match(line)
        if not m:
            continue

        state = int(m.group("state"))
        f = float(m.group("f"))
        nm = float(m.group("nm"))
        rest = m.group("rest")

        state_rows.append({"state": state, "f": f, "nm": nm})

        for tm in _TERM_RE.finditer(rest):
            term_rows.append(
                {"state": state, "term": tm.group("label"), "pct": float(tm.group("pct"))}
            )

    states_df = (
        pd.DataFrame(state_rows)
        .sort_values("state")
        .reset_index(drop=True)
    )
    terms_df = (
        pd.DataFrame(term_rows)
        .sort_values(["state", "pct"], ascending=[True, False])
        .reset_index(drop=True)
    )

    if min_pct is not None and not terms_df.empty:
        terms_df = terms_df[terms_df["pct"] >= float(min_pct)].reset_index(drop=True)

    if not wide:
        return states_df, terms_df

    # build wide-form terms
    df = terms_df.copy()
    if top_n is not None and not df.empty:
        df = (
            df.sort_values(["state", "pct"], ascending=[True, False])
              .groupby("state", as_index=False)
              .head(int(top_n))
        )

    wide_terms = (
        df.pivot_table(index="state", columns="term", values="pct", aggfunc="max")
          .reset_index()
    )
    wide_terms.columns.name = None

    merged = states_df.merge(wide_terms, on="state", how="left")
    return merged

