from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cats.orca.tddft.ct import extract_state_transition_vectors
from cats.multiwfn.iroot_scoring import _normalise_transition_vector, _cosine_overlap


SAMPLE_ORCA_OUTPUT = """
                     *****************************
                     * TD-DFT/TDA EXCITED STATES *
                     *****************************

STATE  1:  E=   0.021918 au      0.596 eV     4810.5 cm**-1 <S**2> =   2.657189 Mult 4
   254a -> 1337a  :     0.000000 (c= -0.00021683)
   255a -> 1345a  :     0.000001 (c=  0.00086410)
   597b -> 617b   :     0.371400 (c= -0.60942616)

STATE  2:  E=   0.023558 au      0.641 eV     5170.4 cm**-1 <S**2> =   0.838562 Mult 2
   597b -> 617b   :     0.350000 (c= -0.59160797)
   592b -> 611b   :     0.081300 (c=  0.28514464)

STATE  3:  E=   0.025398 au      0.691 eV     5574.2 cm**-1 <S**2> =   0.871597 Mult 2
   10 -> 20       :     0.250000

-----------------------------
TD-DFT/TDA-EXCITATION SPECTRA
-----------------------------
"""


def test_extract_state_transition_vectors_parses_c_coeff(tmp_path: Path):
    out_file = tmp_path / "orca.out"
    out_file.write_text(SAMPLE_ORCA_OUTPUT, encoding="utf-8")

    vectors = extract_state_transition_vectors(out_file)
    assert 1 in vectors
    assert 2 in vectors
    assert 3 in vectors

    # Verify State 1 captures c= values instead of 0.000000
    assert abs(vectors[1]["254a->1337a"] - (-0.00021683)) < 1e-8
    assert abs(vectors[1]["255a->1345a"] - 0.00086410) < 1e-8
    assert abs(vectors[1]["597b->617b"] - (-0.60942616)) < 1e-8

    # Verify State 3 falls back to weight when c= is absent
    assert abs(vectors[3]["10->20"] - 0.250000) < 1e-6


def test_extract_state_transition_vectors_with_filter(tmp_path: Path):
    out_file = tmp_path / "orca.out"
    out_file.write_text(SAMPLE_ORCA_OUTPUT, encoding="utf-8")

    vectors = extract_state_transition_vectors(out_file, state_filter={1, 3})
    assert 1 in vectors
    assert 3 in vectors
    assert 2 not in vectors


def test_transition_overlap_end_to_end(tmp_path: Path):
    out_file = tmp_path / "orca.out"
    out_file.write_text(SAMPLE_ORCA_OUTPUT, encoding="utf-8")

    vectors = extract_state_transition_vectors(out_file)
    nv1 = _normalise_transition_vector(vectors[1])
    nv2 = _normalise_transition_vector(vectors[2])

    assert nv1 is not None
    assert nv2 is not None
    overlap = _cosine_overlap(nv1, nv2)
    assert overlap is not None
    # Both states have 597b->617b as dominant transition
    assert overlap > 0.8
