import pytest
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from gui_tools.screener_core import (
    DEFAULT_SIMILARITY_THRESHOLD,
    HAS_REMATCH,
    partition_trajectories,
    screen_trajectory,
)


def test_has_rematch_available():
    assert HAS_REMATCH is True


def test_screen_trajectory_rematch_deduplication():
    # Construct trajectory where frame 1 is nearly identical to frame 0,
    # and frame 2 has a significant structural displacement
    m0 = molecule("H2O")
    m1 = m0.copy()
    m1.positions[0] += 0.005  # high similarity duplicate (>0.999)
    m2 = m0.copy()
    m2.positions[0] += 0.80  # distinct geometry (~0.87 similarity)
    m3 = m0.copy()
    m3.positions[1] += 0.01

    traj = [m0, m1, m2, m3]
    indices, rejected, stats = screen_trajectory(traj, similarity_threshold=0.98)

    assert 0 in indices  # t0
    assert 3 in indices  # t_final
    assert 1 not in indices  # dropped as duplicate by SOAP REMatch
    assert 2 in indices  # kept as distinct intermediate
    assert stats["endpoints"] == 2
    assert stats["intermediate"] == 1


def test_screen_trajectory_preserves_high_force_and_high_energy():
    m0 = molecule("H2O")
    m0.calc = SinglePointCalculator(m0, energy=-14.0, forces=[[0.0, 0.0, 0.0]] * len(m0))

    # Barrier structure with high force
    m1 = m0.copy()
    m1.calc = SinglePointCalculator(m1, energy=-13.0, forces=[[6.0, 0.0, 0.0]] * len(m1))

    # High energy structure
    m2 = m0.copy()
    m2.calc = SinglePointCalculator(m2, energy=-11.0, forces=[[0.1, 0.0, 0.0]] * len(m2))

    # Relaxed end state
    m3 = m0.copy()
    m3.calc = SinglePointCalculator(m3, energy=-14.2, forces=[[0.01, 0.0, 0.0]] * len(m3))

    traj = [m0, m1, m2, m3]
    indices, rejected, stats = screen_trajectory(traj, similarity_threshold=0.98)

    assert 0 in indices
    assert 1 in indices  # high force >= 5.0
    assert 2 in indices  # high energy >= 0.5 eV/atom
    assert 3 in indices
    assert stats["high_force"] == 2


def test_screen_trajectory_dynamic_threshold_subtle_relaxation():
    # Subtle relaxation where all similarities are > 0.999
    # Fixed 0.98 threshold would drop everything, but dynamic threshold adapts and retains frames
    m0 = molecule("H2O")
    traj = [m0.copy() for _ in range(6)]
    for i, m in enumerate(traj):
        m.positions[0] += [i * 0.05, 0, 0]

    indices, rejected, stats = screen_trajectory(traj, similarity_threshold="dynamic")

    assert 0 in indices
    assert 5 in indices
    # Dynamic threshold should be > 0.999 and retain at least 1 intermediate frame
    assert stats["threshold_used"] > 0.999
    assert stats["intermediate"] >= 1


def test_run_screening_workflow_directory_with_mixed_formats(tmp_path):
    import ase.db
    from ase.io import write
    from ase.io.trajectory import Trajectory
    from gui_tools.screener_core import discover_trajectory_files, inspect_dataset, run_screening_workflow

    folder = tmp_path / "mixed_trajectories"
    folder.mkdir()

    # 1. extxyz with 2 trajectories partitioned by source_frame
    m0 = molecule("H2O")
    t1_f0 = m0.copy()
    t1_f0.info["source_frame"] = 0
    t1_f1 = m0.copy()
    t1_f1.positions[0] += 0.8
    t1_f1.info["source_frame"] = 1

    t2_f0 = m0.copy()
    t2_f0.info["source_frame"] = 0
    t2_f1 = m0.copy()
    t2_f1.positions[0] += 0.9
    t2_f1.info["source_frame"] = 1

    extxyz_file = folder / "runs.extxyz"
    write(str(extxyz_file), [t1_f0, t1_f1, t2_f0, t2_f1], format="extxyz")

    # 2. .traj file with 1 trajectory
    traj_file = folder / "run_opt.traj"
    with Trajectory(str(traj_file), mode="w") as t:
        t.write(m0.copy())
        m_step = m0.copy()
        m_step.positions[1] += 0.5
        t.write(m_step)

    # 3. .db file with 1 trajectory
    db_file = folder / "calc.db"
    with ase.db.connect(str(db_file)) as db:
        db.write(m0.copy())
        m_step2 = m0.copy()
        m_step2.positions[2] += 0.4
        db.write(m_step2)

    # Test discovery
    discovered = discover_trajectory_files(folder)
    assert len(discovered) == 3

    # Test inspection
    info = inspect_dataset(folder)
    assert info["is_directory"] is True
    assert info["file_count"] == 3
    assert info["total_frames"] == 8

    # Test screening workflow on entire directory
    out_file = tmp_path / "screened_all.xyz"
    audit_file = tmp_path / "audit.csv"

    res = run_screening_workflow(
        folder,
        out_file,
        audit_csv_path=audit_file,
        similarity_threshold="dynamic",
    )

    assert res["status"] == "success"
    assert res["files_count"] == 3
    # 2 trajectories from extxyz + 1 from traj + 1 from db = 4 trajectory bins
    assert res["trajectories_count"] == 4
    assert res["final_count"] > 0
    assert out_file.is_file()
    assert audit_file.is_file()

    # Check audit rows
    import csv
    with open(audit_file, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 4
        source_files = {r["source_file"] for r in reader}
        assert "runs.extxyz" in source_files
        assert "run_opt.traj" in source_files
        assert "calc.db" in source_files
        for row in reader:
            assert "rejected_similarity" in row
            assert "rejected_cap" in row
            assert "rejected_energy" in row
            assert "rejected_force" in row


def test_force_and_energy_retention_in_extxyz_and_traj(tmp_path):
    import numpy as np
    from ase.io import read, write
    from gui_tools.screener_core import run_screening_workflow

    m0 = molecule("H2O")
    forces0 = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3], [0.0, 0.0, 0.0]])
    m0.calc = SinglePointCalculator(m0, energy=-14.25, forces=forces0)

    m1 = m0.copy()
    m1.positions[0] += 0.5
    forces1 = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    m1.calc = SinglePointCalculator(m1, energy=-12.50, forces=forces1)

    in_xyz = tmp_path / "input_traj.xyz"
    write(str(in_xyz), [m0, m1], format="extxyz")

    out_xyz = tmp_path / "screened_retained.xyz"
    res = run_screening_workflow(in_xyz, out_xyz, tag_ref_properties=True)
    assert res["status"] == "success"
    assert res["final_count"] == 2
    assert "rejected_similarity" in res
    assert "rejected_cap" in res
    assert "rejected_energy" in res
    assert "rejected_force" in res

    # Read back screened structures and verify calculators & forces
    read_images = list(read(str(out_xyz), index=":"))
    assert len(read_images) == 2

    for img in read_images:
        assert img.calc is not None
        read_forces = img.get_forces()
        assert read_forces is not None
        assert read_forces.shape == (len(img), 3)
        assert np.all(np.isfinite(read_forces))
        # Verify REF tags match standard calculator properties
        assert "REF_forces" in img.arrays
        np.testing.assert_allclose(read_forces, img.arrays["REF_forces"])
        assert "REF_energy" in img.info
        assert abs(img.get_potential_energy() - img.info["REF_energy"]) < 1e-6

    # Verify first frame exact numbers
    np.testing.assert_allclose(read_images[0].get_forces(), forces0, atol=1e-5)
    assert abs(read_images[0].get_potential_energy() - (-14.25)) < 1e-5


def test_screen_trajectory_rejection_breakdown():
    m0 = molecule("H2O")
    m0.calc = SinglePointCalculator(m0, energy=-14.0, forces=[[0.0, 0.0, 0.0]] * len(m0))

    # 1. Diverged force frame -> rejected_force
    m_force_div = m0.copy()
    m_force_div.calc = SinglePointCalculator(m_force_div, energy=-13.0, forces=[[150.0, 0.0, 0.0]] * len(m0))

    # 2. Extreme energy frame -> rejected_energy
    m_energy_bad = m0.copy()
    m_energy_bad.calc = SinglePointCalculator(m_energy_bad, energy=2000.0, forces=[[0.1, 0.0, 0.0]] * len(m0))

    # 3. Intermediate duplicate -> rejected_similarity
    m_dup = m0.copy()
    m_dup.positions[0] += 0.002
    m_dup.calc = SinglePointCalculator(m_dup, energy=-14.0, forces=[[0.1, 0.0, 0.0]] * len(m0))

    # 4. Distinct intermediate 1 -> kept
    m_distinct1 = m0.copy()
    m_distinct1.positions[0] += 0.6
    m_distinct1.calc = SinglePointCalculator(m_distinct1, energy=-13.8, forces=[[0.2, 0.0, 0.0]] * len(m0))

    # 5. Distinct intermediate 2 -> rejected by max_intermediate_frames cap if cap=1
    m_distinct2 = m0.copy()
    m_distinct2.positions[1] += 0.7
    m_distinct2.calc = SinglePointCalculator(m_distinct2, energy=-13.7, forces=[[0.2, 0.0, 0.0]] * len(m0))

    # 6. Final relaxed state -> endpoint
    m_end = m0.copy()
    m_end.calc = SinglePointCalculator(m_end, energy=-14.1, forces=[[0.01, 0.0, 0.0]] * len(m0))

    traj = [m0, m_force_div, m_energy_bad, m_dup, m_distinct1, m_distinct2, m_end]
    indices, rejected, stats = screen_trajectory(
        traj,
        max_intermediate_frames=1,
        similarity_threshold=0.98,
        max_frame_fmax_ev_a=100.0,
        max_abs_energy_per_atom_ev=500.0,
    )

    assert stats["rejected_force"] == 1
    assert stats["rejected_energy"] == 1
    assert stats["rejected_similarity"] >= 1
    assert stats["rejected_cap"] == 1
    assert stats["total_rejected"] == (
        stats["rejected_force"]
        + stats["rejected_energy"]
        + stats["rejected_similarity"]
        + stats["rejected_cap"]
    )
    assert len(indices) + stats["total_rejected"] == len(traj)


