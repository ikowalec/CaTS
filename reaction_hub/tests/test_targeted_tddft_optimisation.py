from types import SimpleNamespace

import pytest

from jobs import tasks
from jobs.models import HPCSystem
from jobs.submission import _build_wrapper_script
from references.models import CalculationSettingsProfile
from results.models import ComputedProperty
from workflows.models import WorkflowDefinition, WorkflowRun


def test_ts_linked_path_payload_and_wrapper():
    payload = tasks._validate_analysis_request("ts_vibrations_gradient", {
        "source_run_id": 9683, "structure_artifact_id": 169875,
        "ground_state_only": True, "relax_endpoints": True,
        "reaction_bond": [20, 22], "hpc_name": "FALCON",
    })
    assert payload["displacement_A"] == .1
    with pytest.raises(ValueError, match="ground_state_only"):
        tasks._validate_analysis_request("ts_vibrations_gradient", {**payload, "ground_state_only": False})
    with pytest.raises(ValueError, match="displacement_A"):
        tasks._validate_analysis_request("ts_vibrations_gradient", {**payload, "displacement_A": float("nan")})
    script = _build_wrapper_script(
        workflow_module="workflow", run_mode="charge_transfer", hpc_name="FALCON",
        qm_backend="orca", qm_backend_settings={},
        n_iter=1, use_tddft_optimisation=False, fmax=.02, active_learning_optimizer="bfgs",
        target_wavelength_nm=500., basis_keyword="6-31g(d,p)", xc_functional="CAM-B3LYP",
        solvent=None, spin_multiplicity=1, nodes=1, tasks_per_node=48, cpus_per_task=1,
        al_steps_per_iter=200, max_al_steps_per_iter=None,
        charge_transfer_config={"density_analysis_only": True, "density_analysis": {"ts_vibrations_gradient": payload}},
    )
    compile(script, "wrapper", "exec")
    assert 'relax_endpoints=bool(ts_cfg.get("relax_endpoints", False))' in script
    assert 'reaction_bond=ts_cfg.get("reaction_bond")' in script


@pytest.mark.django_db
def test_ground_ts_path_supplies_valid_fragments(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="ts-path")
    workflow = WorkflowDefinition.objects.create(name="ts-path", code_path="x.py")
    source = WorkflowRun.objects.create(workflow=workflow, settings_profile=profile, run_name="CuCO")
    ComputedProperty.objects.create(workflow_run=source, name="final_structure_snapshot",
        metadata={"symbols": ["Cu", "C", "O"], "positions_A": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]})
    HPCSystem.objects.create(name="FALCON", host="example.org", username="user")
    captured = []
    monkeypatch.setattr("jobs.submission.submit_slurm_workflow_run",
        lambda **kw: captured.append(kw) or SimpleNamespace(workflow_run_id=1, job_record_id=2, scheduler_job_id="3"))
    tasks._submit_ts_vibrations_gradient_run(source.pk, ground_state_only=True, relax_endpoints=True)
    assert captured[0]["charge_transfer_config"]["fragments_0based"] == {"1": [0], "2": [1, 2]}
    assert captured[0]["slurm_memory"] == "320G"


@pytest.mark.django_db
def test_targeted_tddft_optimisation_uses_source_geometry_and_standard_queue(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(
        name="tda-opt", basis_set="LANL2DZ", xc_functional="CAM-B3LYP"
    )
    workflow = WorkflowDefinition.objects.create(name="tda-opt", code_path="x.py")
    source = WorkflowRun.objects.create(workflow=workflow, settings_profile=profile, run_name="Au20_O2")
    ComputedProperty.objects.create(
        workflow_run=source,
        name="final_structure_snapshot",
        metadata={"symbols": ["Au", "O", "O"], "positions_A": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
    )
    HPCSystem.objects.create(name="FALCON", host="example.org", username="user")
    captured = []
    monkeypatch.setattr(
        "jobs.submission.submit_slurm_workflow_run",
        lambda **kwargs: captured.append(kwargs)
        or SimpleNamespace(workflow_run_id=41, job_record_id=42, scheduler_job_id="43"),
    )

    clean = tasks._validate_analysis_request(
        "targeted_tddft_optimisation",
        {"source_run_id": source.pk, "target_wavelength_nm": 611.4, "nroots": 22, "iroot": 22, "qos": "long"},
    )
    result = tasks._submit_targeted_tddft_optimisation(**clean)

    assert result["scheduler_job_id"] == "43"
    job = captured[0]
    assert job["account"] == "FALCON_SLURM_ACCOUNT"
    assert (job["partition"], job["qos"], job["walltime"]) == ("htc_genoa", "", "24:00:00")
    assert job["structure_xyz"].splitlines()[0] == "3"
    assert job["use_tddft_optimisation"] is True
    assert job["target_wavelength_nm"] == 611.4
    assert job["single_constraints_config"] == {"tddft_scan_nroots": 22, "initial_iroot": 22}
    assert job["run_name"].startswith("Au20_O2_TDA_opt_S22_from_")


@pytest.mark.django_db
def test_targeted_tddft_optimisation_with_reservation(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(
        name="tda-opt-res", basis_set="LANL2DZ", xc_functional="CAM-B3LYP"
    )
    workflow = WorkflowDefinition.objects.create(name="tda-opt-res", code_path="x.py")
    source = WorkflowRun.objects.create(workflow=workflow, settings_profile=profile, run_name="Au20_O2_res")
    ComputedProperty.objects.create(
        workflow_run=source,
        name="final_structure_snapshot",
        metadata={"symbols": ["Au", "O", "O"], "positions_A": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
    )
    HPCSystem.objects.create(name="FALCON", host="example.org", username="user")
    captured = []
    monkeypatch.setattr(
        "jobs.submission.submit_slurm_workflow_run",
        lambda **kwargs: captured.append(kwargs)
        or SimpleNamespace(workflow_run_id=45, job_record_id=46, scheduler_job_id="47"),
    )

    clean = tasks._validate_analysis_request(
        "targeted_tddft_optimisation",
        {
            "source_run_id": source.pk,
            "target_wavelength_nm": 560.1,
            "nroots": 20,
            "iroot": 6,
            "reservation": "SLURM_ACCOUNT_RES_4",
        },
    )
    result = tasks._submit_targeted_tddft_optimisation(**clean)

    assert result["scheduler_job_id"] == "47"
    job = captured[0]
    assert job["account"] == "SLURM_ACCOUNT"
    assert job["reservation"] == "SLURM_ACCOUNT_RES_4"
    assert job["qos"] == "reservation"
    assert job["use_tddft_optimisation"] is True
    assert job["target_wavelength_nm"] == 560.1
    assert job["single_constraints_config"] == {"tddft_scan_nroots": 20, "initial_iroot": 6}



@pytest.mark.django_db
def test_dft_optimisation_uses_falcon_profile(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(
        name="dft-opt", basis_set="LANL2DZ", xc_functional="CAM-B3LYP"
    )
    workflow = WorkflowDefinition.objects.create(name="dft-opt", code_path="x.py")
    source = WorkflowRun.objects.create(workflow=workflow, settings_profile=profile, run_name="Au20_intermediate")
    ComputedProperty.objects.create(
        workflow_run=source,
        name="final_structure_snapshot",
        metadata={"symbols": ["Au", "C", "O"], "positions_A": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
    )
    HPCSystem.objects.create(name="FALCON", host="example.org", username="user")
    captured = []
    monkeypatch.setattr(
        "jobs.submission.submit_slurm_workflow_run",
        lambda **kwargs: captured.append(kwargs)
        or SimpleNamespace(workflow_run_id=51, job_record_id=52, scheduler_job_id="53"),
    )

    result = tasks._submit_dft_optimisation(source.pk, hpc_name="FALCON")

    assert result["scheduler_job_id"] == "53"
    job = captured[0]
    assert (job["partition"], job["qos"], job["tasks_per_node"]) == ("htc_genoa", "", 48)
    assert job["account"] == "FALCON_SLURM_ACCOUNT"
    assert job["slurm_preset"] == "falcon_htc"


def test_single_optimisation_wrapper_honours_tddft_and_scan_root_count():
    script = _build_wrapper_script(
        workflow_module="cats.working_directory.tddft_active_learning_workflow",
        run_mode="single_optimisation",
        hpc_name="ARCHER2",
        n_iter=1,
        use_tddft_optimisation=True,
        fmax=0.01,
        active_learning_optimizer="bfgs",
        qm_backend="orca",
        qm_backend_settings={},
        target_wavelength_nm=611.4,
        basis_keyword="lanl2dz",
        xc_functional="cam-b3lyp",
        solvent=None,
        spin_multiplicity=1,
        nodes=1,
        tasks_per_node=128,
        cpus_per_task=1,
        al_steps_per_iter=200,
        max_al_steps_per_iter=None,
        single_constraints_config={"tddft_scan_nroots": 22, "initial_iroot": 22},
    )
    assert "use_tddft_optimisation=True" in script
    assert "ensure_initial_dft_converged=True" in script
    assert "tddft_scan_nroots=22" in script
    assert "initial_iroot=22" in script
    assert "target_wavelength_nm=611.4" in script
    assert "snapshot_single_point=False" in script


@pytest.mark.django_db
def test_excited_state_dimer_refinement_falcon(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(
        name="dft-opt", basis_set="LANL2DZ", xc_functional="CAM-B3LYP"
    )
    workflow = WorkflowDefinition.objects.create(
        name="active-learning-neb", code_path="cats/working_directory/active_learning_neb_workflow.py"
    )
    source = WorkflowRun.objects.create(workflow=workflow, settings_profile=profile, run_name="Au20_TS", status="completed")
    ComputedProperty.objects.create(
        workflow_run=source,
        name="final_structure_snapshot",
        metadata={"symbols": ["Au", "C", "O"], "positions_A": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
    )
    HPCSystem.objects.create(name="FALCON", host="example.org", username="user")
    captured = []
    monkeypatch.setattr(
        "jobs.submission.submit_slurm_workflow_run",
        lambda **kwargs: captured.append(kwargs)
        or SimpleNamespace(workflow_run_id=61, job_record_id=62, scheduler_job_id="63"),
    )

    clean = tasks._validate_analysis_request(
        "dimer_refine",
        {"source_run_id": source.pk, "hpc_name": "FALCON", "backend": "tddft", "root_state": 5, "tddft_scan_nroots": 50},
    )
    result = tasks._submit_dimer_refinement(**clean)

    assert result["scheduler_job_id"] == "63"
    job = captured[0]
    assert job["account"] == "FALCON_SLURM_ACCOUNT"
    assert job["partition"] == "htc_genoa"
    assert job["use_tddft_optimisation"] is True
    assert job["neb_config"]["backend"] == "tddft"
    assert job["neb_config"]["initial_iroot"] == 5
    assert job["neb_config"]["tddft_scan_nroots"] == 50
    assert "excited_s5" in job["run_name"]


def test_active_learning_neb_wrapper_honours_tddft_dimer():
    script = _build_wrapper_script(
        workflow_module="cats.working_directory.active_learning_neb_workflow",
        run_mode="active_learning_neb",
        hpc_name="FALCON",
        n_iter=1,
        use_tddft_optimisation=True,
        fmax=0.05,
        active_learning_optimizer="bfgs",
        qm_backend="orca",
        qm_backend_settings={},
        target_wavelength_nm=520.0,
        basis_keyword="lanl2dz_mixed",
        xc_functional="cam-b3lyp",
        solvent=None,
        spin_multiplicity=1,
        nodes=1,
        tasks_per_node=48,
        cpus_per_task=1,
        al_steps_per_iter=200,
        max_al_steps_per_iter=None,
        neb_config={
            "final_image_strategy": "dimer",
            "backend": "tddft",
            "initial_iroot": 5,
            "tddft_scan_nroots": 50,
        },
    )
    assert 'backend="tddft"' in script or "backend='tddft'" in script
    assert "initial_iroot=5" in script
    assert "tddft_scan_nroots=50" in script
