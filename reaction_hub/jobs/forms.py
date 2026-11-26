from __future__ import annotations

from pathlib import Path
import re

from django import forms
from django.conf import settings

from references.models import CalculationSettingsProfile, ReferenceAdsorbate, ReferenceNanocluster
from results.models import Artifact, ArtifactKind
from workflows.models import WorkflowDefinition

from .models import HPCSystem

_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_UPLOAD_STRUCTURE_SUFFIXES = (".xyz", ".extxyz", ".traj", ".db", ".cif", ".pdb")
_BATCH_STRUCTURE_SUFFIXES = {".traj", ".db"}
_CANONICAL_SUBMIT_WORKFLOWS: tuple[tuple[str, str, str], ...] = (
    (
        "TDDFT Active Learning",
        "cats/working_directory/tddft_active_learning_workflow.py",
        "Iterative TDDFT active-learning workflow for optimisation and bond scans.",
    ),
    (
        "Active Learning NEB (MACE)",
        "cats/working_directory/active_learning_neb_workflow.py",
        "CatLearn-assisted active-learning NEB workflow with MACE on GPU.",
    ),
    (
        "TDDFT MLNEB",
        "cats/working_directory/tddft_mlneb_workflow.py",
        "CatLearn MLNEB transition-state workflow with TDDFT state following.",
    ),
    (
        "MACE Adsorbate Screening",
        "similarity/workflows/coadsorbate_similarity.py",
        "MACE-based co-adsorbate placement and screening workflow.",
    ),
)
_CANONICAL_SUBMIT_SETTINGS_PROFILES: tuple[tuple[str, dict[str, object]], ...] = (
    (
        "nwchem_pbe0_def2svp",
        {
            "approved": True,
            "xc_functional": "PBE0",
            "basis_set": "def2-SVP",
            "solvent": "",
            "integration_grid": "fine",
            "scf_convergence": "TightSCF",
            "metadata": {
                "source": "submit_bootstrap",
                "qm_backend": "nwchem",
                "calculator": "nwchem",
                "nwchem": {
                    "version": "7.2.2",
                    "memory_mb": 1000,
                    "molden": True,
                },
            },
            "notes": "Default NWChem settings for online test submissions.",
        },
    ),
)


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    widget = MultipleFileInput

    def clean(self, data, initial=None):
        if not data:
            return []
        files = list(data) if isinstance(data, (list, tuple)) else [data]
        cleaned_files = []
        single_clean = super().clean
        for item in files:
            cleaned_files.append(single_clean(item, initial))
        return cleaned_files


class ReferenceAdsorbateSelectMultiple(forms.SelectMultiple):
    def create_option(self, name, value, label, selected, index, subindex=None, attrs=None):
        option = super().create_option(name, value, label, selected, index, subindex=subindex, attrs=attrs)
        instance = getattr(value, "instance", None)
        settings_profile_id = getattr(instance, "settings_profile_id", None)
        if settings_profile_id is not None:
            option.setdefault("attrs", {})["data-settings-profile-id"] = str(settings_profile_id)
        return option


def _filename_forces_vasp_format(filename: str) -> bool:
    basename = Path(str(filename or "")).name.lower()
    return "poscar" in basename or "contcar" in basename


def _workflow_code_path_is_neb(code_path: str) -> bool:
    path = str(code_path or "").strip().lower()
    return "active_learning_neb_workflow.py" in path or "tddft_mlneb_workflow.py" in path


def _workflow_code_path_is_tddft_mlneb(code_path: str) -> bool:
    return "tddft_mlneb_workflow.py" in str(code_path or "").strip().lower()


def _workflow_queryset_for_submit():
    _ensure_submit_workflow_definitions()
    workflows = list(WorkflowDefinition.objects.exclude(code_path="").order_by("name", "pk"))
    selected_ids: list[int] = []
    seen_code_paths: set[str] = set()

    for workflow in workflows:
        code_path = str(workflow.code_path or "").strip()
        if not code_path:
            continue
        normalized_code_path = code_path.lower()
        if normalized_code_path in seen_code_paths:
            continue
        seen_code_paths.add(normalized_code_path)
        selected_ids.append(int(workflow.pk))

    return WorkflowDefinition.objects.filter(pk__in=selected_ids).order_by("name")


def _ensure_submit_workflow_definitions() -> None:
    for workflow_name, code_path, description in _CANONICAL_SUBMIT_WORKFLOWS:
        normalized_code_path = str(code_path or "").strip()
        if not normalized_code_path:
            continue

        existing = (
            WorkflowDefinition.objects.filter(code_path=normalized_code_path)
            .order_by("name", "pk")
            .first()
        )
        if existing is not None:
            updates: list[str] = []
            if (
                str(existing.name or "").strip() != workflow_name
                and not WorkflowDefinition.objects.filter(name=workflow_name).exclude(pk=existing.pk).exists()
            ):
                existing.name = workflow_name
                updates.append("name")
            if description and not str(existing.description or "").strip():
                existing.description = description
                updates.append("description")
            if updates:
                existing.save(update_fields=[*updates, "updated_at"])
            continue

        create_name = workflow_name
        if WorkflowDefinition.objects.filter(name=create_name).exists():
            suffix = 2
            while WorkflowDefinition.objects.filter(name=f"{workflow_name} ({suffix})").exists():
                suffix += 1
            create_name = f"{workflow_name} ({suffix})"
        WorkflowDefinition.objects.create(
            name=create_name,
            code_path=normalized_code_path,
            description=description,
        )


def _ensure_submit_hpc_systems() -> None:
    for system_name, default_host in (("TRH408", "trh408"), ("FALCON", "falcon")):
        host = str(getattr(settings, f"REACTION_HUB_SUBMIT_{system_name}_HOST", default_host) or default_host).strip() or default_host
        port = int(getattr(settings, f"REACTION_HUB_SUBMIT_{system_name}_PORT", 22) or 22)
        username = str(getattr(settings, f"REACTION_HUB_SUBMIT_{system_name}_USERNAME", "") or "").strip()
        scheduler_type = "slurm"
        existing = (
            HPCSystem.objects.filter(name__iexact=system_name)
            .order_by("name", "pk")
            .first()
        )
        if existing is None:
            HPCSystem.objects.create(
                name=system_name,
                scheduler_type=scheduler_type,
                host=host,
                port=int(port),
                username=username,
                is_active=True,
            )
            continue

        updates: list[str] = []
        if (
            str(existing.name or "").strip() != system_name
            and not HPCSystem.objects.filter(name=system_name).exclude(pk=existing.pk).exists()
        ):
            existing.name = system_name
            updates.append("name")
        if str(existing.scheduler_type or "").strip().lower() != str(scheduler_type).strip().lower():
            existing.scheduler_type = scheduler_type
            updates.append("scheduler_type")
        if not str(existing.host or "").strip():
            existing.host = host
            updates.append("host")
        if not str(existing.username or "").strip():
            existing.username = username
            updates.append("username")
        if not int(getattr(existing, "port", 0) or 0):
            existing.port = int(port)
            updates.append("port")
        if not bool(existing.is_active):
            existing.is_active = True
            updates.append("is_active")
        if updates:
            existing.save(update_fields=[*updates, "updated_at"])


def _ensure_submit_settings_profiles() -> None:
    for profile_name, defaults in _CANONICAL_SUBMIT_SETTINGS_PROFILES:
        profile, created = CalculationSettingsProfile.objects.get_or_create(
            name=profile_name,
            defaults=defaults,
        )
        if created:
            continue
        metadata = profile.metadata if isinstance(profile.metadata, dict) else {}
        if str(metadata.get("qm_backend") or "").strip().lower() == "nwchem":
            continue
        if profile_name.startswith("nwchem_"):
            profile.metadata = {
                **metadata,
                **dict(defaults.get("metadata", {})),
            }
            profile.save(update_fields=["metadata", "updated_at"])


class RunMode:
    SINGLE_OPTIMISATION = "single_optimisation"
    TDDFT_BOND_SCAN = "tddft_bond_scan"
    CHARGE_TRANSFER = "charge_transfer"
    ACTIVE_LEARNING_NEB = "active_learning_neb"
    COADSORBATE_PLACEMENT = "coadsorbate_placement"
    CHOICES = (
        (SINGLE_OPTIMISATION, "Single optimisation"),
        (TDDFT_BOND_SCAN, "TDDFT bond-length scan workflow"),
        (CHARGE_TRANSFER, "Charge-transfer spectrum (Multiwfn)"),
        (ACTIVE_LEARNING_NEB, "Active-learning NEB / TDDFT MLNEB"),
        (COADSORBATE_PLACEMENT, "Co-adsorbate placement workflow"),
    )


class CoadsorbatePlacementMode:
    STANDARD = "standard"
    SILICATE_PORE = "silicate_pore"
    CHOICES = (
        (STANDARD, "Standard surface / cluster"),
        (SILICATE_PORE, "Silicate pore"),
    )


class ActiveLearningOptimizer:
    BFGS = "bfgs"
    FIRE2 = "fire2"
    CHOICES = (
        (BFGS, "BFGS"),
        (FIRE2, "FIRE2"),
    )


class QMBackend:
    ORCA = "orca"
    NWCHEM = "nwchem"
    CHOICES = (
        (ORCA, "ORCA"),
        (NWCHEM, "NWChem"),
    )


class FinalTddftNrootsMode:
    MANUAL = "manual"
    SCALED_BY_ATOMS = "scaled_by_atoms"
    CHOICES = (
        (MANUAL, "Manual"),
        (SCALED_BY_ATOMS, "Scale from atom count"),
    )


class NEBFinalImageStrategy:
    EXPLICIT = "explicit_final_image"
    SET_BOND_DISTANCE = "set_bond_distance_from_initial"
    LEGACY_BOND_EXTENSION = "bond_extension_from_initial"
    CHOICES = (
        (EXPLICIT, "Provide explicit final image"),
        (SET_BOND_DISTANCE, "Generate final by setting bond distance"),
    )


class GeometrySource:
    UPLOAD_FILE = "upload_file"
    ASE_MOLECULE = "ase_molecule"
    PUBCHEM = "pubchem"
    EXISTING_RUN = "existing_run"
    CHOICES = (
        (UPLOAD_FILE, "Upload structure file"),
        (ASE_MOLECULE, "ASE molecule builder"),
        (PUBCHEM, "PubChem lookup"),
        (EXISTING_RUN, "Use structure from existing workflow run"),
    )


class ReferenceIntent:
    NONE = "none"
    ADSORBATE = "adsorbate"
    NANOCLUSTER = "nanocluster"
    CHOICES = (
        (NONE, "Not a reference calculation"),
        (ADSORBATE, "Reference adsorbate calculation"),
        (NANOCLUSTER, "Reference nanocluster calculation"),
    )


def _split_multivalue(text: str) -> list[str]:
    if not text:
        return []
    parts: list[str] = []
    for raw in text.replace(",", "\n").splitlines():
        value = raw.strip()
        if value:
            parts.append(value)
    return parts


def _parse_positive_float_list(text: str, *, field_label: str) -> list[float]:
    values: list[float] = []
    seen: set[str] = set()
    for token in _split_multivalue(text):
        try:
            parsed = float(token)
        except ValueError as exc:
            raise forms.ValidationError(f"Invalid {field_label} value '{token}'.") from exc
        if parsed <= 0.0:
            raise forms.ValidationError(f"{field_label} values must be positive; got {token}.")
        normalized = f"{parsed:.12g}"
        if normalized in seen:
            raise forms.ValidationError(
                f"Duplicate {field_label} value '{normalized}' is not allowed."
            )
        seen.add(normalized)
        values.append(float(normalized))
    return values


def _parse_zero_based_indices(text: str, *, field_label: str) -> list[int]:
    raw_tokens = re.split(r"[\s,;]+", str(text or "").strip())
    values: list[int] = []
    seen: set[int] = set()
    for token in (item for item in raw_tokens if item):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if match is None:
            raise forms.ValidationError(
                f"Invalid {field_label} token '{token}'. Use 0-based indices such as 0-62, 63,64."
            )
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            raise forms.ValidationError(
                f"Invalid descending {field_label} range '{token}'."
            )
        for value in range(start, end + 1):
            if value in seen:
                raise forms.ValidationError(
                    f"Duplicate {field_label} atom index {value}."
                )
            seen.add(value)
            values.append(value)
    if not values:
        raise forms.ValidationError(f"Provide at least one {field_label} atom index.")
    return values


def _format_one_based_multiwfn_indices(indices: list[int] | tuple[int, ...]) -> str:
    values = sorted({int(value) + 1 for value in indices})
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


class JobSubmitForm(forms.Form):
    run_name = forms.CharField(max_length=120, help_text="Letters, numbers, dot, dash, underscore.")
    run_mode = forms.ChoiceField(
        choices=RunMode.CHOICES,
        initial=RunMode.SINGLE_OPTIMISATION,
        help_text="Choose optimisation, TDDFT scan, CT analysis, NEB, or co-adsorbate placement.",
    )
    workflow = forms.ModelChoiceField(queryset=WorkflowDefinition.objects.none())
    hpc_system = forms.ModelChoiceField(queryset=HPCSystem.objects.none())
    settings_profile = forms.ModelChoiceField(queryset=CalculationSettingsProfile.objects.none())
    reference_intent = forms.ChoiceField(
        choices=ReferenceIntent.CHOICES,
        initial=ReferenceIntent.NONE,
        required=False,
        help_text="Mark this run as a reusable reference for later adsorption-energy calculations.",
    )
    reference_name = forms.CharField(
        max_length=120,
        required=False,
        help_text="Reference name to create/reuse. If empty, run name is used.",
    )

    geometry_source = forms.ChoiceField(
        choices=GeometrySource.CHOICES,
        initial=GeometrySource.UPLOAD_FILE,
        help_text="Select where geometry should come from.",
    )
    structure_file = forms.FileField(
        required=False,
        help_text="Upload geometry file (.xyz, .traj, .db, .cif, .pdb, or VASP POSCAR/CONTCAR names).",
    )
    traj_frame = forms.IntegerField(
        required=False,
        initial=-1,
        help_text="Frame index for .traj/.db (default: -1, last frame).",
    )
    submit_all_traj_frames = forms.BooleanField(
        required=False,
        initial=False,
        help_text="For .traj/.db input, submit one run per frame as run_name_0, run_name_1, ...",
    )
    ase_molecule_formula = forms.CharField(
        required=False,
        help_text="ASE molecule name/formula or single atom symbol, for example CO, H2O, NH3, Cu.",
    )
    pubchem_cid = forms.IntegerField(
        required=False,
        min_value=1,
        help_text="Optional PubChem compound ID (CID), for example 18053.",
    )
    pubchem_query = forms.CharField(
        required=False,
        help_text="Optional PubChem search text (name or formula).",
    )
    source_run_id = forms.IntegerField(required=False, widget=forms.HiddenInput())
    source_artifact_id = forms.IntegerField(required=False, widget=forms.HiddenInput())
    ct_nanoparticle_indices = forms.CharField(
        required=False,
        initial="",
        help_text=(
            "ASE/Python 0-based fragment 1 indices. Metal atoms are inferred automatically; "
            "ranges are inclusive (for example 0-62)."
        ),
    )
    ct_adsorbate_indices = forms.CharField(
        required=False,
        initial="",
        help_text=(
            "ASE/Python 0-based fragment 2 indices. Non-metal atoms are inferred automatically "
            "and remain editable (for example 63,64)."
        ),
    )
    ct_nroots = forms.IntegerField(
        required=False,
        initial=400,
        min_value=1,
        help_text="Maximum excited-state count, prefilled from the source TDDFT calculation.",
    )
    ct_orca_basename = forms.CharField(
        required=False,
        initial="orca",
        max_length=80,
        help_text="ORCA output/wavefunction basename in the source run directory (normally 'orca').",
    )
    neb_final_geometry_source = forms.ChoiceField(
        required=False,
        choices=(
            (GeometrySource.UPLOAD_FILE, "Upload structure file"),
            (GeometrySource.ASE_MOLECULE, "ASE molecule builder"),
            (GeometrySource.PUBCHEM, "PubChem lookup"),
            (GeometrySource.EXISTING_RUN, "Use structure from existing workflow run"),
        ),
        initial=GeometrySource.UPLOAD_FILE,
        help_text="Final-image geometry source for NEB mode.",
    )
    neb_final_structure_file = forms.FileField(
        required=False,
        help_text="Final-image structure file (.xyz, .traj, .db, .cif, .pdb, or VASP POSCAR/CONTCAR names).",
    )
    neb_final_traj_frame = forms.IntegerField(
        required=False,
        initial=-1,
        help_text="Final-image frame index for .traj/.db (default: -1, last frame).",
    )
    neb_final_ase_molecule_formula = forms.CharField(
        required=False,
        help_text="Final-image ASE molecule name/formula or atom symbol.",
    )
    neb_final_pubchem_cid = forms.IntegerField(
        required=False,
        min_value=1,
        help_text="Final-image PubChem compound ID (CID).",
    )
    neb_final_pubchem_query = forms.CharField(
        required=False,
        help_text="Final-image PubChem query by name or formula.",
    )
    neb_final_source_run_id = forms.IntegerField(required=False, widget=forms.HiddenInput())
    neb_final_source_artifact_id = forms.IntegerField(required=False, widget=forms.HiddenInput())
    neb_final_image_strategy = forms.ChoiceField(
        required=False,
        choices=NEBFinalImageStrategy.CHOICES,
        initial=NEBFinalImageStrategy.EXPLICIT,
        help_text="Use an explicit final image, or generate one by setting a selected bond distance from the initial image.",
    )
    neb_optimize_initial_image = forms.BooleanField(
        required=False,
        initial=False,
        help_text="Optimise the initial endpoint before NEB.",
    )
    neb_optimize_final_image = forms.BooleanField(
        required=False,
        initial=False,
        help_text="Optimise the final endpoint before NEB.",
    )
    neb_endpoint_opt_steps = forms.IntegerField(
        required=False,
        initial=200,
        min_value=1,
        help_text="Maximum steps for endpoint optimisation passes.",
    )
    neb_endpoint_opt_fmax = forms.FloatField(
        required=False,
        min_value=1.0e-6,
        help_text="Optional endpoint optimisation fmax (leave empty to reuse workflow fmax).",
    )
    neb_break_bond_i = forms.IntegerField(
        required=False,
        min_value=0,
        help_text="0-based atom index A for generated set-bond-distance final image.",
    )
    neb_break_bond_j = forms.IntegerField(
        required=False,
        min_value=0,
        help_text="0-based atom index B for generated set-bond-distance final image.",
    )
    neb_break_bond_extension_A = forms.FloatField(
        required=False,
        initial=2.5,
        min_value=1.0e-6,
        help_text="Target bond distance in Angstrom for generated final image.",
    )
    neb_break_bond_fix = forms.FloatField(
        required=False,
        initial=0.5,
        min_value=0.0,
        max_value=1.0,
        help_text="ASE set_distance fix fraction used when setting the target bond distance.",
    )
    neb_break_assert_broken = forms.BooleanField(
        required=False,
        initial=True,
        help_text="Optional: assert that the chosen bond is absent after generated-final preparation.",
    )
    neb_break_cutoff_scale = forms.FloatField(
        required=False,
        initial=1.2,
        min_value=1.0e-6,
        help_text="Covalent-radii cutoff multiplier used for broken-bond assertion.",
    )
    neb_backend = forms.ChoiceField(
        required=False,
        choices=(("mace", "MACE"), ("dft", "DFT"), ("tddft_mlneb", "TDDFT MLNEB")),
        initial="mace",
        help_text="NEB calculator backend. TDDFT MLNEB requires the TDDFT MLNEB workflow.",
    )
    neb_atom_shift_per_image_A = forms.FloatField(
        required=False,
        initial=0.2,
        min_value=1.0e-4,
        help_text="Sampling density: max atom shift per image in Angstrom.",
    )
    neb_min_images = forms.IntegerField(
        required=False,
        initial=7,
        min_value=2,
        help_text="Minimum number of NEB images including endpoints.",
    )
    neb_max_images = forms.IntegerField(
        required=False,
        initial=21,
        min_value=2,
        help_text="Maximum number of NEB images including endpoints.",
    )
    neb_spring_constant = forms.FloatField(
        required=False,
        initial=0.1,
        min_value=1.0e-6,
        help_text="NEB spring constant k (eV/A^2).",
    )
    neb_max_steps = forms.IntegerField(
        required=False,
        initial=300,
        min_value=1,
        help_text="Maximum optimiser steps per NEB pass.",
    )
    neb_climb = forms.BooleanField(
        required=False,
        initial=True,
        help_text="Enable climbing-image NEB.",
    )
    neb_split_on_minimum = forms.BooleanField(
        required=False,
        initial=True,
        help_text="After convergence, split the path around validated interior minima.",
    )
    neb_minimum_energy_window_eV = forms.FloatField(
        required=False,
        initial=0.02,
        min_value=0.0,
        help_text="Minimum local-energy dip to treat an interior image as candidate minimum (eV).",
    )
    neb_mace_model_path = forms.CharField(
        required=False,
        initial=str(getattr(settings, "REACTION_HUB_NEB_MACE_MODEL_PATH", "") or ""),
        help_text="Path to the MACE model on HPC. Leave empty to use workflow default model.",
    )
    neb_mace_device = forms.ChoiceField(
        required=False,
        choices=(("cuda", "cuda"), ("cpu", "cpu")),
        initial="cuda",
        help_text="MACE device for NEB mode.",
    )
    neb_mace_default_dtype = forms.ChoiceField(
        required=False,
        choices=(("float64", "float64"), ("float32", "float32")),
        initial="float64",
        help_text="MACE default dtype.",
    )
    neb_mace_dispersion = forms.BooleanField(
        required=False,
        initial=False,
        help_text="Enable dispersion correction in MACE calculator.",
    )
    neb_catlearn_min_data = forms.IntegerField(
        required=False,
        initial=3,
        min_value=1,
        help_text="Minimum evaluated points before CatLearn local model starts.",
    )
    neb_catlearn_unc_convergence = forms.FloatField(
        required=False,
        initial=0.05,
        min_value=1.0e-6,
        help_text="CatLearn uncertainty convergence threshold.",
    )
    neb_catlearn_max_unc = forms.FloatField(
        required=False,
        initial=0.03,
        min_value=1.0e-6,
        help_text="Maximum allowed uncertainty during CatLearn optimisation.",
    )
    neb_catlearn_steps = forms.IntegerField(
        required=False,
        initial=200,
        min_value=1,
        help_text="Maximum expensive-evaluation optimisation steps in CatLearn refinement.",
    )
    neb_catlearn_ml_steps = forms.IntegerField(
        required=False,
        initial=500,
        min_value=1,
        help_text="Maximum surrogate optimisation steps in CatLearn refinement.",
    )
    neb_catlearn_dtrust = forms.FloatField(
        required=False,
        initial=2.0,
        min_value=1.0e-6,
        help_text="CatLearn trust distance (dtrust).",
    )

    coadsorbate_reference_cluster = forms.ModelChoiceField(
        required=False,
        queryset=ReferenceNanocluster.objects.none(),
        help_text=(
            "Optional reference nanocluster. Used as the base support geometry when no geometry-source "
            "structure is provided."
        ),
    )
    coadsorbate_placement_mode = forms.ChoiceField(
        required=False,
        choices=CoadsorbatePlacementMode.CHOICES,
        initial=CoadsorbatePlacementMode.STANDARD,
        help_text="Use silicate pore mode for Si/O/Al/H zeolite-like frameworks.",
    )
    coadsorbate_reference_adsorbates = forms.ModelMultipleChoiceField(
        required=False,
        queryset=ReferenceAdsorbate.objects.none(),
        widget=ReferenceAdsorbateSelectMultiple(attrs={"size": 6}),
        help_text="Optional adsorbates from saved reference adsorbates matching selected settings profile.",
    )
    coadsorbate_ase_molecules = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="ASE adsorbates (one per line or comma-separated), for example CO, NO2, H2O, Au.",
    )
    coadsorbate_pubchem_cids = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 2}),
        help_text="PubChem CID adsorbates (one per line or comma-separated).",
    )
    coadsorbate_pubchem_queries = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 2}),
        help_text="PubChem query adsorbates by name/formula (one per line or comma-separated).",
    )
    coadsorbate_structure_files = MultipleFileField(
        required=False,
        help_text=(
            "Upload one or more adsorbate files (.xyz, .traj, .db, .cif, .pdb, "
            "or VASP POSCAR/CONTCAR names)."
        ),
    )
    coadsorbate_mace_model_path = forms.CharField(
        required=False,
        initial=str(getattr(settings, "REACTION_HUB_COADSORBATE_MACE_MODEL_PATH", "") or ""),
        help_text="Path to MACE model file on HPC.",
    )
    coadsorbate_mace_fmax = forms.FloatField(
        required=False,
        initial=0.05,
        min_value=1.0e-6,
        help_text="MACE optimisation force threshold (eV/A).",
    )
    coadsorbate_mace_max_steps = forms.IntegerField(
        required=False,
        initial=200,
        min_value=1,
        help_text="Maximum MACE optimisation steps.",
    )
    coadsorbate_mace_device = forms.ChoiceField(
        required=False,
        choices=(("cuda", "cuda"), ("cpu", "cpu")),
        initial="cuda",
        help_text="MACE device.",
    )
    coadsorbate_mace_dispersion = forms.BooleanField(
        required=False,
        initial=False,
        help_text="Enable MACE dispersion corrections during relaxation.",
    )
    coadsorbate_mace_enable_cueq = forms.BooleanField(
        required=False,
        initial=True,
        help_text="Enable cuEquivariance acceleration in MACE when supported.",
    )
    coadsorbate_site_symbols = forms.CharField(
        required=False,
        initial="Ti,Au,Cu,O",
        help_text="Site symbols for co-adsorbate generation. Auto-populated from selected structure.",
    )
    coadsorbate_passive_symbols = forms.CharField(
        required=False,
        initial="Au,Ce,O,Cu",
        help_text=(
            "Passive/support symbols for co-adsorbate generation (treated as fixed support atoms). "
            "Auto-populated from selected structure."
        ),
    )
    coadsorbate_height = forms.FloatField(
        required=False,
        initial=2.0,
        min_value=0.0,
        help_text="Initial adsorbate placement height (Angstrom).",
    )
    coadsorbate_generation_z_cutoff = forms.FloatField(
        required=False,
        min_value=0.0,
        help_text=(
            "Absolute z-coordinate cutoff in Angstrom for generated adsorbates (periodic surfaces only). "
            "A candidate is kept only if all newly placed adsorbate atoms have z >= this value. "
            "Leave blank to disable."
        ),
    )
    coadsorbate_include_below_surface = forms.BooleanField(
        required=False,
        initial=False,
        help_text=(
            "Also place adsorbates below each top-layer site on carbonaceous "
            "periodic supports."
        ),
    )
    coadsorbate_origin_atom_index = forms.IntegerField(
        required=False,
        min_value=0,
        help_text="0-based input-structure atom index used as the active-site origin.",
    )
    coadsorbate_site_cutoff = forms.FloatField(
        required=False,
        min_value=1.0e-12,
        help_text="Keep only adsorption sites within this distance (Angstrom) of the origin atom.",
    )
    coadsorbate_relaxation_neighbor_shells = forms.IntegerField(
        required=False,
        min_value=0,
        help_text="Relax the origin atom and support atoms up to this nearest-neighbour shell.",
    )
    coadsorbate_silicate_origin_atom_index = forms.IntegerField(
        required=False,
        initial=0,
        min_value=0,
        help_text="Silicate pore origin atom index, usually the T-site atom.",
    )
    coadsorbate_silicate_pore_triangle_indices = forms.CharField(
        required=False,
        initial="1,2,3",
        help_text="Three 0-based O atom indices defining the pore-interior triangle.",
    )
    coadsorbate_silicate_site_atom_indices = forms.CharField(
        required=False,
        help_text="Optional site atom indices. Defaults to origin plus the triangle indices.",
    )
    coadsorbate_silicate_heights = forms.CharField(
        required=False,
        initial="1.45,1.70,1.95,2.50,3.00,3.50",
        help_text="Comma-separated initial insertion distances in Angstrom.",
    )
    coadsorbate_silicate_rotation_step_degrees = forms.FloatField(
        required=False,
        initial=60.0,
        min_value=1.0e-6,
        max_value=360.0,
        help_text="Rotation step around the pore vector.",
    )
    coadsorbate_silicate_clash_scale = forms.FloatField(
        required=False,
        initial=0.8,
        min_value=0.0,
        help_text="Reject contacts below this multiplier of covalent-radii sum.",
    )
    coadsorbate_silicate_push_step = forms.FloatField(
        required=False,
        initial=0.1,
        min_value=0.0,
        help_text="Inward push step used to resolve initial clashes.",
    )
    coadsorbate_silicate_max_push_steps = forms.IntegerField(
        required=False,
        initial=8,
        min_value=0,
        help_text="Maximum inward clash-resolution pushes.",
    )
    coadsorbate_keep_bound_only = forms.BooleanField(
        required=False,
        initial=False,
        help_text=(
            "After MACE relaxation, discard structures where an adsorbate component has no "
            "covalent-radius contact with the support."
        ),
    )
    coadsorbate_similarity_slab_shave_distance = forms.FloatField(
        required=False,
        min_value=0.0,
        help_text=(
            "Depth in Angstrom measured downward from the highest z among support species "
            "(periodic surfaces only). Support atoms deeper than this depth are shaved before "
            "similarity scoring. Leave blank to disable."
        ),
    )
    coadsorbate_support_species = forms.CharField(
        required=False,
        initial="Zn,Au,Cu,Ce,Ti,O",
        help_text=(
            "Support species used by similarity kernels. Auto-populated from selected structure. "
            "If slab shave distance is set, this species set defines the top-of-slab z reference."
        ),
    )
    coadsorbate_site_probe_symbol = forms.CharField(
        required=False,
        initial="H",
        max_length=2,
        help_text="Temporary element placed at each candidate site before site-similarity filtering.",
    )
    coadsorbate_site_probe_similarity_cutoff = forms.FloatField(
        required=False,
        initial=0.99,
        min_value=0.0,
        max_value=1.0,
        help_text="Probe-site similarities at or above this value are duplicates. Lower values prune more.",
    )
    coadsorbate_single_mode_duplicate_cutoff = forms.FloatField(
        required=False,
        initial=0.99,
        min_value=0.0,
        max_value=1.0,
        help_text="Duplicate cutoff for independently relaxed single-adsorbate modes.",
    )
    coadsorbate_candidate_duplicate_cutoff = forms.FloatField(
        required=False,
        min_value=0.0,
        max_value=1.0,
        help_text="Optional fixed cutoff for combined candidates. Leave blank for automatic selection.",
    )
    coadsorbate_relaxed_minimum_duplicate_cutoff = forms.FloatField(
        required=False,
        min_value=0.0,
        max_value=1.0,
        help_text="Optional fixed cutoff for relaxed minima. Leave blank for automatic selection.",
    )

    n_iter = forms.IntegerField(
        required=False,
        min_value=1,
        initial=10,
        help_text="Number of scan iterations (TDDFT scan mode only).",
    )
    use_tddft_optimisation = forms.BooleanField(
        required=False,
        initial=True,
        help_text="Use TDDFT for optimisation stages (TDDFT scan mode only).",
    )
    fmax = forms.FloatField(
        required=False,
        initial=0.01,
        min_value=1.0e-6,
        help_text="Force convergence threshold passed to workflow optimisation.",
    )
    active_learning_optimizer = forms.ChoiceField(
        required=False,
        choices=ActiveLearningOptimizer.CHOICES,
        initial=ActiveLearningOptimizer.BFGS,
        help_text="Local optimiser used by active learning (default: BFGS).",
    )
    qm_backend = forms.ChoiceField(
        required=False,
        choices=QMBackend.CHOICES,
        initial=QMBackend.ORCA,
        help_text=(
            "QM software backend for TDDFT/single-optimisation workflows. "
            "Choose the HPC system first so backend-specific module loads can be set. "
            "NWChem is enabled for single-optimisation test submissions; TDDFT "
            "active-learning scans still require ORCA."
        ),
    )
    target_wavelength_nm = forms.FloatField(
        required=False,
        initial=520.0,
        min_value=1.0e-6,
        help_text="Target excitation wavelength in nm used for TDDFT IROOT scoring.",
    )
    target_wavelengths_nm = forms.CharField(
        required=False,
        help_text=(
            "Optional comma-separated TDDFT wavelengths in nm to submit as separate jobs "
            "(for example: 500, 600, 700)."
        ),
    )
    tddft_scan_nroots = forms.IntegerField(
        required=False,
        initial=50,
        min_value=1,
        help_text="TDDFT scan sweep size (NRoots) used when selecting IROOT each scan step.",
    )
    single_final_tddft_spectrum = forms.BooleanField(
        required=False,
        initial=False,
        help_text="Run a TDDFT single-point spectrum on the final single-optimisation geometry.",
    )
    single_cluster_tddft_difference_spectrum = forms.BooleanField(
        required=False,
        initial=False,
        help_text=(
            "Also run an isolated-cluster TDDFT spectrum after optimisation and plot the "
            "cluster+adsorbate minus cluster difference spectrum."
        ),
    )
    single_charge_transfer_spectrum = forms.BooleanField(
        required=False,
        initial=False,
        help_text=(
            "Run Multiwfn charge-transfer analysis from the final TDDFT wavefunction in "
            "the same submitted job. This also enables the final TDDFT spectrum."
        ),
    )
    single_final_tddft_nroots = forms.IntegerField(
        required=False,
        initial=50,
        min_value=1,
        help_text="NRoots for the optional final TDDFT spectrum in single optimisation mode.",
    )
    single_final_tddft_nroots_mode = forms.ChoiceField(
        required=False,
        choices=FinalTddftNrootsMode.CHOICES,
        initial=FinalTddftNrootsMode.MANUAL,
        help_text="Manual, or scale from atom count: 13 atoms -> 50, 63 atoms -> 450, rounded up to 5.",
    )
    spin_multiplicity = forms.IntegerField(
        required=False,
        min_value=1,
        help_text=(
            "Optional spin multiplicity override. Leave empty to use automatic parity-based default "
            "(for example, set 3 for triplet O2)."
        ),
    )
    system_charge = forms.IntegerField(
        required=False,
        initial=0,
        help_text="Overall molecular charge passed to every optimization and spectrum calculation.",
    )
    bond_i = forms.IntegerField(
        required=False,
        min_value=0,
        help_text="0-based first atom index for optional bond constraint.",
    )
    bond_j = forms.IntegerField(
        required=False,
        min_value=0,
        help_text="0-based second atom index for optional bond constraint.",
    )
    increment = forms.FloatField(
        required=False,
        initial=0.05,
        help_text="Bond increment in Angstrom per iteration when target distance is not set.",
    )
    fix = forms.FloatField(
        required=False,
        initial=0.5,
        help_text="ASE set_distance fix parameter (0.0 to 1.0) when setting a target distance.",
    )
    fixed_symbols = forms.CharField(
        required=False,
        initial="Cu",
        help_text="Symbols to keep fixed during optimisation (comma-separated, e.g. Cu,Au).",
    )
    target_distance = forms.FloatField(
        required=False,
        min_value=0.0,
        help_text="Optional absolute bond distance in Angstrom for constrained bond runs.",
    )
    distance_change_fraction = forms.FloatField(
        required=False,
        initial=1.0,
        min_value=1.0e-6,
        help_text="Fraction of remaining distance to target used over remaining iterations.",
    )
    enable_ts_smoothing = forms.BooleanField(
        required=False,
        initial=False,
        help_text=(
            "After the scan, test a smoothed TS-adjacent geometry by taking the "
            "post-TS structure and resetting the scanned bond to the TS bond length."
        ),
    )
    refine_points_each_side = forms.IntegerField(
        required=False,
        initial=0,
        min_value=0,
        max_value=20,
        help_text=(
            "Optional number of extra constrained optimisation points on each side of the "
            "identified TS (0 disables refinement)."
        ),
    )
    restart_extend_steps = forms.IntegerField(
        required=False,
        initial=0,
        min_value=0,
        help_text=(
            "Optional restart-only extension count for TDDFT bond scans. "
            "When continuing from an existing run state, append this many extra scan points."
        ),
    )
    restart_extend_increment_A = forms.FloatField(
        required=False,
        help_text=(
            "Optional restart-only bond increment in Angstrom for appended points. "
            "If blank, the scan increment value is reused."
        ),
    )
    nodes = forms.IntegerField(min_value=1, initial=2)
    tasks_per_node = forms.IntegerField(min_value=1, initial=32)
    cpus_per_task = forms.IntegerField(min_value=1, initial=1)
    slurm_preset = forms.CharField(
        required=False,
        initial="custom",
        widget=forms.Select(choices=[("custom", "Custom (manual)")]),
        help_text="Choose a preset to auto-fill Slurm resources.",
    )

    walltime = forms.CharField(initial=getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_WALLTIME", "24:00:00"))
    account = forms.CharField(
        required=False,
        initial=getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_ACCOUNT", ""),
    )
    partition = forms.CharField(
        required=False,
        initial=getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_PARTITION", "standard"),
    )
    qos = forms.CharField(
        required=False,
        initial=getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_QOS", "standard"),
    )
    reservation = forms.CharField(
        required=False,
        initial=getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_RESERVATION", ""),
    )
    remote_runs_root = forms.CharField(
        initial=getattr(settings, "REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT", "/work/PROJECT/GROUP/{username}/reaction_hub_runs")
    )
    scratch_base = forms.CharField(
        required=False,
        initial=getattr(settings, "REACTION_HUB_SUBMIT_SCRATCH_BASE", "/scratch/PROJECT/GROUP")
    )

    module_loads = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        initial="\n".join(getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_MODULES", [])),
        help_text="One module per line (or comma-separated).",
    )
    venv_activate_path = forms.CharField(
        required=False,
        initial=getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_VENV_ACTIVATE_PATH", ""),
    )
    pythonpath_entries = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        initial="\n".join(getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_PYTHONPATH", [])),
        help_text="One PYTHONPATH entry per line (or comma-separated).",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["workflow"].queryset = _workflow_queryset_for_submit()
        _ensure_submit_hpc_systems()
        _ensure_submit_settings_profiles()
        self.fields["hpc_system"].queryset = HPCSystem.objects.filter(is_active=True, scheduler_type="slurm").order_by("name")
        self.fields["settings_profile"].queryset = CalculationSettingsProfile.objects.order_by("name")
        self.fields["coadsorbate_reference_cluster"].queryset = ReferenceNanocluster.objects.order_by("name")
        selected_settings_profile_id: int | None = None
        raw_settings_profile_id = (
            self.data.get("settings_profile")
            if self.is_bound
            else self.initial.get("settings_profile")
        )
        try:
            selected_settings_profile_id = int(raw_settings_profile_id)
        except (TypeError, ValueError):
            selected_settings_profile_id = None
        reference_adsorbate_qs = ReferenceAdsorbate.objects.select_related("settings_profile").order_by("name")
        if selected_settings_profile_id is not None:
            reference_adsorbate_qs = reference_adsorbate_qs.filter(settings_profile_id=selected_settings_profile_id)
        self.fields["coadsorbate_reference_adsorbates"].queryset = reference_adsorbate_qs
        self.fields["coadsorbate_reference_adsorbates"].label_from_instance = (
            lambda adsorbate: f"{adsorbate.name} [{adsorbate.settings_profile.name}]"
        )

        if not self.is_bound:
            mode_initial = str(self.initial.get("run_mode") or self.fields["run_mode"].initial or "").strip()
            if mode_initial == RunMode.SINGLE_OPTIMISATION:
                for field_name in (
                    "target_wavelength_nm",
                    "target_wavelengths_nm",
                    "increment",
                    "distance_change_fraction",
                    "enable_ts_smoothing",
                    "refine_points_each_side",
                    "restart_extend_steps",
                    "restart_extend_increment_A",
                ):
                    self.initial[field_name] = False if field_name == "enable_ts_smoothing" else ""
                self.initial["single_final_tddft_nroots"] = self.initial.get("single_final_tddft_nroots") or 50
                self.initial["fixed_symbols"] = ""
                self.initial["target_distance"] = ""
            elif mode_initial == RunMode.ACTIVE_LEARNING_NEB:
                workflow_initial = self.initial.get("workflow")
                workflow_path = str(getattr(workflow_initial, "code_path", "") or "").strip().lower()
                tddft_mlneb_initial = _workflow_code_path_is_tddft_mlneb(workflow_path)
                if tddft_mlneb_initial:
                    self.initial["neb_backend"] = "tddft_mlneb"
                    self.initial["target_wavelength_nm"] = self.initial.get("target_wavelength_nm") or 520.0
                    self.initial["tddft_scan_nroots"] = self.initial.get("tddft_scan_nroots") or 50
                    self.initial["target_wavelengths_nm"] = ""
                else:
                    self.initial["target_wavelength_nm"] = ""
                    self.initial["target_wavelengths_nm"] = ""
                    self.initial["tddft_scan_nroots"] = ""
                for field_name in (
                    "bond_i",
                    "bond_j",
                    "increment",
                    "fix",
                    "target_distance",
                    "distance_change_fraction",
                    "enable_ts_smoothing",
                    "refine_points_each_side",
                    "restart_extend_steps",
                    "restart_extend_increment_A",
                ):
                    self.initial[field_name] = False if field_name == "enable_ts_smoothing" else ""
                self.initial["fixed_symbols"] = ""
                if not str(self.initial.get("neb_final_image_strategy") or "").strip():
                    self.initial["neb_final_image_strategy"] = NEBFinalImageStrategy.EXPLICIT
            elif mode_initial == RunMode.COADSORBATE_PLACEMENT:
                for field_name in (
                    "bond_i",
                    "bond_j",
                    "increment",
                    "fix",
                    "fixed_symbols",
                    "target_distance",
                    "distance_change_fraction",
                    "enable_ts_smoothing",
                    "refine_points_each_side",
                    "restart_extend_steps",
                    "restart_extend_increment_A",
                    "target_wavelength_nm",
                    "target_wavelengths_nm",
                    "tddft_scan_nroots",
                ):
                    self.initial[field_name] = False if field_name == "enable_ts_smoothing" else ""

    def clean_run_name(self) -> str:
        value = self.cleaned_data["run_name"].strip()
        if not _RUN_NAME_RE.fullmatch(value):
            raise forms.ValidationError("Run name can only include letters, numbers, '.', '-', and '_'.")
        return value

    def clean_structure_file(self):
        uploaded = self.cleaned_data.get("structure_file")
        if uploaded is None:
            return None
        name = uploaded.name.lower()
        if (not name.endswith(_UPLOAD_STRUCTURE_SUFFIXES)) and (not _filename_forces_vasp_format(name)):
            raise forms.ValidationError(
                "Supported file types: .xyz, .traj, .db, .cif, .pdb, or VASP POSCAR/CONTCAR names."
            )
        return uploaded

    def clean_neb_final_structure_file(self):
        uploaded = self.cleaned_data.get("neb_final_structure_file")
        if uploaded is None:
            return None
        name = uploaded.name.lower()
        if (not name.endswith(_UPLOAD_STRUCTURE_SUFFIXES)) and (not _filename_forces_vasp_format(name)):
            raise forms.ValidationError(
                "Supported file types: .xyz, .traj, .db, .cif, .pdb, or VASP POSCAR/CONTCAR names."
            )
        return uploaded

    def clean_account(self) -> str:
        value = str(self.cleaned_data.get("account") or "").strip()
        hpc_system = self.cleaned_data.get("hpc_system")
        hpc_name = str(getattr(hpc_system, "name", "") or "").strip().lower()
        if (not value) and hpc_name != "trh408":
            raise forms.ValidationError("Account is required for Slurm submission.")
        return value

    def clean(self):
        cleaned_data = super().clean()
        cleaned_data["slurm_preset"] = str(cleaned_data.get("slurm_preset") or "custom").strip() or "custom"
        run_mode = cleaned_data.get("run_mode")
        reference_intent = str(cleaned_data.get("reference_intent") or ReferenceIntent.NONE).strip() or ReferenceIntent.NONE
        if reference_intent not in {ReferenceIntent.NONE, ReferenceIntent.ADSORBATE, ReferenceIntent.NANOCLUSTER}:
            raise forms.ValidationError("Unsupported reference intent.")
        reference_name = str(cleaned_data.get("reference_name") or "").strip()
        if reference_intent == ReferenceIntent.NONE:
            cleaned_data["reference_name"] = ""
        else:
            run_name = str(cleaned_data.get("run_name") or "").strip()
            cleaned_data["reference_name"] = reference_name or run_name
        cleaned_data["reference_intent"] = reference_intent
        if run_mode in {RunMode.COADSORBATE_PLACEMENT, RunMode.CHARGE_TRANSFER}:
            # Analysis and placement runs are not reference-tagging runs.
            cleaned_data["reference_intent"] = ReferenceIntent.NONE
            cleaned_data["reference_name"] = ""

        if cleaned_data.get("fmax") is None:
            cleaned_data["fmax"] = 0.01
        optimizer = str(cleaned_data.get("active_learning_optimizer") or ActiveLearningOptimizer.BFGS).strip().lower()
        if optimizer not in {ActiveLearningOptimizer.BFGS, ActiveLearningOptimizer.FIRE2}:
            raise forms.ValidationError("Unsupported active-learning optimizer. Choose BFGS or FIRE2.")
        cleaned_data["active_learning_optimizer"] = optimizer
        qm_backend = str(cleaned_data.get("qm_backend") or QMBackend.ORCA).strip().lower()
        if qm_backend not in {QMBackend.ORCA, QMBackend.NWCHEM}:
            raise forms.ValidationError(
                "Unsupported QM backend for online submission. Choose ORCA or NWChem."
            )
        if qm_backend == QMBackend.NWCHEM and run_mode != RunMode.SINGLE_OPTIMISATION:
            raise forms.ValidationError(
                "NWChem online submission is currently limited to single-optimisation test jobs."
            )
        cleaned_data["qm_backend"] = qm_backend
        if run_mode == RunMode.TDDFT_BOND_SCAN:
            raw_wavelength_batch = str(cleaned_data.get("target_wavelengths_nm") or "").strip()
            wavelength_batch: list[float] = []
            if raw_wavelength_batch:
                try:
                    wavelength_batch = _parse_positive_float_list(
                        raw_wavelength_batch,
                        field_label="target wavelength",
                    )
                except forms.ValidationError as exc:
                    self.add_error("target_wavelengths_nm", exc)
            if cleaned_data.get("target_wavelength_nm") is None:
                cleaned_data["target_wavelength_nm"] = 520.0
            if cleaned_data.get("tddft_scan_nroots") is None:
                cleaned_data["tddft_scan_nroots"] = 50
            if not wavelength_batch:
                wavelength_batch = [float(cleaned_data["target_wavelength_nm"])]
            cleaned_data["target_wavelength_nm"] = float(wavelength_batch[0])
            cleaned_data["target_wavelengths_nm"] = wavelength_batch
        elif run_mode != RunMode.ACTIVE_LEARNING_NEB:
            cleaned_data["target_wavelength_nm"] = None
            cleaned_data["target_wavelengths_nm"] = []
            if run_mode == RunMode.SINGLE_OPTIMISATION:
                cleaned_data["tddft_scan_nroots"] = cleaned_data.get("single_final_tddft_nroots") or 50
                nroots_mode = str(
                    cleaned_data.get("single_final_tddft_nroots_mode")
                    or FinalTddftNrootsMode.MANUAL
                ).strip()
                if nroots_mode not in {
                    FinalTddftNrootsMode.MANUAL,
                    FinalTddftNrootsMode.SCALED_BY_ATOMS,
                }:
                    raise forms.ValidationError("Unsupported final TDDFT NRoots mode.")
                cleaned_data["single_final_tddft_nroots_mode"] = nroots_mode
            else:
                cleaned_data["tddft_scan_nroots"] = None
            cleaned_data["restart_extend_steps"] = 0
            cleaned_data["restart_extend_increment_A"] = None
        else:
            cleaned_data["restart_extend_steps"] = 0
            cleaned_data["restart_extend_increment_A"] = None

        workflow = cleaned_data.get("workflow")
        workflow_code_path = str(getattr(workflow, "code_path", "") or "").strip().lower()
        workflow_is_coadsorbate = "coadsorbate" in workflow_code_path
        workflow_is_neb = _workflow_code_path_is_neb(workflow_code_path)
        workflow_is_tddft_mlneb = _workflow_code_path_is_tddft_mlneb(workflow_code_path)
        if run_mode == RunMode.COADSORBATE_PLACEMENT:
            if not workflow_is_coadsorbate:
                raise forms.ValidationError(
                    "Co-adsorbate mode requires a workflow definition with a coadsorbate workflow code_path."
                )
        elif run_mode == RunMode.ACTIVE_LEARNING_NEB:
            if not workflow_is_neb:
                raise forms.ValidationError(
                    "Active-learning NEB mode requires a workflow definition with the NEB workflow code_path."
                )
            if workflow_is_coadsorbate:
                raise forms.ValidationError(
                    "Selected workflow is for co-adsorbate placement. "
                    "Choose an active-learning NEB workflow for this run mode."
                )
        elif workflow_is_coadsorbate:
            raise forms.ValidationError(
                "Selected workflow is for co-adsorbate placement. "
                "Choose a TDDFT/single-optimisation workflow for this run mode."
            )
        elif workflow_is_neb:
            raise forms.ValidationError(
                "Selected workflow is for active-learning NEB. "
                "Choose a TDDFT/single-optimisation workflow for this run mode."
            )

        if run_mode == RunMode.COADSORBATE_PLACEMENT:
            cluster = cleaned_data.get("coadsorbate_reference_cluster")
            placement_mode = str(
                cleaned_data.get("coadsorbate_placement_mode")
                or CoadsorbatePlacementMode.STANDARD
            ).strip()
            if placement_mode not in {
                CoadsorbatePlacementMode.STANDARD,
                CoadsorbatePlacementMode.SILICATE_PORE,
            }:
                raise forms.ValidationError("Unsupported co-adsorbate placement mode.")
            cleaned_data["coadsorbate_placement_mode"] = placement_mode
            geometry_source = str(cleaned_data.get("geometry_source") or GeometrySource.UPLOAD_FILE).strip()
            structure_file = cleaned_data.get("structure_file")
            ase_molecule_formula = str(cleaned_data.get("ase_molecule_formula") or "").strip()
            pubchem_query = str(cleaned_data.get("pubchem_query") or "").strip()
            cleaned_data["ase_molecule_formula"] = ase_molecule_formula
            cleaned_data["pubchem_query"] = pubchem_query

            ase_adsorbates = _split_multivalue(str(cleaned_data.get("coadsorbate_ase_molecules") or ""))
            cid_tokens = _split_multivalue(str(cleaned_data.get("coadsorbate_pubchem_cids") or ""))
            query_adsorbates = _split_multivalue(str(cleaned_data.get("coadsorbate_pubchem_queries") or ""))
            selected_reference_adsorbates = list(cleaned_data.get("coadsorbate_reference_adsorbates") or [])
            file_adsorbates = cleaned_data.get("coadsorbate_structure_files") or []
            if not isinstance(file_adsorbates, (list, tuple)):
                file_adsorbates = [file_adsorbates]

            settings_profile = cleaned_data.get("settings_profile")
            if settings_profile is not None:
                mismatched_reference_adsorbates = [
                    adsorbate
                    for adsorbate in selected_reference_adsorbates
                    if adsorbate.settings_profile_id != settings_profile.pk
                ]
                if mismatched_reference_adsorbates:
                    mismatch_names = ", ".join(
                        str(adsorbate.name) for adsorbate in mismatched_reference_adsorbates
                    )
                    raise forms.ValidationError(
                        "Selected reference adsorbates do not match chosen settings profile: "
                        f"{mismatch_names}."
                    )

            cid_adsorbates: list[int] = []
            for token in cid_tokens:
                try:
                    cid = int(token)
                except ValueError as exc:
                    raise forms.ValidationError(f"Invalid PubChem CID '{token}' in co-adsorbate list.") from exc
                if cid <= 0:
                    raise forms.ValidationError(f"PubChem CID must be positive: {token}")
                cid_adsorbates.append(cid)

            normalized_files = []
            for uploaded in file_adsorbates:
                if uploaded is None:
                    continue
                name = str(getattr(uploaded, "name", "")).lower()
                if (not name.endswith(_UPLOAD_STRUCTURE_SUFFIXES)) and (not _filename_forces_vasp_format(name)):
                    raise forms.ValidationError(
                        f"Unsupported co-adsorbate file type for '{getattr(uploaded, 'name', '')}'. "
                        "Supported: .xyz, .traj, .db, .cif, .pdb, or VASP POSCAR/CONTCAR names."
                    )
                normalized_files.append(uploaded)

            total_adsorbates = (
                len(ase_adsorbates)
                + len(cid_adsorbates)
                + len(query_adsorbates)
                + len(normalized_files)
                + len(selected_reference_adsorbates)
            )
            if total_adsorbates <= 0:
                raise forms.ValidationError(
                    "Provide at least one adsorbate (reference adsorbate, ASE molecule, "
                    "PubChem CID/query, or uploaded file)."
                )

            use_reference_cluster_geometry = False
            if geometry_source == GeometrySource.UPLOAD_FILE:
                if structure_file is None:
                    if cluster is None:
                        raise forms.ValidationError("Structure file is required when geometry source is upload file.")
                    use_reference_cluster_geometry = True
            elif geometry_source == GeometrySource.EXISTING_RUN:
                if cleaned_data.get("source_run_id") is None or cleaned_data.get("source_artifact_id") is None:
                    if cluster is None:
                        raise forms.ValidationError("Select a source workflow run structure for this geometry mode.")
                    use_reference_cluster_geometry = True
            elif geometry_source == GeometrySource.ASE_MOLECULE:
                if not ase_molecule_formula:
                    if cluster is None:
                        raise forms.ValidationError("ASE molecule formula/name is required for ASE molecule source.")
                    use_reference_cluster_geometry = True
            elif geometry_source == GeometrySource.PUBCHEM:
                if cleaned_data.get("pubchem_cid") is None and not pubchem_query:
                    if cluster is None:
                        raise forms.ValidationError("Provide either PubChem CID or PubChem query.")
                    use_reference_cluster_geometry = True
            else:
                raise forms.ValidationError("Unsupported geometry source.")

            cleaned_data["coadsorbate_ase_molecules"] = ase_adsorbates
            cleaned_data["coadsorbate_pubchem_cids"] = cid_adsorbates
            cleaned_data["coadsorbate_pubchem_queries"] = query_adsorbates
            cleaned_data["coadsorbate_reference_adsorbates"] = selected_reference_adsorbates
            cleaned_data["coadsorbate_structure_files"] = normalized_files
            cleaned_data["coadsorbate_site_symbols"] = _split_multivalue(
                str(cleaned_data.get("coadsorbate_site_symbols") or "")
            )
            cleaned_data["coadsorbate_passive_symbols"] = _split_multivalue(
                str(cleaned_data.get("coadsorbate_passive_symbols") or "")
            )
            cleaned_data["coadsorbate_support_species"] = _split_multivalue(
                str(cleaned_data.get("coadsorbate_support_species") or "")
            )
            origin = cleaned_data.get("coadsorbate_origin_atom_index")
            cutoff = cleaned_data.get("coadsorbate_site_cutoff")
            relaxation_shells = cleaned_data.get("coadsorbate_relaxation_neighbor_shells")
            if origin is None and (cutoff is not None or relaxation_shells is not None):
                raise forms.ValidationError(
                    "Coadsorbate origin atom index is required for site cutoff or support relaxation."
                )
            if origin is not None and cutoff is None and relaxation_shells is None:
                raise forms.ValidationError(
                    "Coadsorbate origin atom index requires a site cutoff or support relaxation shells."
                )
            if placement_mode == CoadsorbatePlacementMode.SILICATE_PORE:
                triangle_tokens = _split_multivalue(
                    str(cleaned_data.get("coadsorbate_silicate_pore_triangle_indices") or "")
                )
                try:
                    triangle_indices = tuple(int(token) for token in triangle_tokens)
                except ValueError as exc:
                    raise forms.ValidationError("Silicate pore triangle indices must be integers.") from exc
                if len(triangle_indices) != 3 or any(idx < 0 for idx in triangle_indices):
                    raise forms.ValidationError("Silicate pore mode requires exactly three non-negative triangle indices.")
                site_tokens = _split_multivalue(
                    str(cleaned_data.get("coadsorbate_silicate_site_atom_indices") or "")
                )
                try:
                    site_indices = tuple(int(token) for token in site_tokens)
                except ValueError as exc:
                    raise forms.ValidationError("Silicate site atom indices must be integers.") from exc
                if any(idx < 0 for idx in site_indices):
                    raise forms.ValidationError("Silicate site atom indices must be non-negative.")
                try:
                    heights = tuple(
                        _parse_positive_float_list(
                            str(cleaned_data.get("coadsorbate_silicate_heights") or ""),
                            field_label="silicate placement height",
                        )
                    )
                except forms.ValidationError as exc:
                    self.add_error("coadsorbate_silicate_heights", exc)
                    heights = ()
                cleaned_data["coadsorbate_silicate_pore_triangle_indices"] = triangle_indices
                cleaned_data["coadsorbate_silicate_site_atom_indices"] = site_indices
                cleaned_data["coadsorbate_silicate_heights"] = heights
                if cleaned_data.get("coadsorbate_silicate_origin_atom_index") is None:
                    cleaned_data["coadsorbate_silicate_origin_atom_index"] = 0

            if cleaned_data.get("coadsorbate_mace_fmax") is None:
                cleaned_data["coadsorbate_mace_fmax"] = 0.05
            if cleaned_data.get("coadsorbate_mace_max_steps") is None:
                cleaned_data["coadsorbate_mace_max_steps"] = 200
            if cleaned_data.get("coadsorbate_height") is None:
                cleaned_data["coadsorbate_height"] = 2.0
            if not str(cleaned_data.get("coadsorbate_mace_device") or "").strip():
                cleaned_data["coadsorbate_mace_device"] = "cuda"

            cleaned_data["coadsorbate_use_reference_cluster_geometry"] = use_reference_cluster_geometry
            cleaned_data["submit_all_traj_frames"] = False
            cleaned_data["n_iter"] = 1
            cleaned_data["use_tddft_optimisation"] = False
            if cleaned_data.get("single_final_tddft_nroots") is None:
                cleaned_data["single_final_tddft_nroots"] = 50
            cleaned_data["single_final_tddft_nroots_mode"] = FinalTddftNrootsMode.MANUAL
            return cleaned_data

        geometry_source = cleaned_data.get("geometry_source")
        structure_file = cleaned_data.get("structure_file")
        ase_molecule_formula = str(cleaned_data.get("ase_molecule_formula") or "").strip()
        pubchem_query = str(cleaned_data.get("pubchem_query") or "").strip()
        cleaned_data["ase_molecule_formula"] = ase_molecule_formula
        cleaned_data["pubchem_query"] = pubchem_query

        if geometry_source == GeometrySource.UPLOAD_FILE:
            if structure_file is None:
                raise forms.ValidationError("Structure file is required when geometry source is upload file.")
            if cleaned_data.get("submit_all_traj_frames"):
                suffix = Path(structure_file.name).suffix.lower()
                if suffix not in _BATCH_STRUCTURE_SUFFIXES:
                    raise forms.ValidationError("Batch frame submission is only supported for .traj or .db files.")
        elif geometry_source == GeometrySource.EXISTING_RUN:
            if cleaned_data.get("source_run_id") is None or cleaned_data.get("source_artifact_id") is None:
                raise forms.ValidationError("Select a source workflow run structure for this geometry mode.")
            if cleaned_data.get("submit_all_traj_frames"):
                artifact = Artifact.objects.filter(
                    pk=cleaned_data.get("source_artifact_id"),
                    workflow_run_id=cleaned_data.get("source_run_id"),
                ).only("kind", "path").first()
                if artifact is None:
                    raise forms.ValidationError("Selected source artifact for batch submission was not found.")
                suffix = Path(str(artifact.path or "")).suffix.lower()
                if artifact.kind not in {ArtifactKind.TRAJECTORY, ArtifactKind.DATABASE} and suffix not in _BATCH_STRUCTURE_SUFFIXES:
                    raise forms.ValidationError(
                        "Batch submission from existing run requires a .traj/.db artifact."
                    )
        elif geometry_source == GeometrySource.ASE_MOLECULE:
            if not ase_molecule_formula:
                raise forms.ValidationError("ASE molecule formula/name is required for ASE molecule source.")
            cleaned_data["submit_all_traj_frames"] = False
        elif geometry_source == GeometrySource.PUBCHEM:
            if cleaned_data.get("pubchem_cid") is None and not pubchem_query:
                raise forms.ValidationError("Provide either PubChem CID or PubChem query.")
            cleaned_data["submit_all_traj_frames"] = False
        else:
            raise forms.ValidationError("Unsupported geometry source.")

        if run_mode == RunMode.ACTIVE_LEARNING_NEB and cleaned_data.get("submit_all_traj_frames"):
            raise forms.ValidationError("Batch frame submission is not supported for active-learning NEB mode.")
        charge_transfer_requested = run_mode == RunMode.CHARGE_TRANSFER or (
            run_mode == RunMode.SINGLE_OPTIMISATION
            and bool(cleaned_data.get("single_charge_transfer_spectrum"))
        )
        if charge_transfer_requested:
            if run_mode == RunMode.CHARGE_TRANSFER:
                if cleaned_data.get("submit_all_traj_frames"):
                    raise forms.ValidationError("Batch frame submission is not supported for charge-transfer analysis.")
                if geometry_source != GeometrySource.EXISTING_RUN:
                    raise forms.ValidationError(
                        "Standalone charge-transfer analysis must use an existing workflow run "
                        "containing ORCA output files."
                    )
            if qm_backend != QMBackend.ORCA:
                raise forms.ValidationError("Charge-transfer analysis currently requires an ORCA source calculation.")
            try:
                nanoparticle_indices = _parse_zero_based_indices(
                    cleaned_data.get("ct_nanoparticle_indices", ""),
                    field_label="nanoparticle fragment",
                )
            except forms.ValidationError as exc:
                self.add_error("ct_nanoparticle_indices", exc)
                nanoparticle_indices = []
            try:
                adsorbate_indices = _parse_zero_based_indices(
                    cleaned_data.get("ct_adsorbate_indices", ""),
                    field_label="adsorbate fragment",
                )
            except forms.ValidationError as exc:
                self.add_error("ct_adsorbate_indices", exc)
                adsorbate_indices = []
            overlap = sorted(set(nanoparticle_indices).intersection(adsorbate_indices))
            if overlap:
                raise forms.ValidationError(
                    "Nanoparticle and adsorbate fragments overlap at 0-based atom indices: "
                    + ", ".join(str(value) for value in overlap[:12])
                )
            basename = str(cleaned_data.get("ct_orca_basename") or "orca").strip() or "orca"
            if re.fullmatch(r"[A-Za-z0-9_.-]+", basename) is None or basename in {".", ".."}:
                self.add_error(
                    "ct_orca_basename",
                    "Use a filename basename containing only letters, numbers, dot, dash, or underscore.",
                )
            cleaned_data["ct_orca_basename"] = basename
            cleaned_data["ct_nanoparticle_indices_parsed"] = nanoparticle_indices
            cleaned_data["ct_adsorbate_indices_parsed"] = adsorbate_indices
            cleaned_data["ct_nroots"] = int(cleaned_data.get("ct_nroots") or 400)
            if run_mode == RunMode.SINGLE_OPTIMISATION:
                cleaned_data["single_final_tddft_spectrum"] = True

        if cleaned_data.get("submit_all_traj_frames") and reference_intent != ReferenceIntent.NONE:
            raise forms.ValidationError("Reference tagging is only supported for single-run submission.")
        if run_mode == RunMode.TDDFT_BOND_SCAN:
            wavelength_batch = list(cleaned_data.get("target_wavelengths_nm") or [])
            if cleaned_data.get("submit_all_traj_frames") and len(wavelength_batch) > 1:
                raise forms.ValidationError(
                    "Multi-wavelength submission cannot be combined with batch frame submission."
                )
            if len(wavelength_batch) > 1 and reference_intent != ReferenceIntent.NONE:
                raise forms.ValidationError(
                    "Reference tagging is only supported for single-wavelength TDDFT submission."
                )

        if run_mode == RunMode.ACTIVE_LEARNING_NEB:
            final_image_strategy = str(
                cleaned_data.get("neb_final_image_strategy") or NEBFinalImageStrategy.EXPLICIT
            ).strip()
            if final_image_strategy not in {
                NEBFinalImageStrategy.EXPLICIT,
                NEBFinalImageStrategy.SET_BOND_DISTANCE,
                NEBFinalImageStrategy.LEGACY_BOND_EXTENSION,
            }:
                raise forms.ValidationError("Unsupported NEB final-image strategy.")
            if final_image_strategy == NEBFinalImageStrategy.LEGACY_BOND_EXTENSION:
                final_image_strategy = NEBFinalImageStrategy.SET_BOND_DISTANCE
            cleaned_data["neb_final_image_strategy"] = final_image_strategy

            final_geometry_source = cleaned_data.get("neb_final_geometry_source")
            final_structure_file = cleaned_data.get("neb_final_structure_file")
            final_formula = str(cleaned_data.get("neb_final_ase_molecule_formula") or "").strip()
            final_pubchem_query = str(cleaned_data.get("neb_final_pubchem_query") or "").strip()
            cleaned_data["neb_final_ase_molecule_formula"] = final_formula
            cleaned_data["neb_final_pubchem_query"] = final_pubchem_query

            if final_image_strategy == NEBFinalImageStrategy.EXPLICIT:
                if final_geometry_source == GeometrySource.UPLOAD_FILE:
                    if final_structure_file is None:
                        raise forms.ValidationError("Final-image structure file is required for active-learning NEB mode.")
                elif final_geometry_source == GeometrySource.EXISTING_RUN:
                    final_source_run_id = cleaned_data.get("neb_final_source_run_id")
                    final_source_artifact_id = cleaned_data.get("neb_final_source_artifact_id")
                    if final_source_run_id is None or final_source_artifact_id is None:
                        if (
                            cleaned_data.get("source_run_id") is not None
                            and cleaned_data.get("source_artifact_id") is not None
                        ):
                            cleaned_data["neb_final_source_run_id"] = int(cleaned_data.get("source_run_id"))
                            cleaned_data["neb_final_source_artifact_id"] = int(cleaned_data.get("source_artifact_id"))
                        else:
                            raise forms.ValidationError(
                                "Select source run/artifact for the final image in active-learning NEB mode."
                            )
                elif final_geometry_source == GeometrySource.ASE_MOLECULE:
                    if not final_formula:
                        raise forms.ValidationError(
                            "Final-image ASE molecule formula/name is required for active-learning NEB mode."
                        )
                elif final_geometry_source == GeometrySource.PUBCHEM:
                    if cleaned_data.get("neb_final_pubchem_cid") is None and not final_pubchem_query:
                        raise forms.ValidationError(
                            "Provide final-image PubChem CID or query for active-learning NEB mode."
                        )
                else:
                    raise forms.ValidationError("Unsupported final-image geometry source.")
            else:
                break_bond_i = cleaned_data.get("neb_break_bond_i")
                break_bond_j = cleaned_data.get("neb_break_bond_j")
                if break_bond_i is None or break_bond_j is None:
                    raise forms.ValidationError(
                        "Set-bond-distance final-image generation requires both atom indices (A and B)."
                    )
                if int(break_bond_i) == int(break_bond_j):
                    raise forms.ValidationError(
                        "Set-bond atom indices must refer to two different atoms."
                    )
                if cleaned_data.get("neb_break_bond_extension_A") is None:
                    cleaned_data["neb_break_bond_extension_A"] = 2.5
                if cleaned_data.get("neb_break_bond_fix") is None:
                    cleaned_data["neb_break_bond_fix"] = 0.5
                if cleaned_data.get("neb_break_cutoff_scale") is None:
                    cleaned_data["neb_break_cutoff_scale"] = 1.2

            if cleaned_data.get("neb_atom_shift_per_image_A") is None:
                cleaned_data["neb_atom_shift_per_image_A"] = 0.2
            if cleaned_data.get("neb_min_images") is None:
                cleaned_data["neb_min_images"] = 7
            if cleaned_data.get("neb_max_images") is None:
                cleaned_data["neb_max_images"] = 21
            if int(cleaned_data.get("neb_min_images") or 0) > int(cleaned_data.get("neb_max_images") or 0):
                raise forms.ValidationError("NEB min images cannot exceed max images.")
            if cleaned_data.get("neb_spring_constant") is None:
                cleaned_data["neb_spring_constant"] = 0.1
            if cleaned_data.get("neb_max_steps") is None:
                cleaned_data["neb_max_steps"] = 300
            if cleaned_data.get("neb_minimum_energy_window_eV") is None:
                cleaned_data["neb_minimum_energy_window_eV"] = 0.02
            if cleaned_data.get("neb_catlearn_min_data") is None:
                cleaned_data["neb_catlearn_min_data"] = 3
            if cleaned_data.get("neb_catlearn_unc_convergence") is None:
                cleaned_data["neb_catlearn_unc_convergence"] = 0.05
            if cleaned_data.get("neb_catlearn_max_unc") is None:
                cleaned_data["neb_catlearn_max_unc"] = 0.03
            if cleaned_data.get("neb_catlearn_steps") is None:
                cleaned_data["neb_catlearn_steps"] = 200
            if cleaned_data.get("neb_catlearn_ml_steps") is None:
                cleaned_data["neb_catlearn_ml_steps"] = 500
            if cleaned_data.get("neb_catlearn_dtrust") is None:
                cleaned_data["neb_catlearn_dtrust"] = 2.0
            if cleaned_data.get("neb_endpoint_opt_steps") is None:
                cleaned_data["neb_endpoint_opt_steps"] = 200

            backend = str(cleaned_data.get("neb_backend") or "mace").strip().lower()
            if backend == "tddft":
                backend = "tddft_mlneb"
            if backend not in {"mace", "dft", "tddft_mlneb"}:
                raise forms.ValidationError("Unsupported NEB backend.")
            if workflow_is_tddft_mlneb and backend != "tddft_mlneb":
                raise forms.ValidationError("TDDFT MLNEB workflow requires the TDDFT MLNEB backend.")
            if (not workflow_is_tddft_mlneb) and backend == "tddft_mlneb":
                raise forms.ValidationError("TDDFT MLNEB backend requires the TDDFT MLNEB workflow.")
            cleaned_data["neb_backend"] = backend

            if not str(cleaned_data.get("neb_mace_device") or "").strip():
                cleaned_data["neb_mace_device"] = "cuda"
            if not str(cleaned_data.get("neb_mace_default_dtype") or "").strip():
                cleaned_data["neb_mace_default_dtype"] = "float64"

            if backend == "tddft_mlneb":
                if cleaned_data.get("target_wavelength_nm") is None:
                    cleaned_data["target_wavelength_nm"] = 520.0
                if cleaned_data.get("tddft_scan_nroots") is None:
                    cleaned_data["tddft_scan_nroots"] = 50
                cleaned_data["target_wavelengths_nm"] = []
            else:
                cleaned_data["target_wavelength_nm"] = None
                cleaned_data["target_wavelengths_nm"] = []
                cleaned_data["tddft_scan_nroots"] = None
            cleaned_data["submit_all_traj_frames"] = False
            cleaned_data["n_iter"] = 1
            cleaned_data["use_tddft_optimisation"] = False
            return cleaned_data

        if run_mode == RunMode.SINGLE_OPTIMISATION:
            bond_i = cleaned_data.get("bond_i")
            bond_j = cleaned_data.get("bond_j")
            has_bond_i = bond_i is not None
            has_bond_j = bond_j is not None
            if has_bond_i != has_bond_j:
                raise forms.ValidationError(
                    "Provide both bond indices (i and j) for single-optimisation bond constraints."
                )
            if has_bond_i and has_bond_j and bond_i == bond_j:
                raise forms.ValidationError("Bond indices must refer to two different atoms.")

            target_distance = cleaned_data.get("target_distance")
            if target_distance is not None and not (has_bond_i and has_bond_j):
                raise forms.ValidationError(
                    "Target distance requires both bond indices for single-optimisation constraints."
                )

            fix = cleaned_data.get("fix")
            if fix is None:
                cleaned_data["fix"] = 0.5
            elif not (0.0 <= float(fix) <= 1.0):
                raise forms.ValidationError("Fix must be between 0.0 and 1.0.")

            cleaned_data["n_iter"] = 1
            cleaned_data["use_tddft_optimisation"] = False
            return cleaned_data

        if run_mode == RunMode.CHARGE_TRANSFER:
            cleaned_data["n_iter"] = 1
            cleaned_data["use_tddft_optimisation"] = False
            return cleaned_data

        if run_mode != RunMode.TDDFT_BOND_SCAN:
            raise forms.ValidationError("Unsupported run mode.")

        bond_i = cleaned_data.get("bond_i")
        bond_j = cleaned_data.get("bond_j")
        if bond_i is None or bond_j is None:
            raise forms.ValidationError("Bond indices are required for TDDFT bond scan mode.")
        if bond_i == bond_j:
            raise forms.ValidationError("Bond indices must refer to two different atoms.")

        increment = cleaned_data.get("increment")
        target_distance = cleaned_data.get("target_distance")
        if increment is None:
            raise forms.ValidationError("Bond increment is required for TDDFT bond scan mode.")
        if target_distance is None and float(increment) == 0.0:
            raise forms.ValidationError("Increment cannot be zero when target distance is not set.")

        fix = cleaned_data.get("fix")
        if fix is None:
            raise forms.ValidationError("Fix value is required for TDDFT bond scan mode.")
        if not (0.0 <= float(fix) <= 1.0):
            raise forms.ValidationError("Fix must be between 0.0 and 1.0.")

        if cleaned_data.get("distance_change_fraction") is None:
            cleaned_data["distance_change_fraction"] = 1.0
        if cleaned_data.get("enable_ts_smoothing") is None:
            cleaned_data["enable_ts_smoothing"] = False
        if cleaned_data.get("refine_points_each_side") is None:
            cleaned_data["refine_points_each_side"] = 0
        if cleaned_data.get("restart_extend_steps") is None:
            cleaned_data["restart_extend_steps"] = 0

        restart_extend_steps = int(cleaned_data.get("restart_extend_steps") or 0)
        restart_extend_increment = cleaned_data.get("restart_extend_increment_A")
        if restart_extend_steps <= 0 and restart_extend_increment is not None:
            raise forms.ValidationError(
                "Restart extension increment requires restart extension steps greater than zero."
            )
        if restart_extend_steps > 0:
            effective_restart_increment = (
                float(restart_extend_increment)
                if restart_extend_increment is not None
                else float(cleaned_data.get("increment") or 0.0)
            )
            if abs(effective_restart_increment) <= 1.0e-12:
                raise forms.ValidationError(
                    "Restart extension increment cannot be zero when restart extension steps are provided."
                )
        else:
            cleaned_data["restart_extend_increment_A"] = None
        if cleaned_data.get("n_iter") is None:
            cleaned_data["n_iter"] = 10

        return cleaned_data

    def submission_lists(self) -> tuple[list[str], list[str]]:
        return _split_multivalue(self.cleaned_data.get("module_loads", "")), _split_multivalue(
            self.cleaned_data.get("pythonpath_entries", "")
        )

    def scan_config(self) -> dict[str, object]:
        if self.cleaned_data.get("run_mode") != RunMode.TDDFT_BOND_SCAN:
            return {}
        return {
            "bond_i": int(self.cleaned_data["bond_i"]),
            "bond_j": int(self.cleaned_data["bond_j"]),
            "increment": float(self.cleaned_data["increment"]),
            "fix": float(self.cleaned_data["fix"]),
            "fixed_symbols": tuple(_split_multivalue(self.cleaned_data.get("fixed_symbols", ""))),
            "target_distance": (
                float(self.cleaned_data["target_distance"]) if self.cleaned_data.get("target_distance") is not None else None
            ),
            "distance_change_fraction": float(self.cleaned_data.get("distance_change_fraction") or 1.0),
            "enable_ts_smoothing": bool(self.cleaned_data.get("enable_ts_smoothing")),
            "refine_points_each_side": int(self.cleaned_data.get("refine_points_each_side") or 0),
            "restart_extend_steps": int(self.cleaned_data.get("restart_extend_steps") or 0),
            "restart_extend_increment_A": (
                float(self.cleaned_data["restart_extend_increment_A"])
                if self.cleaned_data.get("restart_extend_increment_A") is not None
                else None
            ),
            "tddft_scan_nroots": int(self.cleaned_data.get("tddft_scan_nroots") or 50),
        }

    def single_constraint_config(self) -> dict[str, object]:
        if self.cleaned_data.get("run_mode") != RunMode.SINGLE_OPTIMISATION:
            return {}
        bond_i = self.cleaned_data.get("bond_i")
        bond_j = self.cleaned_data.get("bond_j")
        bond_pair_indices = None
        if bond_i is not None and bond_j is not None:
            bond_pair_indices = (int(bond_i), int(bond_j))
        return {
            "fixed_symbols": tuple(_split_multivalue(self.cleaned_data.get("fixed_symbols", ""))),
            "bond_pair_indices": bond_pair_indices,
            "target_distance": (
                float(self.cleaned_data["target_distance"])
                if self.cleaned_data.get("target_distance") is not None
                else None
            ),
            "fix": float(self.cleaned_data.get("fix") or 0.5),
            "final_tddft_spectrum": bool(self.cleaned_data.get("single_final_tddft_spectrum")),
            "cluster_tddft_difference_spectrum": bool(
                self.cleaned_data.get("single_cluster_tddft_difference_spectrum")
            ),
            "charge_transfer_spectrum": bool(
                self.cleaned_data.get("single_charge_transfer_spectrum")
            ),
            "final_tddft_nroots": int(self.cleaned_data.get("single_final_tddft_nroots") or 50),
            "final_tddft_nroots_mode": str(
                self.cleaned_data.get("single_final_tddft_nroots_mode")
                or FinalTddftNrootsMode.MANUAL
            ),
        }

    def charge_transfer_config(self) -> dict[str, object]:
        run_mode = self.cleaned_data.get("run_mode")
        if run_mode != RunMode.CHARGE_TRANSFER and not (
            run_mode == RunMode.SINGLE_OPTIMISATION
            and self.cleaned_data.get("single_charge_transfer_spectrum")
        ):
            return {}
        nanoparticle = [
            int(value)
            for value in self.cleaned_data.get("ct_nanoparticle_indices_parsed", [])
        ]
        adsorbate = [
            int(value)
            for value in self.cleaned_data.get("ct_adsorbate_indices_parsed", [])
        ]
        config: dict[str, object] = {
            "orca_basename": (
                str(self.cleaned_data.get("ct_orca_basename") or "orca")
                if run_mode == RunMode.CHARGE_TRANSFER
                else "orca"
            ),
            "nroots": int(
                self.cleaned_data.get("single_final_tddft_nroots") or 50
                if run_mode == RunMode.SINGLE_OPTIMISATION
                else self.cleaned_data.get("ct_nroots") or 400
            ),
            "fragments_0based": {
                "1": nanoparticle,
                "2": adsorbate,
            },
            "fragments_1based": {
                "1": _format_one_based_multiwfn_indices(nanoparticle),
                "2": _format_one_based_multiwfn_indices(adsorbate),
            },
            "fragment_labels": {"1": "NC", "2": "adsorbate"},
        }
        if run_mode == RunMode.CHARGE_TRANSFER:
            config["source_run_id"] = int(self.cleaned_data["source_run_id"])
        return config

    def coadsorbate_config(self) -> dict[str, object]:
        if self.cleaned_data.get("run_mode") != RunMode.COADSORBATE_PLACEMENT:
            return {}
        cluster = self.cleaned_data.get("coadsorbate_reference_cluster")
        return {
            "reference_cluster_id": int(cluster.pk) if cluster is not None else None,
            "placement_mode": str(
                self.cleaned_data.get("coadsorbate_placement_mode")
                or CoadsorbatePlacementMode.STANDARD
            ).strip(),
            "reference_adsorbate_ids": tuple(
                int(adsorbate.pk)
                for adsorbate in (self.cleaned_data.get("coadsorbate_reference_adsorbates") or [])
            ),
            "mace_model_path": str(self.cleaned_data.get("coadsorbate_mace_model_path") or "").strip(),
            "mace_fmax": float(self.cleaned_data.get("coadsorbate_mace_fmax") or 0.05),
            "mace_max_steps": int(self.cleaned_data.get("coadsorbate_mace_max_steps") or 200),
            "mace_device": str(self.cleaned_data.get("coadsorbate_mace_device") or "cuda").strip() or "cuda",
            "mace_dispersion": bool(self.cleaned_data.get("coadsorbate_mace_dispersion")),
            "mace_enable_cueq": bool(self.cleaned_data.get("coadsorbate_mace_enable_cueq")),
            "generation_site_symbols": tuple(self.cleaned_data.get("coadsorbate_site_symbols") or ()),
            "generation_passive_symbols": tuple(self.cleaned_data.get("coadsorbate_passive_symbols") or ()),
            "generation_height": float(self.cleaned_data.get("coadsorbate_height") or 2.0),
            "generation_z_cutoff": (
                float(self.cleaned_data["coadsorbate_generation_z_cutoff"])
                if self.cleaned_data.get("coadsorbate_generation_z_cutoff") is not None
                else None
            ),
            "generation_include_below_surface": bool(
                self.cleaned_data.get("coadsorbate_include_below_surface")
            ),
            "generation_origin_atom_index": self.cleaned_data.get("coadsorbate_origin_atom_index"),
            "generation_site_cutoff": self.cleaned_data.get("coadsorbate_site_cutoff"),
            "support_relaxation_origin_atom_index": (
                self.cleaned_data.get("coadsorbate_origin_atom_index")
                if self.cleaned_data.get("coadsorbate_relaxation_neighbor_shells") is not None
                else None
            ),
            "support_relaxation_neighbor_shells": self.cleaned_data.get(
                "coadsorbate_relaxation_neighbor_shells"
            ),
            "silicate_origin_atom_index": int(
                self.cleaned_data.get("coadsorbate_silicate_origin_atom_index") or 0
            ),
            "silicate_pore_triangle_indices": tuple(
                self.cleaned_data.get("coadsorbate_silicate_pore_triangle_indices") or (1, 2, 3)
            ),
            "silicate_site_atom_indices": tuple(
                self.cleaned_data.get("coadsorbate_silicate_site_atom_indices") or ()
            ),
            "silicate_heights": tuple(
                self.cleaned_data.get("coadsorbate_silicate_heights") or ()
            ),
            "silicate_rotation_step_degrees": float(
                self.cleaned_data.get("coadsorbate_silicate_rotation_step_degrees") or 60.0
            ),
            "silicate_clash_scale": float(
                self.cleaned_data.get("coadsorbate_silicate_clash_scale") or 0.8
            ),
            "silicate_push_step": float(
                self.cleaned_data.get("coadsorbate_silicate_push_step") or 0.1
            ),
            "silicate_max_push_steps": int(
                self.cleaned_data.get("coadsorbate_silicate_max_push_steps") or 8
            ),
            "keep_bound_only": bool(self.cleaned_data.get("coadsorbate_keep_bound_only")),
            "similarity_support_species": (
                ("Si", "O", "Al", "H")
                if str(
                    self.cleaned_data.get("coadsorbate_placement_mode")
                    or CoadsorbatePlacementMode.STANDARD
                ).strip()
                == CoadsorbatePlacementMode.SILICATE_PORE
                else tuple(self.cleaned_data.get("coadsorbate_support_species") or ())
            ),
            "similarity_slab_shave_distance": (
                float(self.cleaned_data["coadsorbate_similarity_slab_shave_distance"])
                if self.cleaned_data.get("coadsorbate_similarity_slab_shave_distance") is not None
                else None
            ),
            "candidate_reference_row_id": 1,
            "relaxed_minimum_reference_row_id": 1,
            "single_mode_duplicate_cutoff": float(
                0.99
                if self.cleaned_data.get("coadsorbate_single_mode_duplicate_cutoff") is None
                else self.cleaned_data["coadsorbate_single_mode_duplicate_cutoff"]
            ),
            "candidate_duplicate_cutoff": self.cleaned_data.get(
                "coadsorbate_candidate_duplicate_cutoff"
            ),
            "candidate_suggested_cutoff_scale": 1.0,
            "relaxed_minimum_duplicate_cutoff": self.cleaned_data.get(
                "coadsorbate_relaxed_minimum_duplicate_cutoff"
            ),
            "relaxed_minimum_suggested_cutoff_scale": 1.0,
            "site_probe_symbol": str(
                self.cleaned_data.get("coadsorbate_site_probe_symbol") or "H"
            ).strip(),
            "site_probe_similarity_cutoff": float(
                0.99
                if self.cleaned_data.get("coadsorbate_site_probe_similarity_cutoff") is None
                else self.cleaned_data["coadsorbate_site_probe_similarity_cutoff"]
            ),
            "require_converged": True,
        }

    def neb_config(self) -> dict[str, object]:
        if self.cleaned_data.get("run_mode") != RunMode.ACTIVE_LEARNING_NEB:
            return {}
        return {
            "final_image_strategy": str(
                self.cleaned_data.get("neb_final_image_strategy") or NEBFinalImageStrategy.EXPLICIT
            ).strip(),
            "optimize_initial_image": bool(self.cleaned_data.get("neb_optimize_initial_image")),
            "optimize_final_image": bool(self.cleaned_data.get("neb_optimize_final_image")),
            "endpoint_opt_steps": int(self.cleaned_data.get("neb_endpoint_opt_steps") or 200),
            "endpoint_opt_fmax": (
                float(self.cleaned_data["neb_endpoint_opt_fmax"])
                if self.cleaned_data.get("neb_endpoint_opt_fmax") is not None
                else None
            ),
            "break_bond_i": (
                int(self.cleaned_data["neb_break_bond_i"])
                if self.cleaned_data.get("neb_break_bond_i") is not None
                else None
            ),
            "break_bond_j": (
                int(self.cleaned_data["neb_break_bond_j"])
                if self.cleaned_data.get("neb_break_bond_j") is not None
                else None
            ),
            "set_bond_distance_A": float(self.cleaned_data.get("neb_break_bond_extension_A") or 2.5),
            "break_bond_fix": float(self.cleaned_data.get("neb_break_bond_fix") or 0.5),
            "break_assert_broken": bool(self.cleaned_data.get("neb_break_assert_broken")),
            "break_cutoff_scale": float(self.cleaned_data.get("neb_break_cutoff_scale") or 1.2),
            "backend": str(self.cleaned_data.get("neb_backend") or "mace").strip().lower(),
            "atom_shift_per_image_A": float(self.cleaned_data.get("neb_atom_shift_per_image_A") or 0.2),
            "min_images": int(self.cleaned_data.get("neb_min_images") or 7),
            "max_images": int(self.cleaned_data.get("neb_max_images") or 21),
            "spring_constant": float(self.cleaned_data.get("neb_spring_constant") or 0.1),
            "max_steps": int(self.cleaned_data.get("neb_max_steps") or 300),
            "climb": bool(self.cleaned_data.get("neb_climb")),
            "split_on_minimum": bool(self.cleaned_data.get("neb_split_on_minimum")),
            "minimum_energy_window_eV": float(self.cleaned_data.get("neb_minimum_energy_window_eV") or 0.02),
            "fixed_symbols": tuple(_split_multivalue(str(self.cleaned_data.get("fixed_symbols") or ""))),
            "mace_model_path": str(self.cleaned_data.get("neb_mace_model_path") or "").strip(),
            "mace_device": str(self.cleaned_data.get("neb_mace_device") or "cuda").strip() or "cuda",
            "mace_default_dtype": (
                str(self.cleaned_data.get("neb_mace_default_dtype") or "float64").strip() or "float64"
            ),
            "mace_dispersion": bool(self.cleaned_data.get("neb_mace_dispersion")),
            "catlearn_min_data": int(self.cleaned_data.get("neb_catlearn_min_data") or 3),
            "catlearn_unc_convergence": float(self.cleaned_data.get("neb_catlearn_unc_convergence") or 0.05),
            "catlearn_max_unc": float(self.cleaned_data.get("neb_catlearn_max_unc") or 0.03),
            "catlearn_steps": int(self.cleaned_data.get("neb_catlearn_steps") or 200),
            "catlearn_ml_steps": int(self.cleaned_data.get("neb_catlearn_ml_steps") or 500),
            "catlearn_dtrust": float(self.cleaned_data.get("neb_catlearn_dtrust") or 2.0),
            "tddft_scan_nroots": int(self.cleaned_data.get("tddft_scan_nroots") or 50),
        }

    def geometry_source_payload(self) -> dict[str, object]:
        return {
            "geometry_source": self.cleaned_data.get("geometry_source"),
            "structure_file": self.cleaned_data.get("structure_file"),
            "traj_frame": self.cleaned_data.get("traj_frame"),
            "ase_molecule_formula": self.cleaned_data.get("ase_molecule_formula"),
            "pubchem_cid": self.cleaned_data.get("pubchem_cid"),
            "pubchem_query": self.cleaned_data.get("pubchem_query"),
            "source_run_id": self.cleaned_data.get("source_run_id"),
            "source_artifact_id": self.cleaned_data.get("source_artifact_id"),
        }

    def neb_final_geometry_source_payload(self) -> dict[str, object]:
        return {
            "geometry_source": self.cleaned_data.get("neb_final_geometry_source"),
            "structure_file": self.cleaned_data.get("neb_final_structure_file"),
            "traj_frame": self.cleaned_data.get("neb_final_traj_frame"),
            "ase_molecule_formula": self.cleaned_data.get("neb_final_ase_molecule_formula"),
            "pubchem_cid": self.cleaned_data.get("neb_final_pubchem_cid"),
            "pubchem_query": self.cleaned_data.get("neb_final_pubchem_query"),
            "source_run_id": self.cleaned_data.get("neb_final_source_run_id"),
            "source_artifact_id": self.cleaned_data.get("neb_final_source_artifact_id"),
        }
