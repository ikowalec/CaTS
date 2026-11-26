"""Continuous HPC-side MEP Excited-State Tracking & CDD Progression Engine.

Performs sequential image-by-image state following entirely on the HPC cluster:
1. Starts from root state at Image 00 (e.g. S25).
2. For each step i -> i+1:
   - Evaluates candidate states in image i+1 within energy/wavelength window.
   - Generates NTO pairs and CDD cubes using Multiwfn in parallel on node.
   - Computes 3D spatial NTO & CDD grid overlap via StateTracker.
   - Selects winning adiabatic counterpart and cleans non-winning cubes.
3. Renders winning CDD cubes with POV-Ray and compiles an animated GIF.
4. Produces publication plots and JSON summary of the complete MEP trajectory.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from cats.multiwfn.cdd import charge_density_difference_routine
from cats.multiwfn.nto import nto_routine
from cats.multiwfn.nto_povray import render_pov_scene, write_cdd_pov_scene
from cats.multiwfn.state_tracker import (
    MIN_ORBITAL_OVERLAP_SQ,
    OverlapResult,
    StateTracker,
    StateTrackPoint,
    generate_state_cubes,
)
from cats.orca.routines.orca_uv_extractor import extract_uv_transitions

logger = logging.getLogger(__name__)


class ContinuousMEPTracker:
    """End-to-end MEP excited state tracker designed to execute directly on HPC."""

    def __init__(
        self,
        image_directories: List[Path | str],
        *,
        root_state: int = 25,
        energy_window_ev: float = 0.35,
        output_directory: Path | str = "mep_tracking_output",
        density_workers: int = 4,
        nto_max_pairs: int = 3,
        nto_target_coverage: float = 0.8,
        grid_quality: int = 2,
    ):
        self.image_dirs = [Path(d).resolve() for d in image_directories]
        self.root_state = int(root_state)
        self.energy_window_ev = float(energy_window_ev)
        self.output_dir = Path(output_directory).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.density_workers = int(density_workers)
        self.nto_max_pairs = int(nto_max_pairs)
        self.nto_target_coverage = float(nto_target_coverage)
        self.grid_quality = int(grid_quality)

        self.tracker = StateTracker(
            root_state=self.root_state,
            energy_window_ev=self.energy_window_ev,
            mode="mep",
        )
        self.history: List[StateTrackPoint] = []
        self.step_overlap_matrix: List[List[Dict[str, Any]]] = []

    def _resolve_fragment_labels(self, image_dir: Path) -> Dict[int, str]:
        """Dynamically resolve fragment labels (e.g. Frag 1 (Cu13), Frag 2 (O), Frag 3 (CO))."""
        cfg_cand = [
            image_dir / "config.txt",
            image_dir / "DFT" / "final_tddft_spectrum" / "config.txt",
            image_dir / "final_tddft_spectrum" / "config.txt",
        ]
        frag_defs = []
        for cfg in cfg_cand:
            if cfg.is_file():
                try:
                    lines = cfg.read_text(encoding="utf-8", errors="replace").splitlines()
                    if len(lines) >= 5 and lines[1].strip() == "18":
                        n_frags = int(lines[3].strip())
                        frag_defs = [lines[4 + i].strip() for i in range(n_frags)]
                        break
                except Exception:
                    pass

        atoms = None
        for struct_name in ["input_structure.extxyz", "structure.extxyz", "orca.xyz"]:
            p = image_dir / struct_name
            if p.is_file():
                try:
                    from ase.io import read
                    atoms = read(str(p))
                    break
                except Exception:
                    pass

        labels = {}
        for idx, fdef in enumerate(frag_defs, start=1):
            frag_formula = ""
            if atoms is not None:
                indices = []
                for part in fdef.split(","):
                    part = part.strip()
                    if "-" in part:
                        s_idx, e_idx = part.split("-")
                        indices.extend(range(int(s_idx) - 1, int(e_idx)))
                    elif part.isdigit():
                        indices.append(int(part) - 1)
                valid_idx = [i for i in indices if 0 <= i < len(atoms)]
                if valid_idx:
                    frag_formula = f" ({atoms[valid_idx].get_chemical_formula()})"
            labels[idx] = f"Frag {idx}{frag_formula}"
        return labels

    def _get_transitions(self, image_dir: Path) -> List[Dict[str, Any]]:
        """Extract UV transitions from orca.out in image directory."""
        candidates = [
            image_dir / "orca.out",
            image_dir / "DFT" / "final_tddft_spectrum" / "orca.out",
            image_dir / "final_tddft_spectrum" / "orca.out",
        ]
        orca_out = next((p for p in candidates if p.is_file()), None)
        if not orca_out:
            raise FileNotFoundError(f"orca.out not found in {image_dir}")

        trans = extract_uv_transitions(str(orca_out))
        wl_list = (
            trans.wavelengths_nm
            if trans.wavelengths_nm is not None
            else 1.0e7 / trans.wavenumbers_cm
        )
        ct_data_map = {}
        frag_labels = self._resolve_fragment_labels(image_dir)

        # 1. Parse IFCTdata.txt if present in image directory
        ifct_candidates = [
            image_dir / "IFCTdata.txt",
            image_dir / "DFT" / "final_tddft_spectrum" / "IFCTdata.txt",
            image_dir / "final_tddft_spectrum" / "IFCTdata.txt",
        ]
        ifct_file = next((p for p in ifct_candidates if p.is_file()), None)
        if ifct_file:
            try:
                lines = ifct_file.read_text(encoding="utf-8", errors="replace").splitlines()
                header = lines[0].split()
                if not frag_labels:
                    max_k = 1
                    for col in header:
                        if col.startswith("hole(") and col.endswith(")"):
                            try:
                                max_k = max(max_k, int(col[5:-1]))
                            except Exception:
                                pass
                    frag_labels = {k: f"Frag {k}" for k in range(1, max_k + 1)}

                for l in lines[1:]:
                    parts = l.split()
                    if len(parts) == len(header) and parts[0].isdigit():
                        st = int(parts[0])
                        row = dict(zip(header, parts))
                        frag_shifts = {}
                        q_ct_leaving_1 = 0.0
                        for k, label in frag_labels.items():
                            h_val = float(row.get(f"hole({k})", 0.0))
                            e_val = float(row.get(f"ele({k})", 0.0))
                            frag_shifts[label] = e_val - h_val
                            if k > 1:
                                q_ct_leaving_1 += float(row.get(f"1->{k}", 0.0))

                        ct_data_map[st] = {
                            "q_ct": q_ct_leaving_1,
                            "fragment_q": frag_shifts,
                            "frag_labels": frag_labels,
                        }
            except Exception:
                pass

        # 2. Parse reaction_hub_summary.json if present
        for sum_cand in [image_dir / "reaction_hub_summary.json", image_dir / "summary.json"]:
            if sum_cand.is_file():
                try:
                    summary = json.loads(sum_cand.read_text(encoding="utf-8"))
                    ct_sec = summary.get("charge_transfer", {})
                    for entry in ct_sec.get("ct_states", []):
                        st = entry.get("state")
                        if st and int(st) not in ct_data_map:
                            q_val = entry.get("CT(e)") or entry.get("CT(%)") or 0.0
                            ct_data_map[int(st)] = {"q_ct": float(q_val)}
                    states_ct = ct_sec.get("states") or ct_sec.get("density_analysis", {}).get("states", {})
                    if isinstance(states_ct, dict):
                        for s_key, s_val in states_ct.items():
                            try:
                                s_int = int(str(s_key).lower().replace("state_", "").replace("s", ""))
                                if isinstance(s_val, dict) and s_int not in ct_data_map:
                                    ct_data_map[s_int] = s_val
                            except Exception:
                                pass
                except Exception:
                    pass
                break

        states = []
        for idx, (wl, f) in enumerate(zip(wl_list, trans.intensities), start=1):
            wl_val = float(wl)
            ev_val = 1239.841984 / wl_val if wl_val > 0 else 0.0
            states.append({
                "state": idx,
                "wavelength_nm": wl_val,
                "energy_ev": ev_val,
                "oscillator_strength": float(f),
                "ct_metrics": ct_data_map.get(idx, {}),
            })
        return states

    def _resolve_molden(self, image_dir: Path) -> Path:
        """Find molden input file or convert from gbw."""
        candidates = [
            image_dir / "orca.molden.input",
            image_dir / "orca.molden",
            image_dir / "DFT" / "final_tddft_spectrum" / "orca.molden.input",
            image_dir / "DFT" / "final_tddft_spectrum" / "orca.molden",
            image_dir / "final_tddft_spectrum" / "orca.molden.input",
            image_dir / "final_tddft_spectrum" / "orca.molden",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        gbw_candidates = [
            image_dir / "orca.gbw",
            image_dir / "DFT" / "final_tddft_spectrum" / "orca.gbw",
            image_dir / "final_tddft_spectrum" / "orca.gbw",
        ]
        gbw = next((p for p in gbw_candidates if p.is_file()), None)
        if gbw:
            local_molden = self.output_dir / f"{image_dir.name}_orca.molden.input"
            if local_molden.is_file():
                return local_molden
            import subprocess
            orca_2mkl = shutil.which("orca_2mkl")
            if not orca_2mkl:
                orcadir = str(os.environ.get("ORCADIR") or "").strip()
                if orcadir and (Path(orcadir) / "orca_2mkl").is_file():
                    orca_2mkl = str(Path(orcadir) / "orca_2mkl")
            if orca_2mkl:
                tmp_dir = self.output_dir / f"tmp_gbw_{image_dir.name}"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_gbw = tmp_dir / "orca.gbw"
                shutil.copy2(gbw, tmp_gbw)
                subprocess.run([orca_2mkl, "orca", "-molden"], cwd=tmp_dir, check=True)
                gen = tmp_dir / "orca.molden.input"
                if gen.is_file():
                    shutil.move(gen, local_molden)
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return local_molden
        raise FileNotFoundError(f"Neither orca.molden.input nor orca.gbw found in {image_dir}")

    def _resolve_orca_out(self, image_dir: Path) -> Path:
        """Find orca.out in image directory."""
        candidates = [
            image_dir / "orca.out",
            image_dir / "DFT" / "final_tddft_spectrum" / "orca.out",
            image_dir / "final_tddft_spectrum" / "orca.out",
        ]
        orca_out = next((p for p in candidates if p.is_file()), None)
        if not orca_out:
            raise FileNotFoundError(f"orca.out not found in {image_dir}")
        return orca_out

    def _generate_state_cubes(
        self,
        image_dir: Path,
        states: List[int],
        working_dir: Path,
    ) -> Dict[int, Dict[str, Any]]:
        """Run Multiwfn on HPC to generate NTO and CDD cubes for states."""
        return generate_state_cubes(
            image_dir,
            states,
            working_dir,
            density_workers=self.density_workers,
            nto_max_pairs=self.nto_max_pairs,
            nto_target_coverage=self.nto_target_coverage,
            grid_quality=self.grid_quality,
        )

    def _save_visualisation_cdd(self, image_dir: Path, state: int, step: int, tracking_cube: Path):
        """Keep fine visualisation cubes separate from tracking descriptors."""
        from cats.multiwfn.density_analysis import generate_excitation_density_analysis

        source = tracking_cube
        if self.grid_quality != 3:
            directory = self.output_dir / f"step_{step:02d}" / "visualisation"
            generate_excitation_density_analysis(
                self._resolve_molden(image_dir),
                self._resolve_orca_out(image_dir),
                [{"excitation_number": state}],
                output_directory=directory,
                excitation_numbers=[state],
                grid_quality=3,
                include_nto=False,
                export_pov_scenes=False,
                delete_cubes_after_export=False,
            )
            source = directory / f"state_{state:03d}_cdd.cub"
        shutil.copy2(source, self.output_dir / f"mep_cdd_image_{step:02d}.cub")

    def _compute_individual_ct(
        self,
        image_dir: Path,
        states: List[int],
        workdir: Path,
    ) -> Dict[int, Dict[str, float]]:
        """Run Multiwfn IFCT menu 18->8->1 for specific states ad-hoc."""
        molden = self._resolve_molden(image_dir)
        candidates = [
            image_dir / "orca.out",
            image_dir / "DFT" / "final_tddft_spectrum" / "orca.out",
            image_dir / "final_tddft_spectrum" / "orca.out",
        ]
        orca_out = next((p for p in candidates if p.is_file()), None)
        if not orca_out:
            return {}

        # Detect fragments from config.txt if available
        fragments = ["1-13", "14", "15-16"]
        cfg_cand = [
            image_dir / "config.txt",
            image_dir / "DFT" / "final_tddft_spectrum" / "config.txt",
            image_dir / "final_tddft_spectrum" / "config.txt",
        ]
        for cfg in cfg_cand:
            if cfg.is_file():
                try:
                    lines = cfg.read_text().splitlines()
                    if len(lines) >= 8 and lines[1].strip() == "18":
                        n_frags = int(lines[3].strip())
                        fragments = [lines[4 + i].strip() for i in range(n_frags)]
                        break
                except Exception:
                    pass

        multiwfn_cmd = shutil.which("multiwfn") or os.environ.get("MULTIWFN_DIR") or "multiwfn"
        res_map = {}
        for st in states:
            routine_config = [
                str(molden),
                "18",
                "8",
                "1",
                str(orca_out),
                f"{st}",
                f"{len(fragments)}",
            ]
            for frag in fragments:
                routine_config.append(frag)
            routine_config.extend(["0", "0", "q"])

            cfg_file = workdir / f"ct_adhoc_s{st}.txt"
            out_file = workdir / f"ct_adhoc_s{st}.out"
            cfg_file.write_text("\n".join(routine_config) + "\n")

            try:
                with open(cfg_file, "r") as stdin_f, open(out_file, "w") as stdout_f:
                    subprocess.run(
                        [multiwfn_cmd],
                        stdin=stdin_f,
                        stdout=stdout_f,
                        stderr=subprocess.DEVNULL,
                        cwd=workdir,
                        check=False,
                        timeout=45,
                    )
                if out_file.is_file():
                    out_txt = out_file.read_text(encoding="utf-8", errors="replace")
                    ct_pct = 0.0
                    q_ct = 0.0
                    for line in out_txt.splitlines():
                        if "Apparent charge transfer percentage, CT(%):" in line:
                            try:
                                ct_pct = float(line.split()[-2])
                            except Exception:
                                pass
                        elif "Net  " in line or "Interfragment charge transfer" in line or "CT amount" in line:
                            try:
                                q_ct = float(line.split()[-1])
                            except Exception:
                                pass
                    res_map[st] = {
                        "q_ct": q_ct if q_ct > 0 else ct_pct / 100.0,
                        "ct_pct": ct_pct,
                    }
            except Exception:
                pass
        return res_map

    def run_tracking(self) -> Dict[str, Any]:
        """Execute full tracking sequentially image-by-image on HPC."""
        n_images = len(self.image_dirs)
        print(f"=== Starting Continuous MEP State Tracking (Images 0 to {n_images - 1}) ===")
        print(f"Root state: S{self.root_state} at Image 00\n")

        # Step 0: Image 00 root state
        img0_dir = self.image_dirs[0]
        img0_trans = self._get_transitions(img0_dir)
        root_info = next(s for s in img0_trans if s["state"] == self.root_state)

        step0_workdir = self.output_dir / "step_00"
        step0_workdir.mkdir(parents=True, exist_ok=True)
        img0_cubes_map = self._generate_state_cubes(img0_dir, [self.root_state], step0_workdir)
        if self.root_state not in img0_cubes_map:
            raise RuntimeError(f"Failed to generate cubes for root S{self.root_state} on Image 00")

        current_cubes = img0_cubes_map[self.root_state]
        # Preserve root CDD cube
        self._save_visualisation_cdd(img0_dir, self.root_state, 0, current_cubes["cdd_path"])

        current_pt = StateTrackPoint(
            step_index=0,
            step_id=img0_dir.name,
            state_number=self.root_state,
            energy_ev=root_info["energy_ev"],
            wavelength_nm=root_info["wavelength_nm"],
            oscillator_strength=root_info["oscillator_strength"],
            ct_metrics=root_info.get("ct_metrics", {}),
            confidence="ROOT_SPECIFIED",
        )
        self.history.append(current_pt)
        root_ct = current_pt.ct_metrics.get("q_ct", 0.0)
        root_frag_q = current_pt.ct_metrics.get("fragment_q", {})
        sub_frags = [f"{k}: {v:+.3f}e" for k, v in root_frag_q.items() if not k.startswith("Frag 1")]
        root_frag_str = f" ({', '.join(sub_frags)})" if sub_frags else ""
        root_ct_str = f", q_CT={root_ct:+.3f}e{root_frag_str}" if (root_ct > 0 or sub_frags) else ""
        print(f"Step 00 ({img0_dir.name}): Root S{self.root_state} ({root_info['wavelength_nm']:.1f} nm, {root_info['energy_ev']:.3f} eV, f={root_info['oscillator_strength']:.4f}{root_ct_str})")

        # Sequential Loop: Image 1 to N-1
        for step in range(1, n_images):
            img_dir = self.image_dirs[step]
            step_workdir = self.output_dir / f"step_{step:02d}"
            step_workdir.mkdir(parents=True, exist_ok=True)

            all_states = self._get_transitions(img_dir)
            candidates = self.tracker.filter_energy_window(
                current_pt.energy_ev,
                all_states,
            )

            cand_numbers = [c["state"] for c in candidates]
            print(f"\nStep {step:02d} ({img_dir.name}): Evaluating candidates {cand_numbers} around S{current_pt.state_number} ({current_pt.energy_ev:.3f} eV)...")

            # Generate candidate cubes on HPC
            cand_cubes_map = self._generate_state_cubes(img_dir, cand_numbers, step_workdir)

            # Compute 3D NTO grid overlaps and pick winner (runs ad-hoc CT only for key top-tier states if needed)
            winner_pt, results = self.tracker.select_next_state(
                current_pt,
                current_cubes,
                candidates,
                cand_cubes_map,
                step_index=step,
                step_id=img_dir.name,
                ct_evaluator=lambda needed: self._compute_individual_ct(img_dir, needed, step_workdir),
            )
            self.history.append(winner_pt)
            self.step_overlap_matrix.append([
                {
                    "ref_state": r.ref_state,
                    "cand_state": r.cand_state,
                    "paired_score": r.paired_score,
                    "hole_ovlp": r.hole_overlap_sq,
                    "elec_ovlp": r.electron_overlap_sq,
                    "cdd_ovlp": r.cdd_overlap_sq,
                    "composite": r.composite_score,
                    "orbital_overlap_gate_passed": min(r.hole_overlap_sq, r.electron_overlap_sq) >= MIN_ORBITAL_OVERLAP_SQ,
                }
                for r in results
            ])

            cand_map = {int(c["state"]): c for c in candidates}
            w_ct = winner_pt.ct_metrics.get("q_ct", 0.0)
            w_frag_q = winner_pt.ct_metrics.get("fragment_q", {})
            w_sub = [f"{k}: {v:+.3f}e" for k, v in w_frag_q.items() if not k.startswith("Frag 1")]
            w_ct_str = f", q_CT={w_ct:+.3f}e ({', '.join(w_sub)})" if w_sub else (f", q_CT={w_ct:+.3f}e" if w_ct > 0 else "")
            print(f"  WINNER: S{winner_pt.state_number} ({winner_pt.wavelength_nm:.1f} nm, {winner_pt.energy_ev:.3f} eV, f={winner_pt.oscillator_strength:.4f}{w_ct_str}, confidence={winner_pt.confidence})")
            for r in results:
                c_ct = cand_map.get(r.cand_state, {}).get("ct_metrics", {})
                q_val = c_ct.get("q_ct", 0.0)
                frag_q = c_ct.get("fragment_q", {})
                sub_frags = [f"{k}: {v:+.3f}e" for k, v in frag_q.items() if not k.startswith("Frag 1")]
                frag_str = f" ({', '.join(sub_frags)})" if sub_frags else ""
                ct_str = f" | q_CT={q_val:+.3f}e{frag_str}" if (q_val > 0 or sub_frags) else ""
                print(f"    S{r.cand_state:02d}: Paired={r.paired_score*100:5.1f}% | Hole={r.hole_overlap_sq*100:5.1f}% | Spatial={r.spatial_score*100:5.1f}% | f_fac={r.intensity_score:4.2f}{ct_str} | Score={r.composite_score*100:5.1f}%")

            # Save winning CDD cube
            winning_cdd = cand_cubes_map[winner_pt.state_number]["cdd_path"]
            self._save_visualisation_cdd(img_dir, winner_pt.state_number, step, winning_cdd)

            # Set current for next step
            current_pt = winner_pt
            current_cubes = cand_cubes_map[winner_pt.state_number]

            # Clean up intermediate cubes from previous step directory to conserve scratch
            for st, cdata in cand_cubes_map.items():
                if st != winner_pt.state_number:
                    paths_to_clean = cdata.get("all_pair_paths", []) + [cdata.get("hole_path"), cdata.get("particle_path"), cdata.get("cdd_path")]
                    for p in paths_to_clean:
                        if p and Path(p).is_file():
                            try:
                                Path(p).unlink(missing_ok=True)
                            except Exception:
                                pass

        # Summary payload
        summary = {
            "root_state": self.root_state,
            "minimum_orbital_overlap_squared": MIN_ORBITAL_OVERLAP_SQ,
            "tracking_grid_quality": self.grid_quality,
            "visualisation_grid_quality": 3,
            "images_count": n_images,
            "trajectory": [
                {
                    "step": pt.step_index,
                    "image_name": pt.step_id,
                    "state": pt.state_number,
                    "energy_ev": pt.energy_ev,
                    "wavelength_nm": pt.wavelength_nm,
                    "oscillator_strength": pt.oscillator_strength,
                    "paired_score": pt.paired_score,
                    "hole_overlap_sq": pt.hole_overlap_sq,
                    "electron_overlap_sq": pt.electron_overlap_sq,
                    "cdd_overlap_sq": pt.cdd_overlap_sq,
                    "cluster_rmsd_A": pt.cluster_rmsd_A,
                    "confidence": pt.confidence,
                }
                for pt in self.history
            ],
            "overlap_matrix": self.step_overlap_matrix,
        }
        # Clean up intermediate candidate trial directories to prevent bloated artifact registration
        for p in self.output_dir.glob("step_*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)

        res_json = self.output_dir / "mep_state_tracking_results.json"
        res_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved tracking results to {res_json}")

        # Render CDD POV-Ray scenes and assemble animation
        try:
            self.render_cdd_animation()
        except Exception as exc:
            print(f"\nWarning: POV-Ray rendering skipped on cluster ({exc}); CDD cubes preserved for local render.")
        return summary

    def render_cdd_animation(self):
        """Render all winning CDD cubes using POV-Ray and compile animated GIF."""
        print("\n=== Rendering CDD Frames with POV-Ray ===")
        pov_frames = []
        for step in range(len(self.image_dirs)):
            cube_path = self.output_dir / f"mep_cdd_image_{step:02d}.cub"
            if not cube_path.is_file():
                continue
            png_path = self.output_dir / f"mep_cdd_image_{step:02d}.png"
            pov_path, ini_path = write_cdd_pov_scene(
                cube_path,
                png_path,
                isovalue=0.001,
                rotation="-8x,18y,8z",
                canvas_width=800,
            )
            try:
                render_pov_scene(ini_path, png_path)
                if png_path.is_file():
                    pov_frames.append(png_path)
                    print(f"  Rendered frame: {png_path.name}")
            except Exception as e:
                logger.warning("POV-Ray render failed for %s: %s", png_path, e)

        if len(pov_frames) >= 2:
            gif_path = self.output_dir / "mep_cdd_animation.gif"
            images = [Image.open(p) for p in pov_frames]
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=600,
                loop=0,
            )
            print(f"Saved animated GIF: {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="Continuous HPC MEP State Tracker")
    parser.add_argument("--root-state", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default="mep_tracking_output")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("image_dirs", nargs="+", help="Ordered list of MEP image calculation directories")
    args = parser.parse_args()

    tracker = ContinuousMEPTracker(
        args.image_dirs,
        root_state=args.root_state,
        output_directory=args.output_dir,
        density_workers=args.workers,
    )
    tracker.run_tracking()


if __name__ == "__main__":
    main()
