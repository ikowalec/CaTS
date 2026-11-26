from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ase import Atoms
from ase.io.pov import write_pov

_POVRAY_EXECUTABLE = r"C:\Program Files\POV-Ray\v3.7\bin\pvengine64.exe"

def find_povray_executable() -> str:
    candidates = [
        _POVRAY_EXECUTABLE,
        r"C:\Program Files\POV-Ray\v3.7\bin\pvengine.exe",
        r"C:\Program Files (x86)\POV-Ray\v3.7\bin\pvengine.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("POV-Ray executable not found on system.")

NONMETAL_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Ge", "As", "Se",
    "Br", "Sb", "Te", "I", "At", "Ts", "He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og",
}

def analyze_structure_connectivity(atoms: Atoms, cluster_formula: str = "") -> tuple[str, str]:
    """
    Resolve nanocluster chemical formula and adsorbate molecular components
    from atomic positions using ASE natural cutoffs and graph connected components.
    """
    if atoms is None or len(atoms) == 0:
        return "", ""
    from collections import Counter
    from ase.neighborlist import natural_cutoffs, NeighborList
    from scipy import sparse
    
    symbols = atoms.get_chemical_symbols()
    cluster_indices: list[int] = []
    if cluster_formula:
        try:
            from ase.formula import Formula

            remaining = Formula(cluster_formula).count()
            for index, symbol in enumerate(symbols):
                if remaining.get(symbol, 0) > 0:
                    cluster_indices.append(index)
                    remaining[symbol] -= 1
            if any(remaining.values()):
                cluster_indices = []
        except ValueError:
            cluster_indices = []
    if not cluster_indices:
        cluster_indices = [i for i, symbol in enumerate(symbols) if symbol not in NONMETAL_ELEMENTS]

    adsorbate_indices = [i for i in range(len(symbols)) if i not in set(cluster_indices)]
    resolved_cluster_formula = (
        cluster_formula or atoms[cluster_indices].get_chemical_formula()
        if cluster_indices
        else ""
    )
    
    if not adsorbate_indices:
        return resolved_cluster_formula, "none (clean surface)"
        
    adsorbate_atoms = atoms[adsorbate_indices]
    cutoffs = natural_cutoffs(adsorbate_atoms, mult=1.15)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(adsorbate_atoms)
    matrix = nl.get_connectivity_matrix()
    n_components, labels = sparse.csgraph.connected_components(matrix)
    
    adsorbate_mol_formulas = []
    for comp_id in range(n_components):
        comp_indices = [i for i, l in enumerate(labels) if l == comp_id]
        mol_symbols = [adsorbate_atoms[i].symbol for i in comp_indices]
        mol_counts = Counter(mol_symbols)
        # Hill system: C first, then H, then others alphabetical
        formula = ""
        elements = []
        if "C" in mol_counts:
            elements.append("C")
        if "H" in mol_counts:
            elements.append("H")
        elements.extend(sorted([k for k in mol_counts.keys() if k not in {"C", "H"}]))
        for el in elements:
            cnt = mol_counts[el]
            cnt_str = str(cnt) if cnt > 1 else ""
            formula += f"{el}{cnt_str}"
        adsorbate_mol_formulas.append(formula)
        
    adsorbate_counts = Counter(adsorbate_mol_formulas)
    parts = []
    for mol, cnt in sorted(adsorbate_counts.items()):
        prefix = f"{cnt} " if cnt > 1 else ""
        parts.append(f"{prefix}{mol}")
    adsorbate_str = " + ".join(parts)
    return resolved_cluster_formula, adsorbate_str

def compute_cumulative_atomic_shifts(images: Sequence[Atoms]) -> list[float]:
    shifts = [0.0]
    for i in range(1, len(images)):
        dr = images[i].positions - images[i - 1].positions
        step_dist = float(np.sqrt(np.sum(dr ** 2)))
        shifts.append(shifts[-1] + step_dist)
    return shifts

def render_mep_pingpong_gif(
    images: Sequence[Atoms],
    output_gif_path: str | Path,
    energies: Sequence[float] | None = None,
    title: str = "Minimum Energy Path (MEP)",
    rotation: str = "15x,25y,0z",
    target_height: int = 800,
    povray_canvas_width: int = 800,
    frame_duration_ms: int = 450,
    endpoint_pause_ms: int = 1000,
    dpi: int = 150,
) -> Path:
    output_gif = Path(output_gif_path).resolve()
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    povray_exe = find_povray_executable()
    
    n_images = len(images)
    if n_images < 2:
        raise ValueError("Need at least 2 images to render an MEP animation.")

    # Calculate cumulative atomic shifts
    shifts = compute_cumulative_atomic_shifts(images)
    
    # Calculate energies
    if energies is None or len(energies) != n_images:
        energies = []
        for img in images:
            if hasattr(img, "calc") and img.calc:
                energies.append(float(img.get_potential_energy()))
            elif "energy_eV" in img.info:
                energies.append(float(img.info["energy_eV"]))
            elif "energy" in img.info:
                energies.append(float(img.info["energy"]))
            else:
                energies.append(0.0)
    
    rel_energies = np.array(energies) - energies[0]
    min_e = float(np.min(rel_energies))
    max_e = float(np.max(rel_energies))
    e_span = max(max_e - min_e, 0.5)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_struct_images: list[Image.Image] = []
        
        # 1. Render POV-Ray structure for each frame
        for idx in range(n_images):
            atoms = images[idx]
            pov_file = tmp_path / f"frame_{idx:03d}.pov"
            ini_file = tmp_path / f"frame_{idx:03d}.ini"
            png_file = tmp_path / f"frame_{idx:03d}.png"
            
            # Write POV-Ray files via ASE
            write_pov(
                str(pov_file),
                atoms,
                rotation=rotation,
                povray_settings={
                    "transparent": False,
                    "canvas_width": povray_canvas_width,
                    "display": False,
                },
            )
            
            # Execute POV-Ray
            cmd = [povray_exe, "/RENDER", str(ini_file), "/EXIT"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if not png_file.exists():
                raise RuntimeError(f"POV-Ray failed to produce {png_file}")
            
            img = Image.open(png_file).convert("RGB")
            raw_struct_images.append(img)
        
        # 2. Scale structure images preserving exact aspect ratio to target_height
        scaled_struct_images: list[Image.Image] = []
        scaled_widths: list[int] = []
        for img in raw_struct_images:
            aspect = img.width / img.height
            new_w = int(round(target_height * aspect))
            scaled = img.resize((new_w, target_height), Image.Resampling.LANCZOS)
            scaled_struct_images.append(scaled)
            scaled_widths.append(new_w)
            
        max_struct_w = max(scaled_widths)
        
        # 3. Render High-DPI Matplotlib plots and composite side-by-side
        rendered_frames: list[Image.Image] = []
        
        # Plot dimensions in inches to match target_height at specified dpi
        plot_fig_h = target_height / dpi
        plot_fig_w = plot_fig_h * 1.15  # aesthetic rectangular plot ratio
        
        for idx in range(n_images):
            # Pad structure image to max_struct_w centered without deformation
            struct_canvas = Image.new("RGB", (max_struct_w, target_height), "#ffffff")
            s_img = scaled_struct_images[idx]
            offset_x = (max_struct_w - s_img.width) // 2
            struct_canvas.paste(s_img, (offset_x, 0))
            
            # High DPI publication plot
            plt.rcParams.update({
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 13,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "figure.titlesize": 13,
            })
            
            fig, ax = plt.subplots(figsize=(plot_fig_w, plot_fig_h), dpi=dpi)
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#f8fafc")
            
            # Smooth spline interpolation for MEP curve
            if len(shifts) > 3:
                try:
                    from scipy.interpolate import make_interp_spline
                    s_fine = np.linspace(shifts[0], shifts[-1], 300)
                    spl = make_interp_spline(shifts, rel_energies, k=3)
                    e_fine = spl(s_fine)
                    ax.plot(s_fine, e_fine, color="#2563eb", linewidth=2.8, zorder=2, label="MEP Spline")
                except Exception:
                    ax.plot(shifts, rel_energies, color="#2563eb", linewidth=2.8, zorder=2, label="MEP Path")
            else:
                ax.plot(shifts, rel_energies, color="#2563eb", linewidth=2.8, zorder=2, label="MEP Path")
            
            # All discrete MEP images (subtle markers)
            ax.scatter(
                shifts, rel_energies,
                color="#64748b", s=45, zorder=3,
                edgecolors="#1e293b", linewidth=1.0,
            )
            
            # Current image: BOLD RED SQUARE
            ax.scatter(
                [shifts[idx]], [rel_energies[idx]],
                color="#dc2626", marker="s", s=140, zorder=5,
                edgecolors="#000000", linewidth=1.8,
                label=f"Image {idx+1}/{n_images}",
            )
            
            # Peak/TS marker
            ts_idx = int(np.argmax(rel_energies))
            ax.scatter(
                [shifts[ts_idx]], [rel_energies[ts_idx]],
                facecolors="none", edgecolors="#dc2626", s=220,
                linestyle=":", linewidth=1.5, zorder=4,
            )
            
            ax.set_title(title, pad=12, fontweight="bold", color="#0f172a")
            ax.set_xlabel("Cumulative Atomic Shift (Å)", labelpad=8, fontweight="semibold")
            ax.set_ylabel("Relative Energy ΔE (eV)", labelpad=8, fontweight="semibold")
            ax.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")
            
            # Energy & shift annotation box
            curr_shift = shifts[idx]
            curr_de = rel_energies[idx]
            ax.annotate(
                f"Image: {idx+1}/{n_images}\nShift: {curr_shift:.2f} Å\nΔE: {curr_de:+.3f} eV ({curr_de*96.485:+.1f} kJ/mol)",
                xy=(0.04, 0.94),
                xycoords="axes fraction",
                verticalalignment="top",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.95, linewidth=1.2),
            )
            
            ax.set_ylim(min_e - 0.12 * e_span, max_e + 0.18 * e_span)
            ax.set_xlim(shifts[0] - 0.05 * (shifts[-1] - shifts[0]), shifts[-1] + 0.05 * (shifts[-1] - shifts[0]))
            
            plt.tight_layout()
            plot_png = tmp_path / f"plot_{idx:03d}.png"
            fig.savefig(plot_png, dpi=dpi)
            plt.close(fig)
            
            plot_img = Image.open(plot_png).convert("RGB")
            
            # Scale plot to exact target_height
            if plot_img.height != target_height:
                p_aspect = plot_img.width / plot_img.height
                plot_img = plot_img.resize((int(round(target_height * p_aspect)), target_height), Image.Resampling.LANCZOS)
            
            # Side-by-side composite: [Structure (undeformed) | Publication Plot]
            composite = Image.new("RGB", (struct_canvas.width + plot_img.width, target_height), "#ffffff")
            composite.paste(struct_canvas, (0, 0))
            composite.paste(plot_img, (struct_canvas.width, 0))
            
            rendered_frames.append(composite)
            
        # 4. Construct Ping-Pong animation: Forward + Reverse
        pingpong_frames: list[Image.Image] = []
        durations: list[int] = []
        
        # Forward sequence: 0 to N-1
        for i in range(n_images):
            pingpong_frames.append(rendered_frames[i])
            durations.append(endpoint_pause_ms if i in (0, n_images - 1) else frame_duration_ms)
        
        # Reverse sequence: N-2 down to 1
        for i in range(n_images - 2, 0, -1):
            pingpong_frames.append(rendered_frames[i])
            durations.append(frame_duration_ms)
        
        # Save high quality GIF
        pingpong_frames[0].save(
            str(output_gif),
            save_all=True,
            append_images=pingpong_frames[1:],
            optimize=True,
            duration=durations,
            loop=0,
        )
        print(f"Generated High-DPI Ping-Pong MEP GIF: {output_gif} ({len(pingpong_frames)} frames, {target_height}px height)")
        return output_gif
