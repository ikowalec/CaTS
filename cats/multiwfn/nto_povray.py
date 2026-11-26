"""POV-Ray rendering for Multiwfn NTO cube files."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io.cube import read_cube
from ase.io.pov import POVRAY, POVRAYIsosurface, get_bondpairs, write_pov
from ase.io.utils import PlottingVariables
from PIL import Image, ImageDraw, ImageFont


class _SmoothPOVRAYIsosurface(POVRAYIsosurface):
    """ASE isosurface with interpolated vertex normals for smooth shading."""

    def compute_mesh(self, *args, **kwargs):
        mesh = super().compute_mesh(*args, **kwargs)
        self._normals = mesh[2]
        return mesh

    def format_mesh(self):
        mesh = super().format_mesh()
        normal_vectors = self.wrapped_triples_section(self._normals)
        return mesh.replace(
            "    face_indices {",
            f"    normal_vectors {{ {len(self._normals):n},\n"
            f"    {normal_vectors}\n    }}\n"
            "    face_indices {",
            1,
        )


def write_structure_pov_scene(
    atoms: Atoms,
    output_stem: str | Path,
    *,
    rotation: str = "-8x,18y,8z",
    canvas_width: int = 900,
    auto_bbox_size: float = 1.40,
) -> tuple[Path, Path]:
    """Write structure .pov and .ini scene with bonds and ASE textures without invoking POV-Ray."""

    output_stem = Path(output_stem).resolve()
    if output_stem.suffix.lower() in {".pov", ".ini", ".png"}:
        output_stem = output_stem.with_suffix("")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    centred = atoms.copy()
    centred.set_cell(None)
    centred.pbc = False
    centred.positions -= centred.positions.mean(axis=0)
    radii_scale = 0.62 if sum(centred.numbers == 79) <= 20 else 0.48
    radii = np.asarray([covalent_radii[number] for number in centred.numbers]) * radii_scale
    pov_path = output_stem.with_suffix(".pov")
    renderer = write_pov(
        str(pov_path),
        centred,
        rotation=rotation,
        radii=radii,
        auto_bbox_size=float(auto_bbox_size),
        povray_settings={
            "canvas_width": int(canvas_width),
            "transparent": False,
            "background": "White",
            "textures": ["ase3"] * len(centred),
            "bondatoms": get_bondpairs(centred, radius=1.05),
            "bondlinewidth": 0.07 if radii_scale < 0.6 else 0.10,
            "display": False,
            "pause": False,
        },
    )
    ini_path = pov_path.with_suffix(".ini")
    ini_text = ini_path.read_text(encoding="utf-8").replace(
        "Pause_When_Done=True", "Pause_When_Done=False"
    )
    lines = []
    for line in ini_text.splitlines():
        if line.startswith("Height="):
            lines.append(f"Height={int(round(float(line.split('=')[1])))}")
        else:
            lines.append(line)
    ini_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pov_path, ini_path


def render_pov_scene(
    ini_path: str | Path,
    output_png: str | Path | None = None,
    *,
    povray_executable: str | Path = "povray",
) -> Path:
    """Render a POV-Ray scene using an existing .ini configuration."""

    ini_path = Path(ini_path).resolve()
    if output_png is None:
        output_png = ini_path.with_suffix(".png")
    else:
        output_png = Path(output_png).resolve().with_suffix(".png")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    executable = str(povray_executable)
    if os.name == "nt" and executable == "povray" and not shutil.which(executable):
        for candidate in (
            r"C:\Program Files\POV-Ray\v3.7\bin\pvengine64.exe",
            r"C:\Program Files\POV-Ray\v3.7\bin\pvengine.exe",
            r"C:\Program Files (x86)\POV-Ray\v3.7\bin\pvengine.exe",
        ):
            if Path(candidate).is_file():
                executable = candidate
                break
    if os.name == "nt" and Path(executable).name.lower().startswith("pvengine"):
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\POV-Ray\v3.7\Windows") as key:
                winreg.QueryValueEx(key, "Home")
                winreg.QueryValueEx(key, "DocPath")
        except OSError as exc:
            raise RuntimeError(
                "POV-Ray user settings are unavailable in this process. "
                "Run rendering under the configured Windows user profile; "
                "launch stopped to prevent setup dialogs."
            ) from exc
    command = (
        [executable, "/NR", "/RENDER", str(ini_path), "/EXIT"]
        if os.name == "nt"
        else [executable, str(ini_path.with_suffix(".pov"))]
    )
    subprocess.run(command, cwd=output_png.parent, check=True)
    if not output_png.is_file():
        raise RuntimeError(f"POV-Ray did not create {output_png}")
    return output_png


def render_structure_povray(
    atoms: Atoms,
    output_png: str | Path,
    *,
    povray_executable: str | Path = "povray",
    rotation: str = "-8x,18y,8z",
    canvas_width: int = 900,
    auto_bbox_size: float = 1.40,
) -> Path:
    """Render a molecular structure with bonds using identical POV-Ray view settings."""

    output_png = Path(output_png).resolve().with_suffix(".png")
    ini_path = output_png.with_suffix(".ini")
    if not ini_path.is_file():
        _, ini_path = write_structure_pov_scene(
            atoms,
            output_png,
            rotation=rotation,
            canvas_width=canvas_width,
            auto_bbox_size=auto_bbox_size,
        )
    return render_pov_scene(ini_path, output_png, povray_executable=povray_executable)


def write_nto_cube_pov_scene(
    cube_path: str | Path,
    output_stem: str | Path,
    *,
    isovalue: float = 0.03,
    rotation: str = "-65x,0y,-25z",
    canvas_width: int = 1000,
    auto_bbox_size: float = 1.40,
    positive_color: tuple[float, float, float, float] = (0.10, 0.35, 0.95, 0.15),
    negative_color: tuple[float, float, float, float] = (0.95, 0.25, 0.10, 0.15),
    material: str = "glass2",
) -> tuple[Path, Path]:
    """Extract isosurface mesh and write .pov + .ini scene without invoking POV-Ray."""

    cube_path = Path(cube_path).resolve()
    output_stem = Path(output_stem).resolve()
    if output_stem.suffix.lower() in {".pov", ".ini", ".png"}:
        output_stem = output_stem.with_suffix("")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    with cube_path.open("r", encoding="utf-8") as cube_file:
        cube = read_cube(cube_file)
    data = cube["data"]
    atoms = cube["atoms"]
    # ASE retains the cube cell but does not apply its non-zero origin to the
    # POV-Ray mesh, so put both atoms and grid in the same origin-zero frame.
    atoms.positions -= cube["origin"]

    projection = PlottingVariables(
        atoms,
        scale=1.0,
        rotation=rotation,
        radii=0.35,
        show_unit_cell=1,
        auto_bbox_size=float(auto_bbox_size),
    )
    povray = POVRAY.from_PlottingVariables(
        projection,
        canvas_width=int(canvas_width),
        transparent=False,
        background="White",
        camera_type="orthographic",
        celllinewidth=0.0,
        textures=["ase3"] * len(atoms),
    )
    isosurfaces = []
    for grid, color in ((data, positive_color), (-data, negative_color)):
        if float(grid.max()) >= float(isovalue) and float(grid.min()) <= float(isovalue):
            try:
                isosurfaces.append(
                    _SmoothPOVRAYIsosurface.from_POVRAY(
                        povray,
                        density_grid=grid,
                        cut_off=float(isovalue),
                        color=color,
                        material=material,
                    )
                )
            except ValueError:
                pass
    povray.isosurfaces = isosurfaces
    pov_path = output_stem.with_suffix(".pov")
    renderer = povray.write(pov_path)
    ini_path = renderer.path.with_suffix(".ini")
    ini_path.write_text(
        ini_path.read_text(encoding="utf-8").replace(
            "Pause_When_Done=True", "Pause_When_Done=False"
        ),
        encoding="utf-8",
    )
    return pov_path, ini_path


def write_cdd_pov_scene(
    cube_path: str | Path,
    output_stem: str | Path,
    *,
    isovalue: float = 0.001,
    rotation: str = "-8x,18y,8z",
    canvas_width: int = 800,
    auto_bbox_size: float = 1.40,
    positive_color: tuple[float, float, float, float] = (0.08, 0.32, 0.90, 0.08),
    negative_color: tuple[float, float, float, float] = (0.05, 0.65, 0.25, 0.08),
    material: str = "ase3",
) -> tuple[Path, Path]:
    """Write .pov and .ini scene for a CDD cube."""
    return write_nto_cube_pov_scene(
        cube_path,
        output_stem,
        isovalue=isovalue,
        rotation=rotation,
        canvas_width=canvas_width,
        auto_bbox_size=auto_bbox_size,
        positive_color=positive_color,
        negative_color=negative_color,
        material=material,
    )


def render_nto_cube_povray(
    cube_path: str | Path,
    output_png: str | Path,
    *,
    povray_executable: str | Path = "povray",
    isovalue: float = 0.03,
    rotation: str = "-65x,0y,-25z",
    canvas_width: int = 1000,
    auto_bbox_size: float = 1.40,
    positive_color: tuple[float, float, float, float] = (0.10, 0.35, 0.95, 0.15),
    negative_color: tuple[float, float, float, float] = (0.95, 0.25, 0.10, 0.15),
    material: str = "glass2",
) -> Path:
    """Render the positive and negative phases of one NTO cube."""

    output_png = Path(output_png).resolve().with_suffix(".png")
    ini_path = output_png.with_suffix(".ini")
    if not ini_path.is_file():
        _, ini_path = write_nto_cube_pov_scene(
            cube_path,
            output_png,
            isovalue=isovalue,
            rotation=rotation,
            canvas_width=canvas_width,
            auto_bbox_size=auto_bbox_size,
            positive_color=positive_color,
            negative_color=negative_color,
            material=material,
        )
    return render_pov_scene(ini_path, output_png, povray_executable=povray_executable)


def export_density_analysis_pov_scenes(
    output_directory: str | Path,
    states: list[dict],
    *,
    rotation: str = "-8x,18y,8z",
    canvas_width: int = 900,
    auto_bbox_size: float = 1.40,
    cdd_isovalue: float = 0.001,
    nto_isovalue: float = 0.03,
    delete_cubes: bool = True,
) -> list[dict]:
    """Export self-contained POV-Ray scenes (.pov + .ini) and optionally delete bulky .cub files.

    Generates:
    - structure.pov / structure.ini (from first available cube)
    - state_XXX_cdd.pov / state_XXX_cdd.ini
    - state_XXX_hole.pov / state_XXX_hole.ini
    - state_XXX_particle.pov / state_XXX_particle.ini
    """

    out_dir = Path(output_directory).resolve()
    structure_written = (out_dir / "structure.ini").is_file()

    for state in states:
        state_id = int(state.get("excitation_number") or state.get("state") or 0)
        prefix = f"state_{state_id:03d}"

        # 1. CDD scene
        cdd_cube_name = state.get("cdd_cube")
        cdd_cube_path = (out_dir / Path(cdd_cube_name).name) if cdd_cube_name else None
        if cdd_cube_path and cdd_cube_path.is_file():
            if not structure_written:
                try:
                    with cdd_cube_path.open("r", encoding="utf-8") as f:
                        c_atoms = read_cube(f)["atoms"]
                    write_structure_pov_scene(
                        c_atoms,
                        out_dir / "structure",
                        rotation=rotation,
                        canvas_width=canvas_width,
                        auto_bbox_size=auto_bbox_size,
                    )
                    structure_written = True
                except Exception:
                    pass
            pov_p, ini_p = write_nto_cube_pov_scene(
                cdd_cube_path,
                out_dir / f"{prefix}_cdd",
                isovalue=cdd_isovalue,
                rotation=rotation,
                canvas_width=canvas_width,
                auto_bbox_size=auto_bbox_size,
                positive_color=(0.08, 0.32, 0.90, 0.08),
                negative_color=(0.05, 0.65, 0.25, 0.08),
                material="ase3",
            )
            state["cdd_pov_scene"] = str(Path(out_dir.name) / pov_p.name)
            state["cdd_ini"] = str(Path(out_dir.name) / ini_p.name)
            if (
                delete_cubes
                and pov_p.is_file()
                and ini_p.is_file()
                and pov_p.stat().st_size > 0
                and ini_p.stat().st_size > 0
            ):
                cdd_cube_path.unlink(missing_ok=True)
                state.pop("cdd_cube", None)

        # 2. Hole NTO scene
        hole_cube_name = state.get("nto_hole_cube")
        hole_cube_path = (out_dir / Path(hole_cube_name).name) if hole_cube_name else None
        if hole_cube_path and hole_cube_path.is_file():
            pov_p, ini_p = write_nto_cube_pov_scene(
                hole_cube_path,
                out_dir / f"{prefix}_hole",
                isovalue=nto_isovalue,
                rotation=rotation,
                canvas_width=canvas_width,
                auto_bbox_size=auto_bbox_size,
                positive_color=(0.10, 0.35, 0.95, 0.15),
                negative_color=(0.95, 0.25, 0.10, 0.15),
                material="glass2",
            )
            state["nto_hole_pov_scene"] = str(Path(out_dir.name) / pov_p.name)
            state["nto_hole_ini"] = str(Path(out_dir.name) / ini_p.name)
            if (
                delete_cubes
                and pov_p.is_file()
                and ini_p.is_file()
                and pov_p.stat().st_size > 0
                and ini_p.stat().st_size > 0
            ):
                hole_cube_path.unlink(missing_ok=True)
                state.pop("nto_hole_cube", None)

        # 3. Particle NTO scene
        particle_cube_name = state.get("nto_particle_cube")
        particle_cube_path = (out_dir / Path(particle_cube_name).name) if particle_cube_name else None
        if particle_cube_path and particle_cube_path.is_file():
            pov_p, ini_p = write_nto_cube_pov_scene(
                particle_cube_path,
                out_dir / f"{prefix}_particle",
                isovalue=nto_isovalue,
                rotation=rotation,
                canvas_width=canvas_width,
                auto_bbox_size=auto_bbox_size,
                positive_color=(0.10, 0.35, 0.95, 0.15),
                negative_color=(0.95, 0.25, 0.10, 0.15),
                material="glass2",
            )
            state["nto_particle_pov_scene"] = str(Path(out_dir.name) / pov_p.name)
            state["nto_particle_ini"] = str(Path(out_dir.name) / ini_p.name)
            if (
                delete_cubes
                and pov_p.is_file()
                and ini_p.is_file()
                and pov_p.stat().st_size > 0
                and ini_p.stat().st_size > 0
            ):
                particle_cube_path.unlink(missing_ok=True)
                state.pop("nto_particle_cube", None)

    return states



def render_nto_pair_povray(
    hole_cube: str | Path,
    particle_cube: str | Path,
    output_png: str | Path,
    *,
    povray_executable: str | Path = "povray",
    title: str = "Natural transition orbital pair",
    isovalue: float = 0.03,
) -> Path:
    """Render hole and particle cubes and combine them into one labelled image."""

    output_png = Path(output_png).resolve().with_suffix(".png")
    stem = output_png.with_suffix("")
    hole_png = render_nto_cube_povray(
        hole_cube,
        f"{stem}_hole.png",
        povray_executable=povray_executable,
        isovalue=isovalue,
    )
    particle_png = render_nto_cube_povray(
        particle_cube,
        f"{stem}_particle.png",
        povray_executable=povray_executable,
        isovalue=isovalue,
    )
    with Image.open(hole_png).convert("RGB") as hole, Image.open(particle_png).convert("RGB") as particle:
        header = 86
        canvas = Image.new("RGB", (hole.width + particle.width, hole.height + header), "white")
        canvas.paste(hole, (0, header))
        canvas.paste(particle, (hole.width, header))
        draw = ImageDraw.Draw(canvas)
        try:
            title_font = ImageFont.truetype("arial.ttf", 30)
            label_font = ImageFont.truetype("arial.ttf", 24)
        except OSError:
            title_font = label_font = ImageFont.load_default()
        draw.text((canvas.width / 2, 10), title, fill="black", font=title_font, anchor="ma")
        draw.text(
            (hole.width / 2, 52),
            "Density removed (hole NTO)",
            fill="black",
            font=label_font,
            anchor="ma",
        )
        draw.text(
            (hole.width + particle.width / 2, 52),
            "Density added (electron NTO)",
            fill="black",
            font=label_font,
            anchor="ma",
        )
        draw.text(
            (canvas.width - 18, 12),
            "blue / red = opposite phases",
            fill="black",
            font=label_font,
            anchor="ra",
        )
        canvas.save(output_png)
    return output_png
