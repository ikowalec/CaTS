"""Excitation charge-density-difference (CDD) analysis with Multiwfn."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from cats.multiwfn.nto_povray import render_nto_cube_povray
from cats.multiwfn.tools import run_multiwfn, save_routine_config


def get_charge_density_difference_routine_config(
    path_to_molden: str,
    path_to_orca_out: str,
    excitation_number: int,
    *,
    grid_quality: int = 3,
) -> list[str]:
    """Return the Multiwfn 18-1 routine that exports ``CDD.cub``."""

    state = int(excitation_number)
    quality = int(grid_quality)
    if state < 1:
        raise ValueError("excitation_number must be at least 1")
    if quality not in {1, 2, 3}:
        raise ValueError("grid_quality must be 1, 2, or 3")
    return [
        str(path_to_molden),
        "18",
        "1",
        str(path_to_orca_out),
        f"{state}",
        "1",
        f"{quality}",
        "15",
        "0",
        "0",
        "0",
        "q",
    ]


def charge_density_difference_routine(
    path_to_molden: str,
    path_to_orca_out: str,
    excitation_number: int,
    *,
    output_prefix: str | None = None,
    grid_quality: int = 3,
    overwrite: bool = True,
    runner: Callable[[str, str], object] = run_multiwfn,
) -> dict[str, object]:
    """Generate a CDD cube where positive values are electron accumulation."""

    state = int(excitation_number)
    prefix = output_prefix or f"state_{state:03d}_cdd"
    cube_path = Path(f"{prefix}.cub")
    config_path = Path(f"{prefix}_config.txt")
    log_path = Path(f"{prefix}_log.txt")
    raw_cube = Path("CDD.cub")
    if overwrite:
        for path in (cube_path, config_path, log_path, raw_cube):
            path.unlink(missing_ok=True)
    if not cube_path.is_file():
        save_routine_config(
            get_charge_density_difference_routine_config(
                path_to_molden,
                path_to_orca_out,
                state,
                grid_quality=grid_quality,
            ),
            str(config_path),
        )
        runner(str(config_path), str(log_path))
        if not raw_cube.is_file():
            raise RuntimeError("Multiwfn did not create CDD.cub")
        raw_cube.replace(cube_path)
    return {
        "excitation_number": state,
        "cdd_cube": str(cube_path),
        "grid_quality": int(grid_quality),
        "sign_convention": "positive=electron accumulation; negative=electron depletion",
    }


def render_charge_density_difference_povray(
    cube_path: str | Path,
    output_png: str | Path,
    *,
    povray_executable: str | Path = "povray",
    title: str = "Excitation charge-density difference",
    isovalue: float = 0.001,
    rotation: str = "-8x,18y,8z",
) -> Path:
    """Render CDD in the manuscript view with blue electron and green hole density."""

    output_png = Path(output_png).resolve().with_suffix(".png")
    rendered = render_nto_cube_povray(
        cube_path,
        output_png,
        povray_executable=povray_executable,
        isovalue=isovalue,
        rotation=rotation,
        canvas_width=1800,
        positive_color=(0.08, 0.32, 0.90, 0.08),
        negative_color=(0.05, 0.65, 0.25, 0.08),
        material="ase3",
    )
    with Image.open(rendered).convert("RGB") as image:
        header = 96
        canvas = Image.new("RGB", (image.width, image.height + header), "white")
        canvas.paste(image, (0, header))
        draw = ImageDraw.Draw(canvas)
        try:
            title_font = ImageFont.truetype("arial.ttf", 38)
            label_font = ImageFont.truetype("arial.ttf", 28)
        except OSError:
            title_font = label_font = ImageFont.load_default()
        draw.text((canvas.width / 2, 10), title, fill="black", font=title_font, anchor="ma")
        draw.text(
            (canvas.width / 2, 58),
            "blue = electron accumulation   |   green = hole / electron depletion",
            fill="black",
            font=label_font,
            anchor="ma",
        )
        canvas.save(output_png)
    return output_png
