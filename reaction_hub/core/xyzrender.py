from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import sys


def parse_canvas_size(value: object) -> tuple[int, int] | None:
    text = str(value or "").strip().lower().replace(" ", "")
    if not text:
        return None
    for separator in ("x", ",", ":"):
        if separator not in text:
            continue
        left, right = text.split(separator, 1)
        try:
            width = int(left)
            height = int(right)
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return width, height
    try:
        size = int(text)
    except (TypeError, ValueError):
        return None
    if size <= 0:
        return None
    return size, size


def normalize_gif_canvas(
    gif_path: Path,
    *,
    size: tuple[int, int] | None,
    background: tuple[int, int, int] = (255, 255, 255),
) -> tuple[bool, str]:
    if size is None:
        return False, ""
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        return False, ""
    try:
        from PIL import Image, ImageOps, ImageSequence
    except Exception as exc:
        return False, f"Pillow unavailable; GIF canvas was not normalized ({exc})."

    try:
        image = Image.open(gif_path)
        frames = []
        durations = []
        for frame in ImageSequence.Iterator(image):
            duration = int(frame.info.get("duration", image.info.get("duration", 100)) or 100)
            durations.append(duration)
            rgba = frame.convert("RGBA")
            fitted = ImageOps.contain(rgba, (width, height))
            canvas = Image.new("RGB", (width, height), background)
            paste_xy = (
                (width - fitted.width) // 2,
                (height - fitted.height) // 2,
            )
            if fitted.mode == "RGBA":
                canvas.paste(fitted, paste_xy, fitted)
            else:
                canvas.paste(fitted, paste_xy)
            frames.append(canvas)
        if not frames:
            return False, "GIF canvas was not normalized; no frames were found."
        loop = int(image.info.get("loop", 0) or 0)
        image.close()
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=loop,
            disposal=[2] * len(frames),
            background=0,
            optimize=False,
        )
    except Exception as exc:
        return False, f"GIF canvas normalization failed: {exc}"
    return True, f"GIF canvas normalized to {width}x{height}px with full-frame disposal."


def _is_bare_python_launcher(token: str) -> bool:
    value = str(token or "").strip().lower()
    return value in {"python", "python3", "python.exe", "python3.exe", "py", "py.exe"}


def _xyzrender_module_supports_main() -> bool:
    try:
        return importlib.util.find_spec("xyzrender.__main__") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _resolve_xyzrender_executable() -> str | None:
    python_dir = Path(sys.executable).resolve().parent
    for name in (
        "xyzrender",
    ):
        candidate = python_dir / name
        if candidate.exists():
            return str(candidate)

    resolved = shutil.which("xyzrender")
    if resolved:
        return resolved
    return None


def _normalize_xyzrender_executable_command(command: list[str]) -> list[str]:
    if not command:
        return command
    launcher_raw = str(command[0] or "").strip()
    # Respect explicitly configured paths (for example C:\\...\\xyzrender.exe).
    if any(sep in launcher_raw for sep in ("/", "\\")):
        return command
    launcher = launcher_raw.lower()
    if launcher not in {
        "xyzrender",
    }:
        return command

    resolved = _resolve_xyzrender_executable()
    if not resolved:
        return command
    return [resolved, *command[1:]]


def _normalize_python_xyzrender_command(command: list[str]) -> tuple[list[str], str]:
    if not command or not _is_bare_python_launcher(command[0]):
        return command, ""
    try:
        module_index = command.index("-m")
    except ValueError:
        return command, ""
    if module_index + 1 >= len(command):
        return command, ""
    module_name = str(command[module_index + 1] or "").strip().lower()
    if module_name != "xyzrender":
        return command, ""

    # Keep interpreter flags while forcing the current interpreter to avoid
    # PATH ambiguities between system Python and the active environment.
    normalized = [sys.executable, *command[1:]]
    if _xyzrender_module_supports_main():
        return normalized, ""

    executable = _resolve_xyzrender_executable()
    trailing_args = command[module_index + 2 :]
    if executable:
        return [executable, *trailing_args], (
            "Configured xyzrender module invocation is not runnable in this environment; "
            "falling back to xyzrender executable."
        )
    return normalized, ""


def _fallback_xyzrender_command() -> list[str]:
    executable = _resolve_xyzrender_executable()
    if executable:
        return [executable]
    return [sys.executable, "-m", "xyzrender"]


def _clean_path_like_value(value: str) -> str:
    entries = [entry for entry in str(value or "").split(os.pathsep) if entry]
    cleaned = [entry for entry in entries if "avogadro" not in entry.lower()]
    return os.pathsep.join(cleaned)


def build_xyzrender_subprocess_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)

    path_key = next((key for key in env if key.upper() == "PATH"), None)
    if path_key is not None:
        env[path_key] = _clean_path_like_value(env.get(path_key, ""))

    for key in list(env):
        key_upper = key.upper()
        if key_upper in {"CAIROCFFI_DLL_DIRECTORIES", "CAIRO_DLL_DIRECTORIES"}:
            cleaned = _clean_path_like_value(env.get(key, ""))
            if cleaned:
                env[key] = cleaned
            else:
                env.pop(key, None)

    return env


def is_xyzrender_cairo_error(detail_text: str) -> bool:
    text = str(detail_text or "").lower()
    if not text:
        return False
    cairo_markers = (
        "libcairo",
        "cairo-2.dll",
        "libcairo.so.2",
        "libcairo.2.dylib",
    )
    return any(marker in text for marker in cairo_markers)


def cairo_runtime_install_hint() -> str:
    platform_value = sys.platform.lower()
    if platform_value.startswith("win"):
        return (
            "Install GTK (Cairo) and run "
            "`reaction_hub/scripts/configure_venv_gtk.ps1` for this virtual environment, "
            "then start a new shell and reactivate `.venv`. "
            "Installer: "
            "https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases/download/2021-01-30/gtk2-runtime-2.24.33-2021-01-30-ts-win64.exe"
        )
    if platform_value.startswith("linux"):
        return "Install Cairo runtime (Ubuntu/WSL: `sudo apt-get install libcairo2`)."
    if platform_value.startswith("darwin"):
        return "Install Cairo runtime (Homebrew: `brew install cairo`)."
    return "Install the Cairo runtime for this environment."


def resolve_xyzrender_command(raw_command: str | None) -> tuple[list[str], str]:
    fallback = _fallback_xyzrender_command()
    command_text = str(raw_command or "").strip()
    if not command_text:
        return fallback, ""

    try:
        configured = shlex.split(command_text)
    except ValueError as exc:
        raise ValueError(f"Invalid REACTION_HUB_XYZRENDER_COMMAND: {exc}") from exc

    if not configured:
        return fallback, ""

    configured, normalization_warning = _normalize_python_xyzrender_command(configured)
    configured = _normalize_xyzrender_executable_command(configured)
    warnings: list[str] = []
    if normalization_warning:
        warnings.append(normalization_warning)

    return configured, " ".join(warnings)
