from __future__ import annotations

import sys

import pytest

import core.xyzrender as xyzrender_module


def test_parse_canvas_size_accepts_square_and_rectangular_values() -> None:
    assert xyzrender_module.parse_canvas_size("900") == (900, 900)
    assert xyzrender_module.parse_canvas_size("900x700") == (900, 700)
    assert xyzrender_module.parse_canvas_size("900,700") == (900, 700)
    assert xyzrender_module.parse_canvas_size("900:700") == (900, 700)


def test_parse_canvas_size_rejects_invalid_values() -> None:
    assert xyzrender_module.parse_canvas_size("") is None
    assert xyzrender_module.parse_canvas_size("0x900") is None
    assert xyzrender_module.parse_canvas_size("900x") is None
    assert xyzrender_module.parse_canvas_size("wide") is None


def test_normalize_gif_canvas_rewrites_full_size_disposal_frames(tmp_path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    sequence_module = pytest.importorskip("PIL.ImageSequence")

    gif_path = tmp_path / "scan.gif"
    frame_a = image_module.new("RGB", (20, 10), "white")
    frame_b = image_module.new("RGB", (20, 10), "black")
    frame_a.save(
        gif_path,
        save_all=True,
        append_images=[frame_b],
        duration=[80, 120],
        loop=0,
        optimize=False,
    )

    normalized, note = xyzrender_module.normalize_gif_canvas(gif_path, size=(40, 30))

    assert normalized is True
    assert "40x30" in note
    with image_module.open(gif_path) as image:
        frames = [frame.copy() for frame in sequence_module.Iterator(image)]
        image.seek(0)
        assert getattr(image, "disposal_method", None) == 2
        image.seek(1)
        assert getattr(image, "disposal_method", None) == 2
    assert [frame.size for frame in frames] == [(40, 30), (40, 30)]


def test_resolve_xyzrender_command_uses_current_interpreter_for_python_module(monkeypatch) -> None:
    monkeypatch.setattr(xyzrender_module, "_xyzrender_module_supports_main", lambda: True)

    command, warning = xyzrender_module.resolve_xyzrender_command("python -m xyzrender")
    assert command == [sys.executable, "-m", "xyzrender"]
    assert warning == ""


def test_resolve_xyzrender_command_falls_back_to_executable_when_module_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(xyzrender_module, "_xyzrender_module_supports_main", lambda: False)
    monkeypatch.setattr(
        xyzrender_module,
        "_resolve_xyzrender_executable",
        lambda: "/tmp/xyzrender.exe",
    )

    command, warning = xyzrender_module.resolve_xyzrender_command("python -I -m xyzrender --foo")
    assert command == ["/tmp/xyzrender.exe", "--foo"]
    assert "falling back to xyzrender executable" in warning.lower()


def test_resolve_xyzrender_command_normalizes_xyzrender_launcher_to_absolute(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        xyzrender_module,
        "_resolve_xyzrender_executable",
        lambda: "/tmp/xyzrender.exe",
    )

    command, warning = xyzrender_module.resolve_xyzrender_command("xyzrender input.xyz")
    assert command == ["/tmp/xyzrender.exe", "input.xyz"]
    assert warning == ""


def test_resolve_xyzrender_command_keeps_explicit_xyzrender_path(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        xyzrender_module,
        "_resolve_xyzrender_executable",
        lambda: "/tmp/xyzrender.exe",
    )

    command, warning = xyzrender_module.resolve_xyzrender_command(
        "C:/Avogadro/bin/xyzrender input.xyz"
    )
    assert command == ["C:/Avogadro/bin/xyzrender", "input.xyz"]
    assert warning == ""


def test_build_xyzrender_subprocess_env_removes_avogadro_paths() -> None:
    env = {
        "PATH": r"C:\ToolA;C:\AvogadroOrca4.2.1.current\Avogadro\bin;C:\ToolB",
        "CAIROCFFI_DLL_DIRECTORIES": r"C:\Avogadro\bin;C:\Other",
    }

    sanitized = xyzrender_module.build_xyzrender_subprocess_env(env)

    assert "avogadro" not in sanitized["PATH"].lower()
    assert "ToolA" in sanitized["PATH"]
    assert "ToolB" in sanitized["PATH"]
    assert "avogadro" not in sanitized["CAIROCFFI_DLL_DIRECTORIES"].lower()
    assert "Other" in sanitized["CAIROCFFI_DLL_DIRECTORIES"]


def test_cairo_runtime_install_hint_windows(monkeypatch) -> None:
    monkeypatch.setattr(xyzrender_module.sys, "platform", "win32")
    hint = xyzrender_module.cairo_runtime_install_hint()
    assert "configure_venv_gtk.ps1" in hint
    assert "GTK" in hint


def test_cairo_runtime_install_hint_linux(monkeypatch) -> None:
    monkeypatch.setattr(xyzrender_module.sys, "platform", "linux")
    hint = xyzrender_module.cairo_runtime_install_hint()
    assert "libcairo2" in hint


def test_cairo_runtime_install_hint_macos(monkeypatch) -> None:
    monkeypatch.setattr(xyzrender_module.sys, "platform", "darwin")
    hint = xyzrender_module.cairo_runtime_install_hint()
    assert "brew install cairo" in hint
