"""Build script for packaging MACE Trajectory Screener into a single-file executable."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GUI_TOOLS_DIR = REPO_ROOT / "gui_tools"
ENTRY_POINT = GUI_TOOLS_DIR / "mace_trajectory_screener_gui.py"
ICON_ICO = GUI_TOOLS_DIR / "icon.ico"
ICON_PNG = GUI_TOOLS_DIR / "icon.png"
DIST_DIR = GUI_TOOLS_DIR / "dist"
BUILD_DIR = REPO_ROOT / "scratch" / "pyinstaller_build"
SPEC_FILE = GUI_TOOLS_DIR / "MACE_Trajectory_Screener.spec"


def main() -> int:
    print(f"=== Packaging MACE Trajectory Screener ===")
    print(f"Entry point: {ENTRY_POINT}")
    print(f"Icon: {ICON_ICO}")

    if not ENTRY_POINT.is_file():
        print(f"Error: Entry point {ENTRY_POINT} not found.")
        return 1

    if not ICON_ICO.is_file():
        print(f"Error: Icon {ICON_ICO} not found.")
        return 1

    # Clean previous build artifacts
    for path in (DIST_DIR, BUILD_DIR, SPEC_FILE):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink(missing_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # PyInstaller arguments
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        f"--name=MACE_Trajectory_Screener",
        f"--icon={ICON_ICO}",
        f"--add-data={ICON_ICO};.",
        f"--add-data={ICON_PNG};.",
        f"--paths={GUI_TOOLS_DIR}",
        f"--paths={REPO_ROOT}",
        "--hidden-import=screener_core",
        "--hidden-import=ase",
        "--hidden-import=ase.io",
        "--hidden-import=ase.io.extxyz",
        "--hidden-import=ase.io.formats",
        "--hidden-import=ase.io.trajectory",
        "--hidden-import=ase.db",
        "--hidden-import=sqlite3",
        "--hidden-import=ase.calculators.singlepoint",
        "--hidden-import=similarity",
        "--hidden-import=similarity.rematch",
        "--hidden-import=similarity.rematch.memory_rematch",
        "--hidden-import=similarity.rematch.get_rematch",
        "--hidden-import=similarity.rematch.prescreen",
        "--hidden-import=similarity.run.evaluate",
        "--hidden-import=dscribe",
        "--hidden-import=dscribe.descriptors",
        "--hidden-import=sklearn",
        "--hidden-import=sklearn.preprocessing",
        "--collect-submodules=ase",
        "--collect-data=ase",
        "--collect-submodules=dscribe",
        "--collect-data=dscribe",
        "--collect-submodules=sklearn",
        "--exclude-module=torch",
        "--exclude-module=mace",
        "--exclude-module=e3nn",
        "--exclude-module=skimage",
        f"--distpath={DIST_DIR}",
        f"--workpath={BUILD_DIR}",
        f"--specpath={GUI_TOOLS_DIR}",
        str(ENTRY_POINT),
    ]

    print("Running PyInstaller...")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        print(f"PyInstaller build failed with code {res.returncode}")
        return res.returncode

    built_exe = DIST_DIR / "MACE_Trajectory_Screener.exe"
    if not built_exe.is_file():
        print(f"Error: Expected output {built_exe} was not produced.")
        return 1

    # Copy to gui_tools directory with fresh name to guarantee Windows Explorer displays the icon
    final_exe = GUI_TOOLS_DIR / "MACE_Trajectory_Screener.exe"
    try:
        shutil.copy2(built_exe, final_exe)
    except Exception as e:
        print(f"Note copying to {final_exe}: {e}")

    # Also create a designated named copy to bypass any cached generic icon in Explorer
    final_renamed_exe = GUI_TOOLS_DIR / "MACETrajectoryScreener.exe"
    try:
        shutil.copy2(built_exe, final_renamed_exe)
    except Exception as e:
        print(f"Note copying to {final_renamed_exe}: {e}")

    size_mb = final_exe.stat().st_size / (1024 * 1024)
    print(f"\nBuild SUCCESS!")
    print(f"Single-file executable: {final_exe} ({size_mb:.1f} MB)")
    print(f"Icon-refreshed executable: {final_renamed_exe} ({size_mb:.1f} MB)")

    # Notify Windows shell to refresh icon cache if on Windows
    if sys.platform == "win32":
        try:
            import ctypes
            # SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, NULL, NULL)
            ctypes.windll.shell32.SHChangeNotify(0x08000000, 0x0000, None, None)
            print("Windows shell icon cache notification sent.")
        except Exception as e:
            print(f"Shell notification note: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
