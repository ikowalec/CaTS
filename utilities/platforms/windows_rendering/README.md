# Windows SVG/GIF rendering setup

The local `xyzrender` viewer needs a native Cairo runtime in addition to its
Python packages. This utility makes an existing GTK/Cairo installation visible
when activating the repository's Windows venv. It does not install GTK, alter
system-wide paths or configure the separate POV-Ray report renderer.

Users are assumed to be able to install and link native software and select
paths appropriate to their machine. Match runtime architecture to Python.
The existing helper supports the GTK runtime layout with `bin` and `lib`
subdirectories. See the
[GTK runtime project](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer)
for installation information; point the helper at the runtime actually installed
rather than assuming its built-in path matches your machine.

## Configure once

From the repository root, run the existing PowerShell helper with your venv and
GTK installation paths:

```powershell
& .\utilities\platforms\windows_rendering\configure_venv_gtk.ps1 -VenvPath "<repository-root>\.venv" -GtkRoot "<installed-gtk-root>"
```

This is a PowerShell setup helper, not a Python command. Use the machine's
existing script-execution policy. It patches `Scripts/Activate.ps1` and
`Scripts/activate.bat` with marked blocks; rerunning replaces its own blocks.
Only run it for the intended environment. It does not change the currently open
shell, so activate that environment in a new terminal afterward.

For Windows Command Prompt, from the repository root:

```bat
call .venv\Scripts\activate.bat
.venv\Scripts\python.exe -c "import ctypes; ctypes.CDLL('libcairo-2.dll'); print('cairo-ok')"
.venv\Scripts\xyzrender.exe --help
```

The first check verifies DLL loading; the second verifies the renderer's command
entry point. Complete a real SVG/GIF render in the dashboard to test the full
rendering path. Start Django from the activated terminal so it inherits native
runtime linking. Processes started through a service or IDE need equivalent
environment propagation.

## Linking details

| Process variable | Purpose |
| --- | --- |
| `GTK_RUNTIME_ROOT` | Selected native runtime root. |
| `PATH` | Makes runtime library directories discoverable by the process. |
| `CAIROCFFI_DLL_DIRECTORIES` | Makes those directories available to Cairo's Python binding. |

No configuration values are listed here. Inspect the helper and pass your local
paths explicitly. Calling the venv interpreter directly does not activate these
shell changes by itself.

If a DLL is still missing, check runtime architecture, the actual library files,
and the environment inherited by Django. Recreating the venv replaces activation
scripts, so rerun this setup afterward. Revert by recreating the venv or removing
only the helper's marked blocks from its activation scripts.
