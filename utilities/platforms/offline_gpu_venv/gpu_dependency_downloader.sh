#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -n "${PYTHON_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
    module load "$PYTHON_MODULE"
fi

trap 'echo "ERROR: downloader failed at line $LINENO" >&2' ERR

# Offline wheelhouse downloader for a CPython 3.11 AMD ROCm environment.
# Run this script on an internet-connected login node.
#
# Optional environment overrides:
#   WORK_ROOT=/path/to/workdir   (default: current directory)
#   PYTHON_MODULE=python/3.11    (optional environment module to load)

WORK_ROOT="${WORK_ROOT:-$PWD}"
WHEELHOUSE="$WORK_ROOT/wheelhouse"
DOWNLOADER_VENV="$WORK_ROOT/.venvs/pip311-downloader"
PIP_TMP="$WORK_ROOT/pip-download-tmp"

PYPI_INDEX="https://pypi.org/simple"
ROCM_INDEX="https://download.pytorch.org/whl/rocm6.3"

TORCH_VERSION="2.7.1+rocm6.3"
TORCHVISION_VERSION="0.22.1+rocm6.3"
TORCHAUDIO_VERSION="2.7.1+rocm6.3"
MACE_VERSION="0.3.16"
TORCH_DFTD_VERSION="0.5.3"
PYTHON_HOSTLIST_VERSION="2.3.0"
ASE_GA_VERSION="1.0.3"
ACAT_VERSION="2.0.3"
ASAP3_VERSION="3.13.11"
SCIKIT_LEARN_VERSION="1.9.0"
DSCRIBE_VERSION="2.1.2"
TYPING_EXTENSIONS_VERSION="4.15.0"

# Locate the login-node Python used only to run the downloader.
if command -v python3.10 >/dev/null 2>&1; then
    SYSTEM_PYTHON="$(command -v python3.10)"
elif command -v python3 >/dev/null 2>&1; then
    SYSTEM_PYTHON="$(command -v python3)"
else
    echo "ERROR: python3.10 or python3 was not found." >&2
    exit 1
fi

echo "Login-node Python:"
echo "  $SYSTEM_PYTHON"
"$SYSTEM_PYTHON" --version

# Start clean. Remove these three rm commands if retaining old downloads is desired.
echo
echo "Removing previous downloader files..."
rm -rf "$DOWNLOADER_VENV" "$WHEELHOUSE" "$PIP_TMP"
mkdir -p "$(dirname "$DOWNLOADER_VENV")" "$WHEELHOUSE" "$PIP_TMP"

# Disable pip caching, including for the multi-gigabyte ROCm torch wheel.
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TMPDIR="$PIP_TMP"

echo
echo "Wheelhouse: $WHEELHOUSE"
echo "Temporary directory: $PIP_TMP"
df -h "$WORK_ROOT" || true

# Create a modern pip environment on the login node.
echo
echo "============================================================"
echo "Creating downloader environment"
echo "============================================================"
"$SYSTEM_PYTHON" -m venv "$DOWNLOADER_VENV"
DOWNLOADER_PYTHON="$DOWNLOADER_VENV/bin/python"

"$DOWNLOADER_PYTHON" -m pip install \
    --no-cache-dir \
    --upgrade \
    pip setuptools wheel packaging build

"$DOWNLOADER_PYTHON" --version
"$DOWNLOADER_PYTHON" -m pip --version

# Target the GPU-node interpreter: CPython 3.11 on Linux x86-64.
TARGET_OPTIONS=(
    --no-cache-dir
    --dest "$WHEELHOUSE"
    --only-binary=:all:
    --python-version 311
    --implementation cp
    --abi cp311
    --abi abi3
    --abi none
    --platform manylinux_2_28_x86_64
    --platform manylinux_2_17_x86_64
    --platform manylinux2014_x86_64
    --platform linux_x86_64
    --platform any
)

# Packaging and offline build tools.
echo
echo "============================================================"
echo "Downloading packaging and build tools"
echo "============================================================"
"$DOWNLOADER_PYTHON" -m pip download \
    "${TARGET_OPTIONS[@]}" \
    --index-url "$PYPI_INDEX" \
    pip setuptools wheel packaging build pyproject-hooks \
    "typing-extensions==$TYPING_EXTENSIONS_VERSION"

# python-hostlist is source-only but pure Python, so build a universal wheel now.
echo
echo "============================================================"
echo "Building python-hostlist universal wheel"
echo "============================================================"
"$DOWNLOADER_PYTHON" -m pip wheel \
    --no-cache-dir \
    --no-build-isolation \
    --wheel-dir "$WHEELHOUSE" \
    --no-deps \
    --index-url "$PYPI_INDEX" \
    "python-hostlist==$PYTHON_HOSTLIST_VERSION"

# Keep later resolution tied to ROCm PyTorch rather than CPU/CUDA PyTorch.
CONSTRAINTS="$PIP_TMP/constraints.txt"
cat > "$CONSTRAINTS" <<CONSTRAINTS_EOF
torch==$TORCH_VERSION
torchvision==$TORCHVISION_VERSION
torchaudio==$TORCHAUDIO_VERSION
typing-extensions==$TYPING_EXTENSIONS_VERSION
python-hostlist==$PYTHON_HOSTLIST_VERSION
CONSTRAINTS_EOF

# ROCm PyTorch packages.
echo
echo "============================================================"
echo "Downloading ROCm PyTorch packages"
echo "============================================================"
"$DOWNLOADER_PYTHON" -m pip download \
    "${TARGET_OPTIONS[@]}" \
    --index-url "$ROCM_INDEX" \
    --extra-index-url "$PYPI_INDEX" \
    --find-links "$WHEELHOUSE" \
    --constraint "$CONSTRAINTS" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION"

# MACE, torch-dftd and their ordinary dependencies. torch-dftd pulls in
# pymatgen. Do not add cupy-cuda12x for ROCm targets.
echo
echo "============================================================"
echo "Downloading MACE, torch-dftd and scientific dependencies"
echo "============================================================"
"$DOWNLOADER_PYTHON" -m pip download \
    "${TARGET_OPTIONS[@]}" \
    --index-url "$PYPI_INDEX" \
    --extra-index-url "$ROCM_INDEX" \
    --find-links "$WHEELHOUSE" \
    --constraint "$CONSTRAINTS" \
    "mace-torch==$MACE_VERSION" \
    "torch-dftd==$TORCH_DFTD_VERSION" \
    ase \
    "ase-ga==$ASE_GA_VERSION" \
    pymatgen \
    pandas \
    "scikit-learn==$SCIKIT_LEARN_VERSION" \
    "dscribe==$DSCRIBE_VERSION" \
    numpy \
    scipy \
    matplotlib \
    networkx \
    h5py \
    pyyaml \
    tqdm \
    requests

# Download a PyPI source archive without asking pip to inspect/build it.
download_pypi_sdist() {
    local project="$1"
    local version="$2"

    "$DOWNLOADER_PYTHON" - "$project" "$version" "$WHEELHOUSE" <<'PY'
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

project, version, destination = sys.argv[1:4]
metadata_url = (
    "https://pypi.org/pypi/"
    f"{urllib.parse.quote(project, safe='')}/"
    f"{urllib.parse.quote(version, safe='')}/json"
)
request = urllib.request.Request(
    metadata_url,
    headers={"User-Agent": "offline-wheelhouse-downloader/1.0"},
)
with urllib.request.urlopen(request) as response:
    metadata = json.load(response)

source_files = [
    item for item in metadata.get("urls", [])
    if item.get("packagetype") == "sdist"
]
if not source_files:
    raise SystemExit(f"ERROR: no sdist found for {project}=={version}")

source = next(
    (item for item in source_files
     if item.get("filename", "").endswith(".tar.gz")),
    source_files[0],
)
filename = source["filename"]
expected_sha256 = source["digests"]["sha256"]
output_path = os.path.join(destination, filename)
temporary_path = output_path + ".part"

digest = hashlib.sha256()
request = urllib.request.Request(
    source["url"],
    headers={"User-Agent": "offline-wheelhouse-downloader/1.0"},
)
try:
    with urllib.request.urlopen(request) as response, open(temporary_path, "wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            digest.update(block)

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {filename}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    os.replace(temporary_path, output_path)
except Exception:
    if os.path.exists(temporary_path):
        os.remove(temporary_path)
    raise

print(f"Downloaded and verified: {output_path}")
PY
}

echo
echo "============================================================"
echo "Downloading ACAT source archive"
echo "============================================================"
download_pypi_sdist "acat" "$ACAT_VERSION"

echo
echo "============================================================"
echo "Downloading ASAP3 source archive"
echo "============================================================"
download_pypi_sdist "asap3" "$ASAP3_VERSION"

# Record the requested top-level packages for reference.
cat > "$WHEELHOUSE/offline-requirements.txt" <<REQUIREMENTS_EOF
torch==$TORCH_VERSION
torchvision==$TORCHVISION_VERSION
torchaudio==$TORCHAUDIO_VERSION
mace-torch==$MACE_VERSION
torch-dftd==$TORCH_DFTD_VERSION
ase
ase-ga==$ASE_GA_VERSION
pymatgen
acat==$ACAT_VERSION
asap3==$ASAP3_VERSION
pandas
scikit-learn==$SCIKIT_LEARN_VERSION
dscribe==$DSCRIBE_VERSION
REQUIREMENTS_EOF

# Verify essential downloads.
require_download() {
    local pattern="$1"
    local description="$2"
    if compgen -G "$WHEELHOUSE/$pattern" >/dev/null; then
        echo "OK: $description"
    else
        echo "ERROR: missing $description" >&2
        echo "Expected filename pattern: $pattern" >&2
        exit 1
    fi
}

echo
echo "============================================================"
echo "Verifying essential downloads"
echo "============================================================"
require_download "torch-${TORCH_VERSION}-cp311-cp311-*.whl" "ROCm torch"
require_download "torchvision-${TORCHVISION_VERSION}-cp311-cp311-*.whl" "ROCm torchvision"
require_download "torchaudio-${TORCHAUDIO_VERSION}-cp311-cp311-*.whl" "ROCm torchaudio"
require_download "mace_torch-${MACE_VERSION}-*.whl" "mace-torch"
require_download "torch_dftd-${TORCH_DFTD_VERSION}-py3-none-any.whl" "torch-dftd"
require_download "python_hostlist-${PYTHON_HOSTLIST_VERSION}-py3-none-any.whl" "python-hostlist"
require_download "acat-${ACAT_VERSION}.tar.gz" "ACAT source archive"
require_download "asap3-${ASAP3_VERSION}.tar.gz" "ASAP3 source archive"
require_download "scikit_learn-${SCIKIT_LEARN_VERSION}-cp311-cp311-*.whl" "scikit-learn"
require_download "dscribe-${DSCRIBE_VERSION}-cp311-cp311-*.whl" "DScribe"

echo
echo "============================================================"
echo "Download completed successfully"
echo "============================================================"
echo "Wheelhouse: $WHEELHOUSE"
echo "Files: $(find "$WHEELHOUSE" -maxdepth 1 -type f | wc -l)"
du -sh "$WHEELHOUSE"
