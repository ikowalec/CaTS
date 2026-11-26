#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "ERROR: installation failed at line $LINENO" >&2' ERR

# Offline installation for an AMD ROCm environment.
#
# Run this script from a GPU allocation after loading the site's
# Python 3.11 and C++ compiler modules.
#
# Default virtual environment:
#   $WORK_ROOT/venvs/mace-rocm311 (WORK_ROOT defaults to $PWD)
#
# To use another environment path:
#   ./offline_gpu_node_venv.sh /path/to/venv
#
# To specify the base Python used when creating a new environment:
#   PYTHON311=/path/to/python3.11 ./offline_gpu_node_venv.sh

WORK_ROOT="${WORK_ROOT:-$PWD}"
WHEELHOUSE="$WORK_ROOT/wheelhouse"
VENV="${1:-$WORK_ROOT/venvs/mace-rocm311}"
INSTALL_TMP="$WORK_ROOT/pip-install-tmp"
BUILT_WHEELS="$WHEELHOUSE/built-cp311"

TORCH_VERSION="2.7.1+rocm6.3"
TORCHVISION_VERSION="0.22.1+rocm6.3"
TORCHAUDIO_VERSION="2.7.1+rocm6.3"
MACE_VERSION="0.3.16"
ACAT_VERSION="2.0.3"
ASAP3_VERSION="3.13.11"
SCIKIT_LEARN_VERSION="1.9.0"
DSCRIBE_VERSION="2.1.2"

# Prevent pip from accessing package indexes or using its cache.
export PIP_NO_INDEX=1
export PIP_FIND_LINKS="$WHEELHOUSE"
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TMPDIR="$INSTALL_TMP"

mkdir -p "$INSTALL_TMP" "$BUILT_WHEELS" "$(dirname "$VENV")"

if [[ ! -d "$WHEELHOUSE" ]]; then
    echo "ERROR: wheelhouse not found: $WHEELHOUSE" >&2
    exit 1
fi

require_file() {
    local pattern="$1"
    local description="$2"

    if ! compgen -G "$WHEELHOUSE/$pattern" >/dev/null; then
        echo "ERROR: missing $description in $WHEELHOUSE" >&2
        echo "Expected filename pattern: $pattern" >&2
        exit 1
    fi
}

require_file "torch-${TORCH_VERSION}-cp311-cp311-*.whl" "ROCm torch wheel"
require_file "torchvision-${TORCHVISION_VERSION}-cp311-cp311-*.whl" "ROCm torchvision wheel"
require_file "torchaudio-${TORCHAUDIO_VERSION}-cp311-cp311-*.whl" "ROCm torchaudio wheel"
require_file "mace_torch-${MACE_VERSION}-*.whl" "mace-torch wheel"
require_file "acat-${ACAT_VERSION}.tar.gz" "ACAT source archive"
require_file "asap3-${ASAP3_VERSION}.tar.gz" "ASAP3 source archive"
require_file "scikit_learn-${SCIKIT_LEARN_VERSION}-cp311-cp311-*.whl" "scikit-learn wheel"
require_file "dscribe-${DSCRIBE_VERSION}-cp311-cp311-*.whl" "DScribe wheel"

find_python311() {
    local candidate

    if [[ -n "${PYTHON311:-}" ]]; then
        candidate="$PYTHON311"
        if [[ ! -x "$candidate" ]]; then
            echo "ERROR: PYTHON311 is not executable: $candidate" >&2
            exit 1
        fi
        printf '%s\n' "$candidate"
        return
    fi

    for candidate in python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
                command -v "$candidate"
                return
            fi
        fi
    done

    echo "ERROR: Python 3.11 is not available in this shell." >&2
    echo "Load the Python 3.11.7 module, then rerun the script." >&2
    exit 1
}

if [[ -x "$VENV/bin/python" ]]; then
    echo "Reusing virtual environment: $VENV"
else
    BASE_PYTHON="$(find_python311)"
    echo "Creating virtual environment with: $BASE_PYTHON"
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV"
fi

PYTHON="$VENV/bin/python"

if ! "$PYTHON" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
    echo "ERROR: $VENV does not use Python 3.11." >&2
    "$PYTHON" --version >&2 || true
    exit 1
fi

if ! "$PYTHON" -c 'import mpi4py' >/dev/null 2>&1; then
    echo "ERROR: $VENV cannot import site mpi4py." >&2
    echo "Load the site Python/MPI module, then recreate this environment with --system-site-packages." >&2
    exit 1
fi

# Locate a C++ compiler for ASAP3. On Cray systems, CC is normally the
# C++ compiler wrapper. Respect an existing CXX setting when provided.
if [[ -z "${CXX:-}" ]]; then
    for candidate in CC g++ c++; do
        if command -v "$candidate" >/dev/null 2>&1; then
            export CXX="$(command -v "$candidate")"
            break
        fi
    done
fi

if [[ -z "${CXX:-}" ]] || ! command -v "$CXX" >/dev/null 2>&1; then
    echo "ERROR: no C++ compiler was found for building ASAP3." >&2
    echo "Load the appropriate compiler module and rerun the script." >&2
    exit 1
fi

echo
echo "Python environment:"
"$PYTHON" --version
"$PYTHON" -m pip --version

echo
echo "C++ compiler:"
"$CXX" --version | head -n 2 || true

echo
echo "Wheelhouse:"
echo "  $WHEELHOUSE"

echo
echo "============================================================"
echo "Installing packaging and build tools"
echo "============================================================"

"$PYTHON" -m pip install \
    --no-index \
    --find-links "$WHEELHOUSE" \
    --upgrade \
    pip \
    setuptools \
    wheel \
    packaging \
    build \
    pyproject-hooks

echo
echo "============================================================"
echo "Installing ROCm PyTorch"
echo "============================================================"

"$PYTHON" -m pip install \
    --no-index \
    --find-links "$WHEELHOUSE" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION"

echo
echo "============================================================"
echo "Installing NumPy and scientific dependencies"
echo "============================================================"

# NumPy must be installed before compiling ASAP3 so its C headers are
# available inside this Python 3.11 environment.
"$PYTHON" -m pip install \
    --no-index \
    --find-links "$WHEELHOUSE" \
    numpy \
    scipy \
    matplotlib \
    pandas \
    "scikit-learn==$SCIKIT_LEARN_VERSION" \
    "dscribe==$DSCRIBE_VERSION" \
    ase \
    ase-ga \
    networkx \
    h5py \
    pyyaml \
    tqdm \
    requests

echo
echo "============================================================"
echo "Installing MACE"
echo "============================================================"

"$PYTHON" -m pip install \
    --no-index \
    --find-links "$WHEELHOUSE" \
    "mace-torch==$MACE_VERSION"

echo "============================================================"
echo "Installing Torch-DFTD and its dependencies"
echo "============================================================"

"$PYTHON" -m pip install \
    --no-index \
    --find-links "$WHEELHOUSE" \
    pymatgen \
    "torch-dftd==0.5.3"

ACAT_SDIST="$(compgen -G "$WHEELHOUSE/acat-${ACAT_VERSION}.tar.gz" | head -n 1)"
ASAP3_SDIST="$(compgen -G "$WHEELHOUSE/asap3-${ASAP3_VERSION}.tar.gz" | head -n 1)"

# Build source packages using the actual Python 3.11 target environment.
# The resulting wheels are retained for quicker reinstalls.
echo
echo "============================================================"
echo "Building and installing ACAT"
echo "============================================================"

if ! compgen -G "$BUILT_WHEELS/acat-${ACAT_VERSION}-*.whl" >/dev/null; then
    "$PYTHON" -m pip wheel \
        --no-index \
        --find-links "$WHEELHOUSE" \
        --no-build-isolation \
        --no-deps \
        --wheel-dir "$BUILT_WHEELS" \
        "$ACAT_SDIST"
fi

"$PYTHON" -m pip install \
    --no-index \
    --find-links "$BUILT_WHEELS" \
    --no-deps \
    "acat==$ACAT_VERSION"

echo
echo "============================================================"
echo "Building and installing ASAP3"
echo "============================================================"

if ! compgen -G "$BUILT_WHEELS/asap3-${ASAP3_VERSION}-cp311-*.whl" >/dev/null; then
    "$PYTHON" -m pip wheel \
        --no-index \
        --find-links "$WHEELHOUSE" \
        --no-build-isolation \
        --no-deps \
        --wheel-dir "$BUILT_WHEELS" \
        "$ASAP3_SDIST"
fi

"$PYTHON" -m pip install \
    --no-index \
    --find-links "$BUILT_WHEELS" \
    --no-deps \
    "asap3==$ASAP3_VERSION"

echo
echo "============================================================"
echo "Checking dependency consistency"
echo "============================================================"

"$PYTHON" -m pip check

echo
echo "============================================================"
echo "Checking imports and ROCm GPU access"
echo "============================================================"

"$PYTHON" - <<'PY'
import importlib
import sys

modules = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "MACE": "mace",
    "ASE": "ase",
    "ACAT": "acat",
    "ASAP3": "asap3",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "DScribe": "dscribe",
}

for label, module_name in modules.items():
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "version not exposed")
    print(f"{label:12s}: OK — {version}")

import torch

print(f"PyTorch:      {torch.__version__}")
print(f"ROCm runtime: {torch.version.hip}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU visible:  {torch.cuda.is_available()}")

if torch.version.hip is None:
    raise SystemExit("ERROR: the installed PyTorch build is not a ROCm build.")

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: ROCm PyTorch is installed, but no GPU is visible. "
        "Run the script inside a GPU allocation."
    )

device_name = torch.cuda.get_device_name(0)
print(f"GPU:          {device_name}")

x = torch.rand((512, 512), device="cuda")
y = x @ x
print(f"GPU test:     OK ({y.device}, mean={y.mean().item():.6f})")
PY

echo
echo "============================================================"
echo "Installation completed successfully"
echo "============================================================"

echo
echo "Activate the environment with:"
echo "  source $VENV/bin/activate"

echo
echo "MACE executable:"
if [[ -x "$VENV/bin/mace_run_train" ]]; then
    echo "  $VENV/bin/mace_run_train"
else
    echo "  WARNING: mace_run_train was not found in $VENV/bin" >&2
fi
