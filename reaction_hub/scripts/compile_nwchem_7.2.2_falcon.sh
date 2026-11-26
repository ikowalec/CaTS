#!/usr/bin/env bash
set -euo pipefail

module purge
module load intel
module load CMake/3.29.3-GCCcore-13.3.0

version="7.2.2"
tag="v${version}-release"
app_root="${APP_ROOT:-/shared/home2/app_shared/PROJECT}"
build_root="${BUILD_ROOT:-${app_root}/build/nwchem-${version}}"
prefix="${PREFIX:-${app_root}/nwchem-${version}}"
jobs="${JOBS:-$(nproc)}"
url="${NWCHEM_URL:-https://github.com/nwchemgit/nwchem/archive/refs/tags/${tag}.tar.gz}"
tarball="${build_root}/nwchem-${version}.tar.gz"

if [[ -z "${MKLROOT:-}" ]]; then
    echo "MKLROOT is not set after loading intel" >&2
    exit 1
fi

mkdir -p "${build_root}" "${prefix}/bin"
cd "${build_root}"

if [[ ! -s "${tarball}" ]]; then
    echo "Downloading NWChem ${version} from ${url}"
    if command -v curl >/dev/null 2>&1; then
        curl -L "${url}" -o "${tarball}"
    else
        wget -O "${tarball}" "${url}"
    fi
fi

rm -rf "nwchem-${tag}" "nwchem-${version}-release"
tar -xzf "${tarball}"
src_dir="${build_root}/nwchem-${version}-release"
if [[ ! -d "${src_dir}" ]]; then
    src_dir="${build_root}/nwchem-${tag}"
fi

export NWCHEM_TOP="${src_dir}"
export NWCHEM_TARGET=LINUX64
export ARMCI_NETWORK=MPI-PR
export USE_MPI=y
export USE_MPIF=y
export USE_MPIF4=y
export USE_SCALAPACK=y
export USE_64TO32=y
export USE_NOFSCHECK=TRUE
export BLAS_SIZE=4
export SCALAPACK_SIZE=4
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"

export FC=mpiifx
export CC=mpiicx
export CXX=mpiicpx
export FOPTIMIZE="-O3 -fp-model=precise -march=core-avx2 -mauto-arch=CORE-AVX2 -fma"
export COPTIMIZE="-O3 -fp-model=precise -std=gnu99 -march=core-avx2 -mauto-arch=CORE-AVX2"

mkl_lib="${MKL_LIB_DIR:-${MKLROOT}/lib/intel64}"
[[ -d "${mkl_lib}" ]] || mkl_lib="${MKLROOT}"
export BLASOPT="-L${mkl_lib} -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -qopenmp"
export LAPACK_LIB="${BLASOPT}"
export SCALAPACK="-L${mkl_lib} -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 ${BLASOPT}"

cd "${NWCHEM_TOP}/src"
make realclean || true
make nwchem_config NWCHEM_MODULES="${NWCHEM_MODULES:-all}"
make -j "${jobs}"

cp "${NWCHEM_TOP}/bin/${NWCHEM_TARGET}/nwchem" "${prefix}/bin/nwchem"
echo "NWChem ${version} installed at ${prefix}/bin/nwchem"
