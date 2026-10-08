#!/usr/bin/env bash
# Source this file; do not execute it: source scripts/activate.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Use: source scripts/activate.sh" >&2
  exit 1
fi

SCOPE2H3_CONDA_BASE="${SCOPE2H3_CONDA_BASE:-/inspire/hdd/project/wuliqifa/public/miniconda3}"
SCOPE2H3_CONDA_ENV="${SCOPE2H3_CONDA_ENV:-h3}"
SCOPE2H3_CONDA_INIT="${SCOPE2H3_CONDA_BASE}/etc/profile.d/conda.sh"

if [[ ! -f "${SCOPE2H3_CONDA_INIT}" ]]; then
  echo "Conda initialization script not found: ${SCOPE2H3_CONDA_INIT}" >&2
  return 1
fi

# shellcheck disable=SC1090
source "${SCOPE2H3_CONDA_INIT}"
conda activate "${SCOPE2H3_CONDA_ENV}"
export DIFFSYNTH_SKIP_DOWNLOAD=True

echo "Activated Conda environment: ${SCOPE2H3_CONDA_ENV}"
echo "DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD}"
