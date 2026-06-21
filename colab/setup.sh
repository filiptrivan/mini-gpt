#!/usr/bin/env bash
#
# colab/setup.sh — install everything Mini GPT needs to build, test, and train
# on Google Colab (Task 12).
#
# Colab runs as the root user, so apt-get needs no sudo. The CUDA toolkit (nvcc)
# already ships with Colab's GPU runtime, so we only add: the build tools, the
# CMocka unit-test framework, OpenMPI (for the distributed build/tests), and
# PyPDF2 (only needed for the OPTIONAL "extract your own PDF" path — the cleaned
# Serbian corpus is already committed at data/processed/skripta_clean.txt).
#
# Usage (from the repo root, inside Colab):
#     !bash colab/setup.sh
#
# `set -e` aborts on the first failing command so a half-installed environment
# doesn't silently produce confusing build errors later.
set -e

echo ">>> Installing apt packages (cmake, compiler, CMocka, OpenMPI)..."
apt-get -qq update
apt-get -qq install -y \
    cmake build-essential pkg-config \
    libcmocka-dev \
    libopenmpi-dev openmpi-bin

echo ">>> Installing Python packages (PyPDF2, for optional PDF extraction)..."
pip install -q PyPDF2

echo
echo ">>> Toolchain versions:"
cmake --version | head -1
printf 'mpirun: '; mpirun --version 2>/dev/null | head -1 || echo "MISSING"
# nvcc lives in /usr/local/cuda on Colab's GPU runtime; grep just the version
# line. If it is missing, the runtime is CPU-only — the build still works but
# falls back to CPU training (and the CUDA tests are skipped).
printf 'nvcc:   '; nvcc --version 2>/dev/null | grep -i release \
    || echo "MISSING — pick Runtime > Change runtime type > T4 GPU for the GPU path"
echo
echo ">>> Setup complete."
