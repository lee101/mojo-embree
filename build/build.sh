#!/usr/bin/env bash
set -euo pipefail

mkdir -p dist
mojo build --emit shared-lib -I src src/embree.mojo -o dist/libmojo-embree.so
"${CXX:-c++}" -O3 -fPIC -shared -I "${CONDA_PREFIX}/include" bench/embree_reference.cpp \
    -L "${CONDA_PREFIX}/lib" -lembree4 -o dist/libembree-reference.so
