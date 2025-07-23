#!/bin/bash

GCC_INCLUDE_PATH=$(g++ --print-file-name=include)

./cmake_clean.sh

cmake  \
      -DMOSSCAP_ARCH="CUDA"              \
      -DYAKL_AUTO_PROFILE="On"         \
      -DMOSSCAP_CXX_FLAGS="-O2 -g -G" \
      -DGCC_INCLUDE_PATH="${GCC_INCLUDE_PATH}" \
      -DNETCDF_INCLUDE_PATH="$(nc-config --includedir)" \
      -DLDLIBS="$(nc-config --libs)" \
      -DKokkos_ROOT="$(pwd)/../kokkos-debug/" \
      -DCMAKE_PREFIX_PATH="$(pwd)/../kokkos-debug/lib/cmake" \
      -DCMAKE_BUILD_TYPE="Debug" \
      ..
