#!/bin/bash
rm -r kokkos-4.5.01

./docker_setup.sh \
    "-DKokkos_ENABLE_SERIAL=ON \
     -DKokkos_ENABLE_OPENMP=ON \
     -DCMAKE_CXX_COMPILER=g++ \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=$(pwd)/../../kokkos"