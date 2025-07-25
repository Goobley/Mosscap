#!/bin/bash

HDF_VERSION="1_14_3"
rm -rf "hdf5-${HDF_VERSION}*"
rm -rf hdfsrc

wget "https://github.com/HDFGroup/hdf5/releases/download/hdf5-${HDF_VERSION}/hdf5-${HDF_VERSION}.tar.gz"
tar xvzf "hdf5-${HDF_VERSION}.tar.gz"
cd hdfsrc
./configure CC=$(which mpicc) CFLAGS="-fPIC" FFLAGS="-fPIC" --enable-parallel --with-default-api-version=v110 --prefix="/usr/local/hdf5-parth"
make -j 10
make install

./configure CC=$(which mpicc) CFLAGS="-fPIC" FFLAGS="-fPIC" --enable-parallel --with-default-api-version=v110 --disable-shared --enable-fortran --prefix="/usr/local/hdf5"
make -j 10
make install