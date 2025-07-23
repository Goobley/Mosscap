cmake_minimum_required(VERSION 3.18)
if ("${HYDRO_ARCH}" STREQUAL "CUDA")
    project(Hydro LANGUAGES CXX CUDA)
else ()
    project(Hydro LANGUAGES CXX)
endif()

add_subdirectory(fmt)
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

include("CheckGit.cmake")
CheckGitSetup()

SET(CMAKE_CXX_FLAGS "${CXX_FLAGS}")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Kokkos REQUIRED GLOBAL)

add_library(yakl INTERFACE)
target_include_directories(yakl INTERFACE YAKL/src YAKL/external YAKL/src/extensions)
set(YAKL_DEFS "")
if (${YAKL_AUTO_PROFILE})
    set(YAKL_DEFS "${YAKL_DEFS} -DYAKL_AUTO_PROFILE")
endif()
if (${YAKL_PROFILE})
    set(YAKL_DEFS "${YAKL_DEFS} -DYAKL_PROFILE")
endif()
if (${YAKL_AUTO_FENCE})
    set(YAKL_DEFS "${YAKL_DEFS} -DYAKL_AUTO_FENCE")
endif()

list(APPEND HYDRO_SOURCE_FILES
    "source/main.cpp"
)

add_executable(hydro
    "${HYDRO_SOURCE_FILES}"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if ("${HYDRO_ARCH}" STREQUAL "CUDA")
    set(CMAKE_CXX_STANDARD 20)
endif()

if ("${HYDRO_SINGLE_PRECISION}")
    target_compile_definitions(scatter PUBLIC HYDRO_SINGLE_PRECISION)
    set(HYDRO_CXX_FLAGS "${HYDRO_CXX_FLAGS} -DHYDRO_SINGLE_PRECISION")
endif()

# NOTE(cmo): Override language type for cuda, may need to do this for HIP too.
set_source_files_properties(${HYDRO_SOURCE_FILES} PROPERTIES CXX_STANDARD 20)
set_source_files_properties(${HYDRO_SOURCE_FILES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
if ("${HYDRO_ARCH}" STREQUAL "CUDA")
    set(HYDRO_EXTRA_CUDA_STD_FLAG "")
    if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
        set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
        set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
        set(HYDRO_EXTRA_CUDA_STD_FLAG "-std=c++20")
    endif()
    set_target_properties(hydro PROPERTIES CUDA_ARCHITECTURES ${Kokkos_CUDA_ARCHITECTURES})

    set_source_files_properties(${HYDRO_SOURCE_FILES} PROPERTIES CUDA_STANDARD 20)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} ${HYDRO_EXTRA_CUDA_STD_FLAG}")
endif()

############## hydro ##############
target_include_directories(hydro SYSTEM PUBLIC
    "${GCC_INCLUDE_PATH}"
    "${MPI_INCLUDE_PATH}"
    "${NETCDF_INCLUDE_PATH}"
    "${CUDA_MATH_INCLUDE_PATH}"
)
target_link_libraries(hydro "${LDLIBS}")
target_link_libraries(hydro fmt::fmt)
target_link_libraries(hydro argparse)
target_link_libraries(hydro git_version)
target_link_libraries(hydro Kokkos::kokkos)
target_link_libraries(hydro yakl)

set_target_properties(hydro PROPERTIES
    COMPILE_FLAGS "${YAKL_DEFS} ${HYDRO_CXX_FLAGS}"
    LINK_FLAGS "${LDFLAGS}"
)

message("Compiling with: ${CMAKE_CXX_FLAGS} ${YAKL_DEFS} ${HYDRO_CXX_FLAGS}")