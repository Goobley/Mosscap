cmake_minimum_required(VERSION 3.18)
if ("${MOSSCAP_ARCH}" STREQUAL "CUDA")
    project(MOSSCAP LANGUAGES CXX CUDA)
else ()
    project(MOSSCAP LANGUAGES CXX)
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

list(APPEND MOSSCAP_SOURCE_FILES
    "source/main.cpp"
)

add_executable(mosscap
    "${MOSSCAP_SOURCE_FILES}"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if ("${MOSSCAP_ARCH}" STREQUAL "CUDA")
    set(CMAKE_CXX_STANDARD 20)
endif()

if ("${MOSSCAP_SINGLE_PRECISION}")
    target_compile_definitions(mosscap PUBLIC MOSSCAP_SINGLE_PRECISION)
    set(MOSSCAP_CXX_FLAGS "${MOSSCAP_CXX_FLAGS} -DMOSSCAP_SINGLE_PRECISION")
endif()

# NOTE(cmo): Override language type for cuda, may need to do this for HIP too.
set_source_files_properties(${MOSSCAP_SOURCE_FILES} PROPERTIES CXX_STANDARD 20)
set_source_files_properties(${MOSSCAP_SOURCE_FILES} PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
if ("${MOSSCAP_ARCH}" STREQUAL "CUDA")
    set(MOSSCAP_EXTRA_CUDA_STD_FLAG "")
    if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
        set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
        set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
        set(MOSSCAP_EXTRA_CUDA_STD_FLAG "-std=c++20")
    endif()
    set_target_properties(mosscap PROPERTIES CUDA_ARCHITECTURES ${Kokkos_CUDA_ARCHITECTURES})

    set_source_files_properties(${MOSSCAP_SOURCE_FILES} PROPERTIES CUDA_STANDARD 20)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} ${MOSSCAP_EXTRA_CUDA_STD_FLAG}")
endif()

############## mosscap ##############
target_include_directories(mosscap SYSTEM PUBLIC
    "${GCC_INCLUDE_PATH}"
    "${MPI_INCLUDE_PATH}"
    "${NETCDF_INCLUDE_PATH}"
    "${CUDA_MATH_INCLUDE_PATH}"
)
target_link_libraries(mosscap "${LDLIBS}")
target_link_libraries(mosscap fmt::fmt)
target_link_libraries(mosscap argparse)
target_link_libraries(mosscap git_version)
target_link_libraries(mosscap Kokkos::kokkos)
target_link_libraries(mosscap yakl)

set_target_properties(mosscap PROPERTIES
    COMPILE_FLAGS "${YAKL_DEFS} ${MOSSCAP_CXX_FLAGS}"
    LINK_FLAGS "${LDFLAGS}"
)

message("Compiling with: ${CMAKE_CXX_FLAGS} ${YAKL_DEFS} ${MOSSCAP_CXX_FLAGS}")