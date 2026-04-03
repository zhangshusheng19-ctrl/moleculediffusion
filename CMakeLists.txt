cmake_minimum_required(VERSION 3.18)
project(MolDiffusion LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenBabel 3.0 QUIET)

# CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 CACHE STRING "CUDA architectures")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

if(OpenBabel_FOUND)
    include_directories(${OPENBABEL3_INCLUDE_DIRS})
    add_definitions(-DUSE_OPENBABEL)
endif()

# Source files
set(CPU_SOURCES
    src/mol_diffusion_trainer.cpp
    src/molecule_builder.cpp
    src/main.cpp
)

set(CUDA_SOURCES
    src/mol_diffusion_model.cu
)

# Create executable
add_executable(mol_diffusion
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# Link libraries
target_link_libraries(mol_diffusion
    ${CUDA_LIBRARIES}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY}
)

if(OpenBabel_FOUND)
    target_link_libraries(mol_diffusion ${OPENBABEL3_LIBRARIES})
endif()

# Install target
install(TARGETS mol_diffusion DESTINATION bin)
