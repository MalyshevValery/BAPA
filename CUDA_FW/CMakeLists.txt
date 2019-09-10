cmake_minimum_required(VERSION 3.14)

# Compilers for CUDA
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER /usr/bin/gcc-6)
set(CMAKE_CXX_COMPILER /usr/bin/g++-6)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -pthread)

#Propjects configuration
project(CUDA_FW LANGUAGES CUDA CXX)
project(FW LANGUAGES CXX)
project(GraphGen LANGUAGES CXX)
enable_language(CUDA)

#Check CUDA
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


#Executables for projects
add_executable(CUDA_FW cuda_fw.cu headers/dev_array.h headers/block_2d.h kernels/block_2d.cu)
add_executable(FW floyd_warshall.cpp)
add_executable(GraphGen graphgen.cpp graphgen.cpp)
