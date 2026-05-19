cmake_minimum_required(VERSION 3.18)
project(sparse_ot_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_bonneel src/cpp/bonneel_solver.cpp)
target_include_directories(_bonneel PRIVATE src/cpp)
target_compile_options(_bonneel PRIVATE -O3)
# Disable OpenMP for the Bonneel network-simplex wrapper: OpenMP thread overhead
# causes 30-40x slowdown on the problem sizes we handle (n ≤ ~5 000).  POT's
# emd_wrap is also built without OpenMP for the same reason.
target_compile_definitions(_bonneel PRIVATE NOOMP)
install(TARGETS _bonneel DESTINATION sparse_ot/_ext)
