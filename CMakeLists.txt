cmake_minimum_required(VERSION 3.24)
project(2_CLionGPU CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_compile_definitions(MULTISPECIES=1)
add_compile_definitions(GPU)

list(APPEND src_file
  BoundCond.h BoundCond.cpp BoundCond.cuh BoundCond.cu
  ChemData.h ChemData.cpp
  DParameter.cu DParameter.h Driver.h Driver.cu
  Element.h Element.cpp
  Field.h Field.cu
  Initialize.h Initialize.cu InviscidScheme.cu InviscidScheme.cuh
  Limiter.cuh
  main.cu Mesh.h Mesh.cpp
  Parallel.h Parallel.cpp Parameter.h Parameter.cpp
  Reconstruction.cu Reconstruction.cuh
  Thermo.cuh Thermo.cu TimeAdvanceFunc.cu TimeAdvanceFunc.cuh Transport.cuh Transport.cu
  ViscousScheme.cu ViscousScheme.cuh

  gxl_lib/Array.hpp
  gxl_lib/MyString.cpp
  )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

cmake_policy(SET CMP0074 NEW)
find_package(MPI REQUIRED)

add_executable(2_CLionGPU ${src_file})

set_target_properties(2_CLionGPU PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(2_CLionGPU
  PROPERTIES
  CXX_STANDARD 20 # A newer version CMake 3.25.2 would support this
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON
  #  "$<$<COMPILE_LANGUAGE:CXX>:${flags_release}>"
  )

target_include_directories(2_CLionGPU
  PUBLIC
  ${CMAKE_SOURCE_DIR}/depends/include/
  )

target_link_directories(2_CLionGPU
  PUBLIC#PRIVATE
  "$<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/depends/lib/debug>"
  "$<$<CONFIG:Release>:${CMAKE_SOURCE_DIR}/depends/lib/release>"
  )

#target_compile_options(2_CLionGPU
#  PUBLIC#private
#  ${flags}
#  "$<$<COMPILE_LANGUAGE:CXX>:${flags_release}>"
#  )

target_link_libraries(2_CLionGPU
  PUBLIC#PRIVATE
  "$<$<CONFIG:Debug>:fmtd>"
  "$<$<CONFIG:Release>:fmt>"
  MPI::MPI_CXX
  )
