# There are several options need to be set here
# 1. "CMAKE_CUDA_ARCHITECTURES", the CUDA architecture should be specified
#     For example, for Nvidia Quadro GP100, the value should be 60, while for GV100, 70
# 2. "MAX_SPEC_NUMBER" if mixture, the number of species should not beyond 55, which is set by allowing the computation of CH4 in Grimech mechanism.
#     If larger mechanism is to be used, the variable should be modified to be larger or equal to the species number.
#     However, the correctness, if the resource is available for the corresponding kernel may need to be tested.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

#add_compile_definitions(MULTISPECIES=1)
add_compile_definitions(MAX_SPEC_NUMBER=20)
add_compile_definitions(GPU)

project(2_CLionGPU CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND src_file
  BoundCond.h BoundCond.cpp BoundCond.cu BoundCond.cuh
  ChemData.h ChemData.cpp Constants.h
  DataCommunication.cuh DataCommunication.cu Define.h DParameter.cu DParameter.h DPLURHPP.cuh Driver.cu Driver.cuh
  Element.h Element.cpp
  Field.h Field.cu FieldOperation.cuh FieldOperation.cu
  ImplicitTreatmentHPP.cuh Initialize.cuh Initialize.cu InviscidScheme.cuh
  Limiter.cuh
  main.cu Mesh.h Mesh.cpp
  Output.hpp
  Parallel.h Parallel.cpp Parallel.cu Parameter.h Parameter.cpp PostProcess.h PostProcess.cu
  Reconstruction.cu Reconstruction.cuh
  SchemeSelector.cuh SourceTerm.cuh SST.cuh SST.cu
  Transport.cuh Transport.cu TimeAdvanceFunc.cu TimeAdvanceFunc.cuh Thermo.cuh Thermo.cu
  ViscousScheme.cuh

  gxl_lib/Array.hpp
  gxl_lib/MyString.cpp
  gxl_lib/Time.cpp gxl_lib/Time.h
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

# target_compile_options(2_CLionGPU
#   PUBLIC
#     "$<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>"
#   )

target_link_directories(2_CLionGPU
  PUBLIC#PRIVATE
  "$<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/depends/lib/debug>"
  "$<$<CONFIG:Release>:${CMAKE_SOURCE_DIR}/depends/lib/release>"
  )

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  target_compile_options(2_CLionGPU
    PRIVATE
      "-Wno-unused-result"
  )
endif()

target_link_libraries(2_CLionGPU
  PUBLIC#PRIVATE
  "$<$<CONFIG:Debug>:fmtd>"
  "$<$<CONFIG:Release>:fmt>"
  MPI::MPI_CXX
  )
