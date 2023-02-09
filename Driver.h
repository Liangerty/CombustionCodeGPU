#pragma once

#include <vector>
#include "ChemData.h"
#include "Field.h"
#include "Mesh.h"
#include "Parameter.h"
#include "BoundCond.cuh"

namespace cfd {
struct DParameter;
struct InviscidScheme;
struct ViscousScheme;
struct TemporalScheme;

struct Driver {
  Driver(Parameter &parameter, Mesh &mesh_
#if MULTISPECIES == 1
      , ChemData &chem_data
#endif
  );

  void initialize_computation();

  void simulate();

  integer myid=0;
//  integer step=0;
  const Mesh &mesh;
  const Parameter& parameter;
  std::vector<Field> field; // The flowfield data of the simulation. Every block is a "Field" object
#ifdef GPU
  DParameter *param = nullptr; // The parameters used for GPU simulation, datas are stored on GPU while the pointer is on CPU
  DBoundCond bound_cond;  // Boundary conditions
  InviscidScheme **inviscid_scheme = nullptr;
  ViscousScheme **viscous_scheme = nullptr;
  TemporalScheme **temporal_scheme = nullptr;
#endif

private:
  void steady_simulation();
};

__global__ void setup_schemes(cfd::InviscidScheme **inviscid_scheme, cfd::ViscousScheme **viscous_scheme,
                              cfd::TemporalScheme **temporal_scheme, cfd::DParameter *param);
}