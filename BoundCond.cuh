#pragma once

#include "Define.h"
#include <vector_types.h>
#include "BoundCond.h"
#include <vector>
#include "Mesh.h"

namespace cfd {

struct BCInfo {
  integer label = 0;
  integer n_boundary = 0;
  int2 *boundary = nullptr;
};

class Mesh;

template<typename BCType>
void register_bc(BCType *&bc, Parameter &parameter, int n_bc, std::vector<integer> &indices, BCInfo *&bc_info, Species &species);

struct DZone;
struct DParameter;
struct Field;

struct DBoundCond {
  DBoundCond()=default;

  void initialize_bc_on_GPU(Mesh &mesh, std::vector<Field> &field, Species &species, Parameter &parameter);

  void link_bc_to_boundaries(Mesh &mesh, std::vector<Field>& field) const;

  void apply_boundary_conditions(const Block &block, Field &field, DParameter *param) const;
//  void apply_boundary_conditions(const Mesh &mesh, std::vector<Field> &field, DParameter *param) const;

  integer n_wall = 0, n_inflow = 0, n_outflow = 0;
  BCInfo *wall_info = nullptr;
  BCInfo *inflow_info = nullptr;
  BCInfo *outflow_info = nullptr;
  Wall *wall = nullptr;
  Inflow *inflow = nullptr;
  Outflow *outflow = nullptr;
};

void count_boundary_of_type_bc(const std::vector<Boundary> &boundary, integer n_bc, integer *sep, integer blk_idx,
                               integer n_block, BCInfo *bc_info);

void link_boundary_and_condition(const std::vector<Boundary> &boundary, BCInfo *bc, integer n_bc, const integer *sep,
                                 integer i_zone);

__global__ void apply_outflow(DZone *zone, integer i_face);

__global__ void apply_inflow(DZone *zone, Inflow *inflow, DParameter *param, integer i_face);

__global__ void apply_wall(DZone *zone, Wall *wall, DParameter *param, integer i_face);

} // cfd
