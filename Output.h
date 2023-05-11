#pragma once
#include <vector>
#include "Parameter.h"

namespace cfd {
// Normally, the forward declaration should either be claimed as struct or class, but this time, the type must match
// , or it will not be able to find the corresponding libs.
class Mesh;
struct Field;
struct Species;

class Output {
public:
  const int myid{0};
  const Mesh &mesh;
  std::vector<Field> &field;
  const Parameter &parameter;
  const Species &species;

  Output(integer _myid, const Mesh &_mesh, std::vector<Field> &_field, const Parameter &_parameter,
         const Species &spec);//

//  Output(const Output&)            = delete;
//  Output(Output&&)                 = delete;
//  Output& operator=(const Output&) = delete;
//  Output& operator=(Output&&)      = delete;

//  void print_mesh(int ngg = 0) const;

  void print_field(integer step, int ngg = 0) const;

  ~Output() = default;
};

void write_str(const char *str, FILE *file);
} // cfd