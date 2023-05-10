#pragma once

#include "Parameter.h"
#include "Field.h"
#include "ChemData.h"


namespace cfd {
class Mesh;

void initialize_basic_variables(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, ChemData &chem_data);

void initialize_from_start(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, ChemData &chem_data);
}