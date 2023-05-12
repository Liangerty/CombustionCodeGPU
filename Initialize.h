#pragma once

#include "Parameter.h"
#include "Field.h"
#include "ChemData.h"


namespace cfd {
class Mesh;

void initialize_basic_variables(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, ChemData &chem_data);

void initialize_from_start(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, ChemData &chem_data);

void read_flowfield(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, ChemData &chem_data);

/**
 * @brief To relate the order of variables from the flowfield files to bv, yk, turbulent arrays
 * @param var_name the array which contains all variables from the flowfield files
 * @return an array of orders. 0~5 means density, u, v, w, p, T; 6~5+ns means the species order, 6+ns~... means other variables such as mut...
 */
std::vector<integer> identify_variable_labels(std::vector<std::string>& var_name, ChemData &chem_data);

void read_one_useless_variable(FILE *fp, integer mx, integer my, integer mz, integer data_format);

#if MULTISPECIES==1
void initialize_spec_from_inflow(cfd::Parameter &parameter, const cfd::Mesh &mesh, std::vector<Field> &field, cfd::ChemData &chem_data);
#endif
}