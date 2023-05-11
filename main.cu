#include <cstdio>
#include "Parallel.h"
#include "Parameter.h"
#include "Mesh.h"
#include "ChemData.h"
#include "Driver.h"

int main(int argc, char *argv[]) {
  cfd::MpiParallel mpi_parallel(&argc, &argv);

  cfd::Parameter parameter(mpi_parallel);

  cfd::Mesh mesh(parameter);

  cfd::ChemData chem_data(parameter);
#if MULTISPECIES == 1
#else
  printf("Air computation.\n");
#endif

  cfd::Driver driver(parameter, mesh, chem_data);

  driver.initialize_computation();

  driver.simulate();

  printf("Fuck off\n");
  return 0;
}
