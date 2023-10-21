#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/tiny_data_workspace.hpp>

using namespace Eigen;
IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

#ifdef __cplusplus
extern "C"
{
  int main()
  {
    int exitflag;
    std::cout << tiny_data_solver.settings->max_iter << std::endl;
    std::cout << tiny_data_solver.cache->AmBKt.format(CleanFmt) << std::endl;
    std::cout << tiny_data_solver.work->Adyn.format(CleanFmt) << std::endl;

    // exitflag = tiny_solve(&tiny_data_solver);
    
    return 0;
  }
}
#endif
