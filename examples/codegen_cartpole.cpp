// Cartpole example with codegen, the code will be generated in the `tinympc_generated_code_cartpole_example` folder
// Build and run the example main function after generation

#include <iostream>
#ifdef __MINGW32__
#include <experimental/filesystem>
namespace std_fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace std_fs = std::filesystem;
#endif

#include <tinympc/tiny_api.hpp>
#include <tinympc/codegen.hpp>

#define NSTATES 4   // state dimension: x (m), theta (rad), dx, dtheta
#define NINPUTS 1   // input dimension: F (Newtons)
#define NHORIZON 10 // horizon

extern "C" {

typedef Matrix<tinytype, NINPUTS, NHORIZON-1> tiny_MatrixNuNhm1;
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;

std_fs::path output_dir_relative = "tinympc_generated_code_cartpole_example/";

int main()
{
    TinySolver *solver;

    tinytype rho_value = 1.0;

    // Discrete, linear model of upright cartpole.
    tinytype Adyn_data[NSTATES * NSTATES] = {1.0, 0.01, 0.0, 0.0, 0.0, 1.0, 0.039, 0.0, 0.0, 0.0, 1.002, 0.01, 0.0, 0.0, 0.458, 1.002};
    tinytype Bdyn_data[NSTATES * NINPUTS] = {0.0, 0.02, 0.0, 0.067};
    tinytype Q_data[NSTATES] = {10.0, 1.0, 10.0, 1.0};
    tinytype R_data[NINPUTS] = {1.0};

    tinyMatrix Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    tinyMatrix Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS>>(Bdyn_data);
    tinyVector Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
    tinyVector R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

    tinyMatrix x_min = tiny_MatrixNxNh::Constant(-1e17);
    tinyMatrix x_max = tiny_MatrixNxNh::Constant(1e17);
    tinyMatrix u_min = tiny_MatrixNuNhm1::Constant(-1e17);
    tinyMatrix u_max = tiny_MatrixNuNhm1::Constant(1e17);

    int verbose = 0;
    int status = tiny_setup(&solver,
                            Adyn, Bdyn, Q.asDiagonal(), R.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON,
                            x_min, x_max, u_min, u_max, verbose);

    tiny_codegen(solver, std_fs::absolute(output_dir_relative).string().c_str(), verbose);

    return 0;
}

} /* extern "C" */
