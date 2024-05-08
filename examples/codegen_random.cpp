// Example of using codegen to generate C++ code for a random MPC problem
// The code will be generated in the `tinympc_generated_code_random_example` folder

#include <iostream>
#include <filesystem>

#include <tinympc/tiny_api.hpp>
#include <tinympc/codegen.hpp>

#define NSTATES 2  // state dimension: x (m), theta (rad), dx, dtheta
#define NINPUTS 2  // input dimension: F (Newtons)
#define NHORIZON 3 // horizon

extern "C"
{

typedef Matrix<tinytype, NINPUTS, NHORIZON-1, ColMajor> tiny_MatrixNuNhm1;
typedef Matrix<tinytype, NSTATES, NHORIZON, ColMajor> tiny_MatrixNxNh;

std::filesystem::path output_dir_relative = "tinympc_generated_code_random_example/";

int main()
{
    TinySolver *solver;

    tinytype rho_value = 0.1;

    tinytype Adyn_data[NSTATES * NSTATES] = {1, 5, 1, 2};
    tinytype Bdyn_data[NSTATES * NINPUTS] = {3, 3, 4, 1};
    tinytype Q_data[NSTATES] = {1, 1};
    tinytype R_data[NINPUTS] = {2, 2};

    tinytype x_min_data[NSTATES * NHORIZON] = {-1, -2, -1, -2, -1, -2};
    tinytype x_max_data[NSTATES * NHORIZON] = {1, 2, 1, 2, 1, 2};
    tinytype u_min_data[NINPUTS * (NHORIZON - 1)] = {-2, -3, -2, -3};
    tinytype u_max_data[NINPUTS * (NHORIZON - 1)] = {2, 3, 2, 3};
    
    tinyMatrix Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    tinyMatrix Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn_data);
    tinyVector Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
    tinyVector R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

    tinyMatrix x_min = Map<tiny_MatrixNxNh>(x_min_data);
    tinyMatrix x_max = Map<tiny_MatrixNxNh>(x_max_data);
    tinyMatrix u_min = Map<tiny_MatrixNuNhm1>(u_min_data);
    tinyMatrix u_max = Map<tiny_MatrixNuNhm1>(u_max_data);

    int verbose = 0;
    int status = tiny_setup(&solver,
                            Adyn, Bdyn, Q.asDiagonal(), R.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON,
                            x_min, x_max, u_min, u_max, verbose);

    // Solver options
    solver->settings->abs_pri_tol = 1e-3;
    solver->settings->abs_dua_tol = 1e-3;
    solver->settings->max_iter = 100;
    solver->settings->check_termination = 1; 


    tiny_codegen(solver, std::filesystem::absolute(output_dir_relative).string().c_str(), verbose);

    return 0;
}

} /* extern "C" */