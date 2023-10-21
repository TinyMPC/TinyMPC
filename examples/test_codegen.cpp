#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/codegen.hpp>

using Eigen::Matrix;

#define DT 1 / 100

// For codegen, double type should be used, otherwise, Riccati won't converge.

extern "C"
{
    tinytype Adyn_data[NSTATES * NSTATES] = {1, 5, 1, 2};  // Row-major
    tinytype Bdyn_data[NSTATES * NINPUTS] = {3, 3, 4, 1};
    tinytype Q_data[NSTATES] = {1, 1};
    tinytype Qf_data[NSTATES] = {1, 1};
    tinytype R_data[NINPUTS] = {2, 2};
    tinytype rho_value = 0.1;

    tinytype x_min_data[NSTATES * NHORIZON] = {-1.1};
    tinytype x_max_data[NSTATES * NHORIZON] = {1.1};
    tinytype u_min_data[NINPUTS * (NHORIZON - 1)] = {-0.5};
    tinytype u_max_data[NINPUTS * (NHORIZON - 1)] = {0.5};

    int main()
    {
        // Python will call this function with the above data
        tiny_codegen(NSTATES, NINPUTS, NHORIZON, Adyn_data, Bdyn_data, Q_data, Qf_data, R_data, x_min_data, x_max_data, u_min_data, u_max_data, rho_value, 1e-3, 1e-3, 100, 1);

        return 0;
    }

} /* extern "C" */