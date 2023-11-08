// Inverted pendulum example with codegen, the code is generated in `generated_code` directory, build and run it to see the result.
// Reference: https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling

#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/codegen.hpp>

// For codegen, double type should be used, otherwise, Riccati won't converge.

extern "C"
{
    // Model parameters
    const tinytype mc = 0.5;
    const tinytype mp = 0.2;
    const tinytype l = 0.3;
    const tinytype b = 0.1;
    const tinytype g = 9.8;
    const tinytype I = 0.006;

    const int n = 4;  // state dimension
    const int m = 1;  // input dimension
    const int N = 10; // horizon

    const tinytype a11 = (-I + mp * l * l) * b / (I * mc + I * mp + mc * mp * l * l);
    const tinytype a12 = mp * mp * g * l * l / (I * mc + I * mp + mc * mp * l * l);
    const tinytype a31 = -mp * l * b / (I * mc + I * mp + mc * mp * l * l);
    const tinytype a32 = mp * g * l * (mp + mc) / (I * mc + I * mp + mc * mp * l * l);
    const tinytype b1 = (I + mp * l * l) / (I * mc + I * mp + mc * mp * l * l);
    const tinytype b3 = mp * l / (I * mc + I * mp + mc * mp * l * l);

    // Model matrices
    tinytype Adyn_data[n * n] = {0, 1, 0, 1,
                                 0, a11, a12, 0,
                                 0, 0, 0, 1,
                                 0, a31, a32, 0}; // Row-major
    tinytype Bdyn_data[n * m] = {0, b1, 0, b3};

    // Cost matrices
    tinytype Q_data[n] = {1, 0, 1, 0};
    tinytype Qf_data[n] = {10, 0, 10, 0};
    tinytype R_data[m] = {1};
    tinytype rho_value = 0.1;

    // Constraints
    tinytype x_min_data[n * N] = {-10};
    tinytype x_max_data[n * N] = {10};
    tinytype u_min_data[m * (N - 1)] = {-10};
    tinytype u_max_data[m * (N - 1)] = {10};

    // Solver options
    tinytype abs_pri_tol = 1e-3;
    tinytype rel_pri_tol = 1e-3;
    int max_iter = 100;
    int verbose = 1; // for code-gen

    // char tinympc_dir[255] = "your absolute path to tinympc";
    char tinympc_dir[255] = "/home/khai/SSD/Code/TinyMPC";
    char output_dir[255] = "/generated_code/cartpole";

    int main()
    {
        // Python will call this function with the above data
        tiny_codegen(n, m, N, Adyn_data, Bdyn_data, Q_data, Qf_data, R_data, x_min_data, x_max_data, u_min_data, u_max_data, rho_value, abs_pri_tol, rel_pri_tol, max_iter, verbose, tinympc_dir, output_dir);

        return 0;
    }

} /* extern "C" */