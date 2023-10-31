#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/codegen.hpp>

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

    char tinympc_dir[255] = "/home/sam/Git/tinympc/TinyMPC/";
    char output_dir[255] = "generated_code/src";

    int main()
    {
        // Python will call this function with the above data
        tiny_codegen(NSTATES, NINPUTS, NHORIZON, Adyn_data, Bdyn_data, Q_data, Qf_data, R_data, x_min_data, x_max_data, u_min_data, u_max_data, rho_value, 1e-3, 1e-3, 100, 1, tinympc_dir, output_dir);
        // This function copies source code to `generated_code` directory, create workspace data, a main.cpp file

        // generated_code/tinympc/glob_opts.hpp
        // generated_code/tinympc/types.hpp (fixed)
        // generated_code/tinympc/admm.hpp (fixed)
        // generated_code/tinympc/admm.cpp (fixed)
        // generated_code/tinympc/tiny_data_workspace.hpp (fixed)

        // generated_code/src/tiny_data_workspace.cpp (save all cache, settings, workspace data)
        // generated_code/src/tiny_main.cpp (example main that setup and solve the problem)
        // Maybe some CMakelists.txt files if needed

        return 0;
    }

} /* extern "C" */