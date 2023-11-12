#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/codegen.hpp>

// For codegen, double type should be used, otherwise, Riccati may fail.
// The embedded code is still float type.

extern "C"
{
    // Random data
    const int n = 2;  // state dimension
    const int m = 2;  // input dimension
    const int N = 10;  // horizon
    tinytype Adyn_data[n * n] = {1, 5, 1, 2};  // Row-major
    tinytype Bdyn_data[n * m] = {3, 3, 4, 1};
    tinytype Q_data[n] = {1, 1};
    tinytype Qf_data[n] = {1, 1};
    tinytype R_data[m] = {2, 2};
    tinytype rho_value = 0.1;

    tinytype x_min_data[n * N];
    tinytype x_max_data[n * N];
    tinytype u_min_data[m * (N - 1)];
    tinytype u_max_data[m * (N - 1)];

    // char tinympc_dir[255] = "your absolute path to tinympc";
    char tinympc_dir[255] = "/home/khai/SSD/Code/TinyMPC";
    char output_dir[255] = "/generated_code";

    int main()
    {
        // Set up constraints (for-loop in main)
        int i = 0;
        for (i = 0; i < n * N; i++) {
            x_min_data[i] = -5;
            x_max_data[i] = 5;
        }
        for (i = 0; i < m * (N - 1); i++) {
            u_min_data[i] = -5;
            u_max_data[i] = 5;
        }

        // Python will call this function with the above data
        tiny_codegen(n, m, N, Adyn_data, Bdyn_data, Q_data, Qf_data, R_data, x_min_data, x_max_data, u_min_data, u_max_data, rho_value, 1e-3, 1e-3, 100, 1, tinympc_dir, output_dir);
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