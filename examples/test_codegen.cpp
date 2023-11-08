#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/codegen.hpp>

// For codegen, double type should be used, otherwise, Riccati won't converge.

extern "C"
{
    // tinytype Adyn_data[NSTATES * NSTATES] = {1, 5, 1, 2};  // Row-major
    // tinytype Bdyn_data[NSTATES * NINPUTS] = {3, 3, 4, 1};
    // tinytype Q_data[NSTATES] = {1, 1};
    // tinytype Qf_data[NSTATES] = {1, 1};
    // tinytype R_data[NINPUTS] = {2, 2};
    // tinytype rho_value = 0.1;

    // tinytype x_min_data[NSTATES * NHORIZON] = {-1.1};
    // tinytype x_max_data[NSTATES * NHORIZON] = {1.1};
    // tinytype u_min_data[NINPUTS * (NHORIZON - 1)] = {-0.5};
    // tinytype u_max_data[NINPUTS * (NHORIZON - 1)] = {0.5};

    // char tinympc_dir[255] = "/home/sam/Git/tinympc/TinyMPC";
    // char output_dir[255] = "/generated_code";

    tinytype Adyn_data[NSTATES * NSTATES] = {1.0, 0.0, 0.0, 0.0, 0.003924, 0.0, 0.02, 0.0, 0.0, 0.0, 1.3080000000000002e-5, 0.0, 0.0, 1.0, 0.0, -0.003924, 0.0, 0.0, 0.0, 0.02, 0.0, -1.3080000000000002e-5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.3924, 0.0, 1.0, 0.0, 0.0, 0.0, 0.001962, 0.0, 0.0, 0.0, 0.0, -0.3924, 0.0, 0.0, 0.0, 1.0, 0.0, -0.001962, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    tinytype Bdyn_data[NSTATES * NINPUTS] = {-1.8096638963250834e-5, 1.9899808540159687e-5, 1.8153045375848536e-5, -1.995621495275738e-5, 1.8008191254201994e-5, 1.9831505978467163e-5, -1.8028336401558313e-5, -1.981136083111085e-5, 0.0008408571158571429, 0.0008408571158571429, 0.0008408571158571429, 0.0008408571158571429, -0.027535460633336378, -0.03032340363679995, 0.02756626361094543, 0.030292600659190892, -0.027670701778670998, 0.030427841804525506, 0.027756950115976353, -0.030514090141830853, 0.001974771168450853, -0.000722363993435152, -0.0027843760966088102, 0.0015319689215931094, -0.0036193277926501667, 0.003979961708031937, 0.003630609075169707, -0.003991242990551476, 0.0036016382508403987, 0.003966301195693433, -0.0036056672803116627, -0.00396227216622217, 0.08408571158571428, 0.08408571158571428, 0.08408571158571428, 0.08408571158571428, -5.507092126667275, -6.06468072735999, 5.513252722189086, 6.058520131838179, -5.5341403557342, 6.085568360905102, 5.551390023195271, -6.10281802836617, 0.3949542336901706, -0.1444727986870304, -0.556875219321762, 0.30639378431862185};
    tinytype Q_data[NSTATES] = {10000.0, 10000.0, 10000.0, 4.0, 4.0, 399.99999999999994, 4.0, 4.0, 4.0, 2.0408163265306127, 2.0408163265306127, 4.0};
    tinytype Qf_data[NSTATES] = {10000.0, 10000.0, 10000.0, 4.0, 4.0, 399.99999999999994, 4.0, 4.0, 4.0, 2.0408163265306127, 2.0408163265306127, 4.0};
    tinytype R_data[NINPUTS] = {99, 99, 99, 99};
    tinytype rho_value = 0.1;

    tinytype x_min_data[NSTATES * NHORIZON] = {-100.0};
    tinytype x_max_data[NSTATES * NHORIZON] = {100.0};
    tinytype u_min_data[NINPUTS * (NHORIZON - 1)] = {-0.5};
    tinytype u_max_data[NINPUTS * (NHORIZON - 1)] = {0.5};

    char tinympc_dir[255] = "/Users/anoushkaalavill/Documents/REx_Lab/TinyMPC";
    char output_dir[255] = "/generated_code";

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