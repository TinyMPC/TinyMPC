// Cartpole example with codegen, the code is generated in `generated_code` directory, build and run it to see the result.
// You just need a discrete-time LTI model of upright cartpole.

#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/codegen.hpp>

extern "C"
{

    // Model size
    const int n = 4;  // state dimension: x (m), theta (rad), dx, dtheta
    const int m = 1;  // input dimension: F (Newtons)
    const int N = 10; // horizon

    // Model matrices (all matrices are col-major to be consistent with Eigen)
    tinytype Adyn_data[n * n] = {1.0, 0.0, 0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 2.2330083403300767e-5, 0.004466210576510177, 1.0002605176397052, 0.05210579005928538, 7.443037974683548e-8, 2.2330083403300767e-5, 0.01000086835443038, 1.0002605176397052};
    tinytype Bdyn_data[n * m] = {7.468368562730335e-5, 0.014936765390161838, 3.79763323185387e-5, 0.007595596218554721};

    // Cost matrices
    tinytype Q_data[n] = {10, 1, 10, 1};
    tinytype R_data[m] = {1};
    tinytype rho_value = 0.1;

    // Constraints
    tinytype x_min_data[n * N];
    tinytype x_max_data[n * N];
    tinytype u_min_data[m * (N - 1)];
    tinytype u_max_data[m * (N - 1)];

    // Solver options
    tinytype abs_pri_tol = 1e-3;
    tinytype abs_dual_tol = 1e-3;
    int max_iter = 100;
    int check_termination = 1; 
    int gen_wrapper = 1;

    // char tinympc_dir[255] = "/your/absolute/path/to/TinyMPC"; // TODO: relative path
    char tinympc_dir[255] = "/home/sam/Git/tinympc/TinyMPC"; 
    char output_dir[255] = "/home/sam/Git/tinympc/TinyMPC/generated_code";

    int main()
    {
        // Set up constraints 
        int i = 0;
        for (i = 0; i < n * N; i++)
        {
            x_min_data[i] = -5;
            x_max_data[i] = 5;
        }
        for (i = 0; i < m * (N - 1); i++)
        {
            u_min_data[i] = -5;
            u_max_data[i] = 5;
        }

        // TODO: change this to be like a normal example with the setup function so we can use the solver here
        TinySolver* solver;
        TinySettings* settings;

        tiny_codegen(solver, );

        return 0;
    }

} /* extern "C" */
