// Cartpole example with codegen, the code is generated in `generated_code` directory, build and run it to see the result.
// You just need a discrete-time LTI model of upright cartpole.

#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/codegen.hpp>

// Codegen only cares tinytype in `glob_opts.hpp`
// For codegen, change it to double, otherwise, Riccati may fail.
// The embedded code is still float type.

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
    char tinympc_dir[255] = "/home/khai/SSD/Code/TinyMPC/";
    char output_dir[255] = "/home/khai/SSD/Code/TinyMPC/generated_code";

    int main()
    {
        // Set up constraints (for-loop in main)
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

        // We can also call this function from Python, Matlab, Julia (expected)
        tiny_codegen(n, m, N, Adyn_data, Bdyn_data, Q_data, R_data,
                     x_min_data, x_max_data, u_min_data, u_max_data,
                     rho_value, abs_pri_tol, abs_dual_tol, max_iter, check_termination, gen_wrapper,
                     tinympc_dir, output_dir);

        return 0;
    }

} /* extern "C" */

/* Copy this to tiny_main.cpp in the generated code to run MPC

int main()
{
    int exitflag = 1;
    TinyWorkspace* work = tiny_data_solver.work;
    tiny_data_solver.work->Xref = tiny_MatrixNxNh::Zero();
    tiny_data_solver.work->Uref = tiny_MatrixNuNhm1::Zero();
    tiny_data_solver.settings->max_iter = 150;
    tiny_data_solver.settings->en_input_bound = 1;
    tiny_data_solver.settings->en_state_bound = 1;

    tiny_VectorNx x0, x1; // current and next simulation states
    x0 << 0.0, 0, 0.1, 0; // initial state

    int i = 0;
    for (int k = 0; k < 300; ++k)
    {
        printf("tracking error at step %2d: %.4f\n", k, (x0 - work->Xref.col(1)).norm());

        // 1. Update measurement
        work->x.col(0) = x0;

        // 2. Update reference (if needed)
        // you can also use C wrapper (intended for high-level languages) 
        // by including tiny_wrapper.hpp and call `set_xref(...)` function

        // 3. Reset dual variables (if needed)
        // work->y = tiny_MatrixNuNhm1::Zero();
        // work->g = tiny_MatrixNxNh::Zero();

        // 4. Solve MPC problem
        exitflag = tiny_solve(&tiny_data_solver);

        // if (exitflag == 0)
        // 	printf("HOORAY! Solved with no error!\n");
        // else
        // 	printf("OOPS! Something went wrong!\n");
        // 	// break;

        std::cout << work->iter << std::endl;
        std::cout << work->u.col(0).transpose().format(CleanFmt) << std::endl;

        // 5. Simulate forward
        // work->u.col(0) = -tiny_data_solver.cache->Kinf * (x0 - work->Xref.col(0));
        x1 = work->Adyn * x0 + work->Bdyn * work->u.col(0);
        x0 = x1;
        // std::cout << x0.transpose().format(CleanFmt) << std::endl;
    }
}

*/