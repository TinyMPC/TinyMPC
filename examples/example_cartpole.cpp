// Inverted pendulum example with codegen, the code is generated in `generated_code` directory, build and run it to see the result.
// Reference: https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling

#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/codegen.hpp>

// For codegen, double type should be used, otherwise, Riccati won't converge.

extern "C"
{
    // Model size
    const int n = 4;  // state dimension
    const int m = 1;  // input dimension
    const int N = 10; // horizon

    // Model matrices
    // Ad' = [1.0 0.0 0.0 0.0; 0.01 1.0 0.0 0.0; 2.2330083403300767e-5 0.004466210576510177 1.0002605176397052 0.05210579005928538; 7.443037974683548e-8 2.2330083403300767e-5 0.01000086835443038 1.0002605176397052]
    tinytype Adyn_data[n * n] = {1.0, 0.0, 0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 2.2330083403300767e-5, 0.004466210576510177, 1.0002605176397052, 0.05210579005928538, 7.443037974683548e-8, 2.2330083403300767e-5, 0.01000086835443038, 1.0002605176397052};
    // Bd = [7.468368562730335e-5, 0.014936765390161838, 3.79763323185387e-5, 0.007595596218554721]
    tinytype Bdyn_data[n * m] = {7.468368562730335e-5, 0.014936765390161838, 3.79763323185387e-5, 0.007595596218554721};

    // Cost matrices
    tinytype Q_data[n] = {10, 1, 10, 1};
    tinytype Qf_data[n] = {10, 1, 10, 1};
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
        tiny_codegen(n, m, N, Adyn_data, Bdyn_data, Q_data, Qf_data, R_data, x_min_data, x_max_data, u_min_data, u_max_data, rho_value, abs_pri_tol, rel_pri_tol, max_iter, verbose, tinympc_dir, output_dir);

        return 0;
    }

} /* extern "C" */


/* Copy this to main in the generated code

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
	x0 << 0.0, 0, 0.2, 0; // initial state

	int i = 0;
	for (int k = 0; k < 200; ++k)
	{
		printf("tracking error at step %2d: %.4f\n", k, (x0 - work->Xref.col(1)).norm());		
		
		// 1. Update measurement
		work->x.col(0) = x0;

		// 2. Update reference (if needed)

		// 3. Reset dual variables (if needed)
		work->y = tiny_MatrixNuNhm1::Zero();
		work->g = tiny_MatrixNxNh::Zero();

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