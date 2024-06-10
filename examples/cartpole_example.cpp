// Quadrotor tracking example

// This script is just to show how to use the library, the data for this example is not tuned for our Crazyflie demo. Check the firmware code for more details.

// - NSTATES = 4
// - NINPUTS = 1
// - NHORIZON = anything
// - NTOTAL = anything greater than NHORIZON
// States: x, theta, dx, dtheta
// Inputs: F (force on cart)

#include <iostream>

#include <tinympc/tiny_api.hpp>


#define NSTATES 4
#define NINPUTS 1
#define NHORIZON 10
#define NTOTAL 400

extern "C"
{

typedef Matrix<tinytype, NINPUTS, NHORIZON-1> tiny_MatrixNuNhm1;
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;
typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;

int main()
{
    TinySolver *solver;

    float rho_value = 1.0;

    tinytype Adyn_data[NSTATES * NSTATES] = {1.0, 0.01, 0.0, 0.0, 0.0, 1.0, 0.039, 0.0, 0.0, 0.0, 1.002, 0.01, 0.0, 0.0, 0.458, 1.002};
    tinytype Bdyn_data[NSTATES * NINPUTS] = {0.0, 0.02, 0.0, 0.067};
    tinytype Q_data[NSTATES] = {10.0, 1.0, 10.0, 1.0};
    tinytype R_data[NINPUTS] = {1.0};

    tinyMatrix Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    tinyMatrix Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS>>(Bdyn_data);
    tinyVector fdyn = tiny_VectorNx::Zero();
    tinyVector Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
    tinyVector R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

    tinyMatrix x_min = tiny_MatrixNxNh::Constant(-1e17);
    tinyMatrix x_max = tiny_MatrixNxNh::Constant(1e17);
    tinyMatrix u_min = tiny_MatrixNuNhm1::Constant(-1e17);
    tinyMatrix u_max = tiny_MatrixNuNhm1::Constant(1e17);

    // Set up problem
    int status = tiny_setup(&solver,
                            Adyn, Bdyn, fdyn, Q.asDiagonal(), R.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON, 1);
    // Set bound constraints
    status = tiny_set_bounds(solver, x_min, x_max, u_min, u_max);
    
    // Update whichever settings we'd like
    solver->settings->max_iter = 100;
    
    // Alias solver->work for brevity
    TinyWorkspace *work = solver->work;

    // Initial state
    tiny_VectorNx x0;
    x0 << 0.5, 0.0, 0.0, 0.0;

    // Reference trajectory
    tiny_VectorNx Xref_origin;
    Xref_origin << 1.0, 0, 0, 0;
    work->Xref = Xref_origin.replicate<1, 10>();

    for (int k = 0; k < NTOTAL - NHORIZON; ++k)
    {
        std::cout << "tracking error: " << (x0 - work->Xref.col(1)).norm() << std::endl;

        // 1. Update measurement
        tiny_set_x0(solver, x0);

        // 2. Solve MPC problem
        tiny_solve(solver);

        // 3. Simulate forward
        x0 = work->Adyn * x0 + work->Bdyn * work->u.col(0);
    }

    return 0;
}

} /* extern "C" */