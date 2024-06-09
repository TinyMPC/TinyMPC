// Quadrotor hovering example

// This script is just to show how to use the library, the data for this example is not tuned for our Crazyflie demo. Check the firmware code for more details.

// - NSTATES = 12
// - NINPUTS = 4
// - NHORIZON = anything you want
// - tinytype = float if you want to run on microcontrollers
// States: x (m), y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi
// phi, theta, psi are NOT Euler angles, they are Rodiguez parameters
// check this paper for more details: https://ieeexplore.ieee.org/document/9326337
// Inputs: u1, u2, u3, u4 (motor thrust 0-1, order from Crazyflie)


#define NSTATES 12
#define NINPUTS 4

#define NHORIZON 10


#include <iostream>
#include <tinympc/tiny_api.hpp>
#include "problem_data/quadrotor_20hz_params.hpp"

extern "C"
{

typedef Matrix<tinytype, NINPUTS, NHORIZON-1> tiny_MatrixNuNhm1;
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;
typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;

int main()
{
    TinySolver *solver;

    tinyMatrix Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    tinyMatrix Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn_data);
    tinyVector fdyn = tiny_VectorNx::Zero();
    tinyVector Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
    tinyVector R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

    tinyMatrix x_min = tiny_MatrixNxNh::Constant(-5);
    tinyMatrix x_max = tiny_MatrixNxNh::Constant(5);
    tinyMatrix u_min = tiny_MatrixNuNhm1::Constant(-0.5);
    tinyMatrix u_max = tiny_MatrixNuNhm1::Constant(0.5);

    int status = tiny_setup(&solver,
                            Adyn, Bdyn, fdyn, Q.asDiagonal(), R.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON,
                            x_min, x_max, u_min, u_max, 1);
    
    // Update whichever settings we'd like
    solver->settings->max_iter = 100;
    
    // Create new pointer to solver->work for brevity
    TinyWorkspace *work = solver->work;

    // Initial state
    tiny_VectorNx x0;
    x0 << 0, 1, 0, 0.2, 0, 0, 0.1, 0, 0, 0, 0, 0;

    // Reference trajectory
    tiny_VectorNx Xref_origin;
    Xref_origin << 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    work->Xref = Xref_origin.replicate<1, NHORIZON>();

    for (int k = 0; k < 70; ++k)
    {
        printf("tracking error at step %2d: %.4f\n", k, (x0 - work->Xref.col(1)).norm());

        // 1. Update measurement
        tiny_set_x0(solver, x0);

        // 2. Solve MPC problem
        tiny_solve(solver);

        // 3. Simulate forward
        x0 = work->Adyn*x0 + work->Bdyn*work->u.col(0);
    }

    return 0;
}

} /* extern "C" */