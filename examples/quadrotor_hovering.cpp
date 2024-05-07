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

#include <tinympc/admm.hpp>
#include <tinympc/tiny_api.hpp>
#include "problem_data/quadrotor_20hz_params.hpp"

extern "C"
{

TinySolution solution;
TinyCache cache;
TinyWorkspace work;
TinySettings settings;
TinySolver solver{&solution, &settings, &cache, &work};

typedef Matrix<tinytype, NINPUTS, NHORIZON-1> tiny_MatrixNuNhm1;
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;
typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;

int main()
{
    work.Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    work.Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn_data);
    work.Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
    work.R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);
    
    work.x_min = tiny_MatrixNxNh::Constant(-5);
    work.x_max = tiny_MatrixNxNh::Constant(5);
    work.u_min = tiny_MatrixNuNhm1::Constant(-0.5);
    work.u_max = tiny_MatrixNuNhm1::Constant(0.5);

    tiny_set_default_settings(&settings);

    int verbose = 0;
    int status = tiny_setup(&cache, &work, &solution,
                            work.Adyn, work.Bdyn, work.Q.asDiagonal(), work.R.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON,
                            work.x_min, work.x_max, work.u_min, work.u_max,
                            &settings, verbose);

    tiny_VectorNx x0; // current and next simulation states

    // Initial state
    x0 << 0, 1, 0, 0.2, 0, 0, 0.1, 0, 0, 0, 0, 0;

    // Reference trajectory
    tiny_VectorNx Xref_origin;
    Xref_origin << 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    work.Xref = Xref_origin.replicate<1, 10>();

    for (int k = 0; k < 70; ++k)
    {
        printf("tracking error at step %2d: %.4f\n", k, (x0 - work.Xref.col(1)).norm());

        // 1. Update measurement
        tiny_set_x0(&solver, x0);

        // 4. Solve MPC problem
        tiny_solve(&solver);

        // 5. Simulate forward
        x0 = work.Adyn * x0 + work.Bdyn * work.u.col(0);
    }

    return 0;
}

} /* extern "C" */