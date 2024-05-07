// Quadrotor tracking example

// This script is just to show how to use the library, the data for this example is not tuned for our Crazyflie demo. Check the firmware code for more details.

// - NSTATES = 12
// - NINPUTS = 4
// - NHORIZON = anything you want
// - NTOTAL = 301 if using reference trajectory from trajectory_data/
// - tinytype = float if you want to run on microcontrollers
// States: x (m), y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi
// phi, theta, psi are NOT Euler angles, they are Rodiguez parameters
// check this paper for more details: https://ieeexplore.ieee.org/document/9326337
// Inputs: u1, u2, u3, u4 (motor thrust 0-1, order from Crazyflie)


#define NSTATES 12
#define NINPUTS 4

#define NHORIZON 10
#define NTOTAL 301

#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/tiny_api.hpp>
#include "problem_data/quadrotor_20hz_params.hpp"
#include "trajectory_data/quadrotor_20hz_y_axis_line.hpp"

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

    int status = tiny_setup(&cache, &work, &solution,
                            work.Adyn, work.Bdyn, work.Q.asDiagonal(), work.R.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON,
                            work.x_min, work.x_max, work.u_min, work.u_max,
                            &settings, 1);

    tiny_VectorNx x0, x1; // current and next simulation states

    // Map data from trajectory_data
    Matrix<tinytype, NSTATES, NTOTAL> Xref_total = Eigen::Map<Matrix<tinytype, NSTATES, NTOTAL>>(Xref_data);
    work.Xref = Xref_total.block<NSTATES, NHORIZON>(0, 0);

    // Initial state
    x0 = work.Xref.col(0);

    // std::cout << work.Xref << std::endl;

    for (int k = 0; k < NTOTAL - NHORIZON; ++k)
    {
        std::cout << "tracking error: " << (x0 - work.Xref.col(1)).norm() << std::endl;

        // 1. Update measurement
        work.x.col(0) = x0;

        // 2. Update reference
        work.Xref = Xref_total.block<NSTATES, NHORIZON>(0, k);

        // 3. Reset dual variables if needed
        work.y = tiny_MatrixNuNhm1::Zero();
        work.g = tiny_MatrixNxNh::Zero();


        // 4. Solve MPC problem
        tiny_solve(&solver);

        // std::cout << work.iter << std::endl;
        // std::cout << work.u.col(0).transpose().format(TinyFmt) << std::endl;

        // 5. Simulate forward
        x0 = work.Adyn * x0 + work.Bdyn * work.u.col(0);

        // std::cout << x0.transpose().format(TinyFmt) << std::endl;
    }

    return 0;
}

} /* extern "C" */