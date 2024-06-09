// Quadrotor tracking example

// This script is just to show how to use the library, the data for this example is not tuned for our Crazyflie demo. Check the firmware code for more details.

// Make sure in glob_opts.hpp:
// - NSTATES = 6, NINPUTS=3
// - NHORIZON = anything you want
// - NTOTAL = 301 if using reference trajectory from trajectory_data/
// - tinytype = float if you want to run on microcontrollers
// States: x (m), y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi
// phi, theta, psi are NOT Euler angles, they are Rodiguez parameters
// check this paper for more details: https://ieeexplore.ieee.org/document/9326337
// Inputs: thrust_x, thrust_y, thrust_z

#define NSTATES 6
#define NINPUTS 3
#define NHORIZON 10
#define NTOTAL 100

#include <iostream>
#include <fstream> // for writing data to csv file for visualization later

#include <tinympc/tiny_api.hpp>
#include "problem_data/rocket_landing_params_20hz.hpp"

Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
Eigen::IOFormat SaveData(4, 0, ", ", "\n");


extern "C"
{

typedef Matrix<tinytype, NINPUTS, NHORIZON-1> tiny_MatrixNuNhm1;
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;
typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;

int main()
{
    TinySolver *solver;
    
    // TinyBounds bounds;
    // TinySocs socs;
    // TinyWorkspace work;
    // work->bounds = &bounds;
    // work->socs = &socs;

    // TinyCache cache;
    // TinySettings settings;
    // TinySolver solver{&settings, &cache, &work};

    // Dynamics and cost
    tinyMatrix Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    tinyMatrix Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn_data);
    tinyVector fdyn = Map<Matrix<tinytype, NSTATES, 1>>(fdyn_data);
    tinyVector Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
    tinyVector R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

    // Box constraints
    tiny_VectorNx x_min_one_time_step(-5.0, -5.0, -0.5, -10.0, -10.0, -20.0);
    tiny_VectorNx x_max_one_time_step(5.0, 5.0, 100.0, 10.0, 10.0, 20.0);
    tinyMatrix x_min = x_min_one_time_step.replicate(1, NHORIZON);
    tinyMatrix x_max = x_max_one_time_step.replicate(1, NHORIZON);
    tinyMatrix u_min = tiny_MatrixNuNhm1::Constant(-10);
    tinyMatrix u_max = tiny_MatrixNuNhm1::Constant(105);

    // SOC constraints
    solver->work->socs->cu[0] = 0.25; // coefficients for input cones (mu)
    solver->work->socs->cx[0] = 0.5; // coefficients for state cones (mu)
    // // Number of contiguous input variables to constrain with each cone
    // // For example if all inputs are [thrust_x, thrust_y, thrust_z, thrust_2x, thrust_2y, thrust_2z]
    // // and we want to put a thrust cone on [thrust_y, thrust_z] we need to set socs->Acu to 1 and socs->qcu to 2
    // // which corresponds to a subvector of all input variables starting at index 1 with length 2.
    // // Support for arbitrary input constraints will be added in the future.
    solver->work->socs->Acu[0] = 0; // start indices for input cones
    solver->work->socs->Acx[0] = 0; // start indices for state cones
    solver->work->socs->qcu[0] = 3; // dimensions for input cones
    solver->work->socs->qcx[0] = 3; // dimensions for state cones

    int status = tiny_setup(&solver,
                            Adyn, Bdyn, fdyn, Q.asDiagonal(), R.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON,
                            x_min, x_max, u_min, u_max, 1);
    
    // Update any settings we want to change
    solver->settings->max_iter = 100;
    solver->settings->abs_pri_tol = 2e-03;

    // Create new pointer to solver->work for brevity
    TinyWorkspace *work = solver->work;

    // Initial state
    tiny_VectorNx xinit(4, 2, 20, -3, 2, -4.5);
    tiny_VectorNx x0 = xinit*1.1;

    // Goal state
    tiny_VectorNx xg(0, 0, 0, 0, 0, 0);
    
    // Uref stays constant
    for (int i=0; i<NHORIZON-1; i++) {
        work->Uref.col(i)(2) = 10;
    }

    // Linearly interpolate Xref
    for (int i=0; i<NHORIZON; i++) {
        work->Xref.col(i) = xinit + (xg - xinit)*tinytype(i)/(NTOTAL-1);
    }
    
    // Set final p to final state in horizon
    work->p.col(NHORIZON-1) = -solver->cache->Pinf*work->Xref.col(NHORIZON-1);

    for (int k = 0; k < NTOTAL - NHORIZON; k++)
    {
        std::cout << "tracking error: " << (x0 - work->Xref.col(1)).norm() << std::endl;

        // 1. Update measurement
        work->x.col(0) = x0;

        // 2. Update reference
        for (int i=0; i<NHORIZON; i++) {
            work->Xref.col(i) = xinit + (xg - xinit)*tinytype(i+k)/(NTOTAL-1);
            if (i < NHORIZON - 1)
                work->Uref.col(i)(2) = 10; // uref stays constant
        }

        // 3. Solve MPC problem
        tiny_solve(solver);

        // 4. Simulate forward
        x0 = work->Adyn*x0 + work->Bdyn*work->u.col(0) + work->fdyn;
    }

    return 0;
}

} /* extern "C" */