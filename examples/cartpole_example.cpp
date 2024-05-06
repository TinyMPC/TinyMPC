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


#define NSTATES 4
#define NINPUTS 1

#define NHORIZON 10
#define NTOTAL 400

#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/tiny_api.hpp>

extern "C"
{

    TinySolution *solution = new TinySolution();
    TinyCache *cache = new TinyCache();
    TinyWorkspace *work = new TinyWorkspace();
    TinySettings *settings = new TinySettings();
    TinySolver solver{solution, settings, cache, work};

    typedef Matrix<tinytype, NINPUTS, NHORIZON-1> tiny_MatrixNuNhm1;
    typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;
    typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;

    int main()
    {
        tinytype Adyn_data[NSTATES * NSTATES] = {1.0, 0.01, 0.0, 0.0, 0.0, 1.0, 0.039, 0.0, 0.0, 0.0, 1.002, 0.01, 0.0, 0.0, 0.458, 1.002};
        tinytype Bdyn_data[NSTATES * NINPUTS] = {0.0, 0.02, 0.0, 0.067};
        tinytype Q_data[NSTATES] = {10.0, 1.0, 10.0, 1.0};
        tinytype R_data[NINPUTS] = {1.0};

        work->Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
        work->Bdyn = Map<Matrix<tinytype, NSTATES, 1>>(Bdyn_data);
        work->Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
        work->R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);
        
        work->x_min = tiny_MatrixNxNh::Constant(-1e17);
        work->x_max = tiny_MatrixNxNh::Constant(1e17);
        work->u_min = tiny_MatrixNuNhm1::Constant(-1e17);
        work->u_max = tiny_MatrixNuNhm1::Constant(1e17);

        settings->abs_pri_tol = 0.001;
        settings->abs_dua_tol = 0.001;
        settings->max_iter = 100;
        settings->check_termination = 1;
        settings->en_input_bound = 1;
        settings->en_state_bound = 1;

        std::cout << work->Q.asDiagonal() << std::endl;

        int status = tiny_setup(cache, work, solution,
                                work->Adyn, work->Bdyn, work->Q.asDiagonal(), work->R.asDiagonal(),
                                5, NSTATES, NINPUTS, NHORIZON,
                                work->x_min, work->x_max, work->u_min, work->u_max,
                                settings, 1);



        tiny_VectorNx x0, x1; // current and next simulation states

        // Map data from trajectory_data
        work->Xref = tinyMatrix::Zero(NSTATES, NHORIZON);

        // Initial state
        x0 << 0.5, 0.0, 0.0, 0.0;

        for (int k = 0; k < NTOTAL - NHORIZON; ++k)
        {
            std::cout << "tracking error: " << (x0 - work->Xref.col(1)).norm() << std::endl;

            // 1. Update measurement
            work->x.col(0) = x0;

            // 2. Update reference

            // 3. Reset dual variables if needed
            work->y = tiny_MatrixNuNhm1::Zero();
            work->g = tiny_MatrixNxNh::Zero();


            // 4. Solve MPC problem
            tiny_solve(&solver);

            // 5. Simulate forward
            x1 = work->Adyn * x0 + work->Bdyn * work->u.col(0);
            x0 = x1;
        }

        return 0;
    }

} /* extern "C" */