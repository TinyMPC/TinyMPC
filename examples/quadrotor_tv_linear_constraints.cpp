// Quadrotor Time Varying Linear Constraint Demo

#include <iostream>
#include <iomanip>
#include <tinympc/tiny_api.hpp>

#define NSTATES 12
#define NINPUTS 4
#define NHORIZON 10
#define NTOTAL 50

#include "problem_data/quadrotor_50hz_params.hpp"

extern "C" {

typedef Matrix<tinytype, NINPUTS, NHORIZON-1> tiny_MatrixNuNhm1;
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;
typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;

int main()
{
    TinySolver *solver;

    // Load quadrotor dynamics
    tinyMatrix Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    tinyMatrix Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn_data);
    tinyVector fdyn = tinyVector::Zero(NSTATES);
    tinyVector Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
    tinyVector R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

    // Set up solver
    int status = tiny_setup(&solver,
                            Adyn, Bdyn, fdyn, Q.asDiagonal(), R.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON, 1);
    
    // ========================================
    // TIME VARYING LINEAR CONSTRAINTS: Altitude safety
    // ========================================
    int num_state_constraints = 1;
    // State constraint: altitude ceiling z <= z_lim_total(t)
    tinyMatrix z_lim_total(num_state_constraints, NTOTAL);
    for (int i = 0; i < NTOTAL; i++) {
        z_lim_total(0, i) = 1.1 + (3.0 - 1.1) * i / (NTOTAL - NHORIZON - 1);
    }

    tinyMatrix tv_Alin_x(num_state_constraints * NHORIZON, NSTATES);
    tinyMatrix tv_blin_x(num_state_constraints, NHORIZON);
    tv_Alin_x.setZero();
    tv_blin_x.setZero();
    for (int i = 0; i < NHORIZON; i++) {
        tv_Alin_x(i*num_state_constraints, 2) = 1.0;   // z coefficient
        tv_blin_x(0, i) = 3.0;   // z <= 3.0 (altitude ceiling)
    }

    // Input constraint: total thrust <= 6.0
    int num_input_constraints = 1;
    tinyMatrix tv_Alin_u(num_input_constraints * (NHORIZON-1), NINPUTS);
    tinyMatrix tv_blin_u(num_input_constraints, NHORIZON-1);
    tv_Alin_u.setZero();
    tv_blin_u.setZero();
    for (int i = 0; i < NHORIZON-1; i++) {
        tv_Alin_u(i*num_input_constraints, 0) = 1.0;      // u1 coefficient
        tv_Alin_u(i*num_input_constraints, 1) = 1.0;  // u2 coefficient
        tv_Alin_u(i*num_input_constraints, 2) = 1.0;  // u3 coefficient
        tv_Alin_u(i*num_input_constraints, 3) = 1.0;  // u4 coefficient

        tv_blin_u(0, i) = 6.0;   // total thrust <= 6.0
    }
    
    // Set time varying linear constraints
    status = tiny_set_tv_linear_constraints(solver, tv_Alin_x, tv_blin_x, tv_Alin_u, tv_blin_u);

    // Solver settings
    solver->settings->max_iter = 100;
    solver->settings->abs_pri_tol = 1e-3;
    solver->settings->abs_dua_tol = 1e-3;
    
    // Disable bound constraints (enabled by default)
    solver->settings->en_state_bound = 0;
    solver->settings->en_input_bound = 0;
    solver->settings->en_tv_state_linear = 1;
    solver->settings->en_tv_input_linear = 1;

    TinyWorkspace *work = solver->work;

    // Initial and goal states - goal is above altitude limit
    tiny_VectorNx x0;
    x0 << -2.0, -2.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0;  // Start position
    tiny_VectorNx xgoal;
    xgoal << 2.0, 2.0, 4.0, 0, 0, 0, 0, 0, 0, 0, 0, 0; // Goal above altitude limit

    for (int k = 0; k < NTOTAL - NHORIZON; ++k)
    {
        // Simple reference trajectory
        for (int i = 0; i < NHORIZON; i++) {
            tinytype alpha = tinytype(k + i) / (NTOTAL - 1);
            work->Xref.col(i) = (1 - alpha) * x0 + alpha * xgoal;
            tv_blin_x(0, i) = z_lim_total(0, k + i);   // z <= z_lim_total(t)
        }

        // Set current state
        tiny_set_x0(solver, x0);
        status = tiny_set_tv_linear_constraints(solver, tv_Alin_x, tv_blin_x, tv_Alin_u, tv_blin_u);

        // Solve MPC problem
        tiny_solve(solver);

        // Track error to goal (like rocket_landing_mpc.cpp)
        tinytype tracking_error = (x0.head(3) - xgoal.head(3)).norm();
        
        std::cout << "tracking error: " << std::setprecision(3) << tracking_error;
        
        // Check altitude violation (separate safety check)
        tinytype z_val = x0(2);
        if (z_val > z_lim_total(0, k) + 1e-6) {
            std::cout << ", altitude violation: z=" << std::setprecision(2) << z_val
                      << " > limit=" << std::setprecision(2) << z_lim_total(0, k);
        }
        
        std::cout << std::endl;

        std::cout << "  states: ";
        for (int i = 0; i < 3; ++i) {
            std::cout << std::setprecision(3) << x0(i) << " ";
        }
        std::cout << "inputs: ";
        for (int i = 0; i < NINPUTS; ++i) {
            std::cout << std::setprecision(3) << work->u(i, 0) << " ";
        }
        std::cout << std::endl;

        // Simulate forward
        if (solver->solution->solved) {
            x0 = work->Adyn * x0 + work->Bdyn * work->u.col(0) + work->fdyn;
        } else {
            // If solve failed, try a small step towards goal
            x0 = 0.98 * x0 + 0.02 * xgoal;
        }
    }

    return 0;
}

} /* extern "C" */ 