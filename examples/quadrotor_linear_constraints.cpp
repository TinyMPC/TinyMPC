// Quadrotor Linear Constraint Demo

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

    // Box constraints
    tinyMatrix x_min = tiny_MatrixNxNh::Constant(-15);
    tinyMatrix x_max = tiny_MatrixNxNh::Constant(15);
    tinyMatrix u_min = tiny_MatrixNuNhm1::Constant(-5.0);
    tinyMatrix u_max = tiny_MatrixNuNhm1::Constant(5.0);

    // Set up solver
    int status = tiny_setup(&solver,
                            Adyn, Bdyn, fdyn, Q.asDiagonal(), R.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON, 1);
    
    // Set bound constraints
    status = tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);

    // ========================================
    // LINEAR CONSTRAINTS: Altitude safety
    // ========================================
    
    // State constraint: altitude ceiling z <= 3.0
    int num_state_constraints = 1;
    tinyMatrix Alin_x(num_state_constraints, NSTATES);
    Alin_x.setZero();
    Alin_x(0, 2) = 1.0;   // z coefficient
    
    tinyVector blin_x(num_state_constraints);
    blin_x << 3.0;   // z <= 3.0 (altitude ceiling)
    
    // Input constraint: total thrust <= 6.0
    int num_input_constraints = 1;
    tinyMatrix Alin_u(num_input_constraints, NINPUTS);
    Alin_u.setZero();
    Alin_u(0, 0) = 1.0;  // u1 coefficient
    Alin_u(0, 1) = 1.0;  // u2 coefficient
    Alin_u(0, 2) = 1.0;  // u3 coefficient
    Alin_u(0, 3) = 1.0;  // u4 coefficient
    
    tinyVector blin_u(num_input_constraints);
    blin_u << 6.0;   // total thrust <= 6.0
    
    // Set linear constraints
    status = tiny_set_linear_constraints(solver, Alin_x, blin_x, Alin_u, blin_u);
    
    // Solver settings
    solver->settings->max_iter = 100;
    solver->settings->abs_pri_tol = 1e-3;
    solver->settings->abs_dua_tol = 1e-3;

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
        }

        // Set current state
        tiny_set_x0(solver, x0);

        // Solve MPC problem
        tiny_solve(solver);

        // Track error to goal (like rocket_landing_mpc.cpp)
        tinytype tracking_error = (x0.head(3) - xgoal.head(3)).norm();
        
        std::cout << "tracking error: " << std::setprecision(3) << tracking_error;
        
        // Check altitude violation (separate safety check)
        tinytype z_val = x0(2);
        if (z_val > 3.0 + 1e-6) {
            std::cout << ", altitude violation: z=" << std::setprecision(2) << z_val;
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