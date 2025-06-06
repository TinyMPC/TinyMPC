// Quadrotor Linear Constraint Demo - Obstacle Avoidance
// Demonstrates linear constraints for obstacle avoidance using equation (21)

#include <iostream>
#include <cmath>
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

    // Load proper quadrotor dynamics (50Hz discretization)
    tinyMatrix Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    tinyMatrix Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn_data);
    tinyVector fdyn = tinyVector::Zero(NSTATES);  // No affine term for this example
    tinyVector Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
    tinyVector R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

    // Box constraints (conservative bounds)
    tinyMatrix x_min = tiny_MatrixNxNh::Constant(-10);
    tinyMatrix x_max = tiny_MatrixNxNh::Constant(10);
    tinyMatrix u_min = tiny_MatrixNuNhm1::Constant(-2.0);
    tinyMatrix u_max = tiny_MatrixNuNhm1::Constant(2.0);

    // Set up solver (use rho from parameter file)
    int status = tiny_setup(&solver,
                            Adyn, Bdyn, fdyn, Q.asDiagonal(), R.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON, 1);
    
    // Set bound constraints
    status = tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);
    
    // Solver settings
    solver->settings->max_iter = 50;
    solver->settings->abs_pri_tol = 1e-3;
    solver->settings->abs_dua_tol = 1e-3;
    
    // Note: Linear constraints must be enabled in tiny_api_constants.hpp (TINY_DEFAULT_EN_STATE_LINEAR = 1)

    TinyWorkspace *work = solver->work;

    // Initial state: [x, y, z, phi, theta, psi, vx, vy, vz, vphi, vtheta, vpsi]
    tiny_VectorNx x0;
    x0 << -2, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    // Goal state (hover at origin)
    tiny_VectorNx xgoal;
    xgoal << 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    // Obstacle parameters
    tinyVector obstacle_center(3);
    obstacle_center << 0.0, 0.0, 1.0; // Obstacle at origin
    tinytype obstacle_radius = 0.8;

    std::cout << "=== Quadrotor Linear Constraint Demo ===" << std::endl;
    std::cout << "Avoiding spherical obstacle using adaptive linear constraints" << std::endl;
    std::cout << "Obstacle: center=[0,0,1], radius=" << obstacle_radius << std::endl << std::endl;

    for (int k = 0; k < NTOTAL - NHORIZON; ++k)
    {
        // Update reference trajectory (linear interpolation to goal)
        for (int i = 0; i < NHORIZON; i++) {
            tinytype alpha = tinytype(k + i) / (NTOTAL - 1);
            work->Xref.col(i) = (1 - alpha) * x0 + alpha * xgoal;
        }

        // Set up adaptive linear constraint for obstacle avoidance
        // Create constraint based on current quadrotor position relative to obstacle
        tinyVector to_obstacle = x0.head(3) - obstacle_center;
        tinytype dist_to_obstacle = to_obstacle.norm();
        
        tinyMatrix Alin_x;
        tinyVector blin_x;
        
        if (dist_to_obstacle > 1e-6 && dist_to_obstacle < 3.0) { // Only apply constraint when near obstacle
            // Normal vector pointing away from obstacle (toward current position)
            tinyVector normal = to_obstacle / dist_to_obstacle;
            
            // Constraint: normal^T * position >= distance_threshold
            // This keeps the quadrotor outside the obstacle sphere
            tinytype distance_threshold = obstacle_radius + 0.2; // Safety margin
            
            // Set up single linear constraint that applies to all time steps
            Alin_x = tinyMatrix::Zero(1, NSTATES);
            blin_x = tinyVector::Zero(1);
            
            // We use the negative normal to make it a ≤ constraint: (-normal)^T * x ≤ -distance_threshold
            Alin_x.row(0).head(3) = -normal.transpose();
            blin_x(0) = -distance_threshold;
            
            std::cout << "Step " << k << ": Applying obstacle constraint with normal=[" 
                      << normal(0) << ", " << normal(1) << ", " << normal(2) << "]" << std::endl;
        } else {
            // No constraints when far from obstacle
            Alin_x = tinyMatrix::Zero(0, NSTATES);
            blin_x = tinyVector::Zero(0);
            std::cout << "Step " << k << ": No obstacle constraint (distance=" << dist_to_obstacle << ")" << std::endl;
        }

        // Update linear constraints
        tiny_set_linear_constraints(solver, Alin_x, blin_x, tinyMatrix::Zero(0, NINPUTS), tinyVector::Zero(0));

        // Set current state
        tiny_set_x0(solver, x0);

        // Solve MPC problem
        tiny_solve(solver);

        // Check if solved successfully
        if (solver->solution->solved) {
            std::cout << "  Solved in " << solver->solution->iter << " iterations" << std::endl;
        } else {
            std::cout << "  Failed to converge!" << std::endl;
        }

        // Print current position and distance to obstacle
        std::cout << "  Position: [" << x0(0) << ", " << x0(1) << ", " << x0(2) << "]";
        std::cout << ", Distance to obstacle: " << dist_to_obstacle;
        std::cout << ", Safe: " << (dist_to_obstacle > obstacle_radius ? "yes" : "no") << std::endl;

        // Simulate forward
        x0 = work->Adyn * x0 + work->Bdyn * work->u.col(0) + work->fdyn;
    }

    std::cout << std::endl << "=== Demo Complete ===" << std::endl;
    std::cout << "Final position: [" << x0(0) << ", " << x0(1) << ", " << x0(2) << "]" << std::endl;
    std::cout << "Goal position:  [" << xgoal(0) << ", " << xgoal(1) << ", " << xgoal(2) << "]" << std::endl;
    std::cout << "Tracking error: " << (x0.head(3) - xgoal.head(3)).norm() << std::endl;

    return 0;
}

} /* extern "C" */ 