// Obstacle Avoidance with SDP Constraints using State Lifting


#define NSTATES NX_AUG    // 20 (augmented states)
#define NINPUTS NU_AUG    // 22 (augmented controls)
#define NHORIZON 31
#define NTOTAL 50

#include <iostream>
#include <fstream>
#include <chrono>
#include <tinympc/tiny_api.hpp>
#include "problem_data/obstacle_avoidance_sdp_params.hpp"

Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
Eigen::IOFormat SaveData(4, 0, ", ", "\n");

typedef Matrix<tinytype, NINPUTS, NHORIZON-1> tiny_MatrixNuNhm1;
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;
typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;

// Function to extract physical state from augmented state
Eigen::Matrix<tinytype, NX_PHYS, 1> extract_physical_state(const Eigen::Matrix<tinytype, NX_AUG, 1>& x_aug) {
    return x_aug.head<NX_PHYS>();
}

// Function to construct augmented state from physical state and quadratic terms
Eigen::Matrix<tinytype, NX_AUG, 1> construct_augmented_state(const Eigen::Matrix<tinytype, NX_PHYS, 1>& x_phys) {
    Eigen::Matrix<tinytype, NX_AUG, 1> x_aug;
    // Physical state
    x_aug.head<NX_PHYS>() = x_phys;
    
    // Quadratic terms vec(xx^T)
    Eigen::Matrix<tinytype, NX_PHYS, NX_PHYS> xx = x_phys * x_phys.transpose();
    
    // Vectorize xx^T (column-major order)
    int idx = NX_PHYS;
    for (int j = 0; j < NX_PHYS; j++) {
        for (int i = 0; i < NX_PHYS; i++) {
            x_aug(idx++) = xx(i, j);
        }
    }
    return x_aug;
}

// Function to construct augmented control from physical control and cross terms
void construct_augmented_control(const Eigen::Matrix<tinytype, NX_PHYS, 1>& x_phys,
                                const Eigen::Matrix<tinytype, NU_PHYS, 1>& u_phys,
                                Eigen::Matrix<tinytype, NU_AUG, 1>& u_aug) {
    // Physical control
    u_aug.head<NU_PHYS>() = u_phys;
    
    // Cross terms vec(xu^T), vec(ux^T), vec(uu^T)
    Eigen::Matrix<tinytype, NX_PHYS, NU_PHYS> xu = x_phys * u_phys.transpose();
    Eigen::Matrix<tinytype, NU_PHYS, NX_PHYS> ux = u_phys * x_phys.transpose();  
    Eigen::Matrix<tinytype, NU_PHYS, NU_PHYS> uu = u_phys * u_phys.transpose();
    
    int idx = NU_PHYS;
    
    // vec(xu^T) - 8 elements
    for (int j = 0; j < NU_PHYS; j++) {
        for (int i = 0; i < NX_PHYS; i++) {
            u_aug(idx++) = xu(i, j);
        }
    }
    
    // vec(ux^T) - 8 elements (CORRECTED: column-major vectorization of ux^T, which is 4x2)
    for (int j = 0; j < NU_PHYS; j++) {      // columns of (ux^T) are control dims
        for (int i = 0; i < NX_PHYS; i++) {  // rows of (ux^T) are state dims
            u_aug(idx++) = ux(j, i);         // equals (ux^T)(i, j)
        }
    }
    
    // vec(uu^T) - 4 elements
    for (int j = 0; j < NU_PHYS; j++) {
        for (int i = 0; i < NU_PHYS; i++) {
            u_aug(idx++) = uu(i, j);
        }
    }
}

extern "C"
{

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "🚀 OBSTACLE AVOIDANCE WITH SDP STATE LIFTING" << std::endl;
    std::cout << "========================================" << std::endl;
    
    TinySolver *solver;
    
    // Build augmented dynamics matrices
    Eigen::Matrix<tinytype, NX_AUG, NX_AUG> A_aug;
    Eigen::Matrix<tinytype, NX_AUG, NU_AUG> B_aug;
    build_augmented_A(A_aug);
    build_augmented_B(B_aug);
    
    // Zero dynamics offset for augmented system
    Eigen::Matrix<tinytype, NX_AUG, 1> f_aug;
    f_aug.setZero();
    
    // Cost matrices
    Eigen::Matrix<tinytype, NX_AUG, 1> Q_aug = Map<Matrix<tinytype, NX_AUG, 1>>(Q_aug_data);
    Eigen::Matrix<tinytype, NU_AUG, 1> R_aug = Map<Matrix<tinytype, NU_AUG, 1>>(R_aug_data);

    std::cout << "Problem setup:" << std::endl;
    std::cout << "- Physical states: " << NX_PHYS << " (px, py, vx, vy)" << std::endl;
    std::cout << "- Physical inputs: " << NU_PHYS << " (ax, ay)" << std::endl;
    std::cout << "- Augmented states: " << NX_AUG << " (x + vec(xx^T))" << std::endl;
    std::cout << "- Augmented inputs: " << NU_AUG << " (u + cross terms)" << std::endl;
    std::cout << "- Horizon: " << NHORIZON << std::endl;
    std::cout << "- Obstacle: [" << OBS_CENTER_X << ", " << OBS_CENTER_Y << "], r=" << OBS_RADIUS << std::endl;

    // Augmented box constraints - loose for quadratic terms, tight for physical states
    tiny_VectorNx x_min_one, x_max_one;
    x_min_one.setConstant(-1000.0);  // Very loose bounds for quadratic terms
    x_max_one.setConstant(1000.0);
    
    // Bounds on physical states - keep wide enough to include initial condition
    x_min_one(0) = -15.0;  // px_min (keep wide for initial condition)
    x_min_one(1) = -2.0;   // py_min (can tighten this)
    x_min_one(2) = -5.0;   // vx_min
    x_min_one(3) = -5.0;   // vy_min
    x_max_one(0) = 5.0;    // px_max (keep wide for trajectory)
    x_max_one(1) = 2.0;    // py_max (tightened for RLT effectiveness)
    x_max_one(2) = 5.0;    // vx_max
    x_max_one(3) = 5.0;    // vy_max
    
    tinyMatrix x_min = x_min_one.replicate(1, NHORIZON);
    tinyMatrix x_max = x_max_one.replicate(1, NHORIZON);
    
    // Augmented control constraints
    tiny_MatrixNuNhm1 u_min, u_max;
    u_min.setConstant(-1000.0);  // Very loose bounds
    u_max.setConstant(1000.0);
    
    // Tighter bounds on physical controls only
    for (int k = 0; k < NHORIZON-1; k++) {
        u_min(0, k) = -2.0;  // ax_min
        u_min(1, k) = -2.0;  // ay_min
        u_max(0, k) = 2.0;   // ax_max
        u_max(1, k) = 2.0;   // ay_max
    }

    std::cout << "\n🔧 Setting up TinyMPC solver..." << std::endl;
    
    tinytype rho_value = 1.0;
    
    // Set up problem with augmented dimensions
    int status = tiny_setup(&solver,
                            A_aug, B_aug, f_aug, Q_aug.asDiagonal(), R_aug.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON, 0);
    
    if (status != 0) {
        std::cout << "❌ TinyMPC setup failed with status: " << status << std::endl;
        return -1;
    }
    
    // Set bound constraints
    status = tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);
    
    if (status != 0) {
        std::cout << "❌ Setting bound constraints failed with status: " << status << std::endl;
        return -1;
    }
    
    // Add linear collision avoidance constraint
    Eigen::Matrix<tinytype, 1, NX_AUG> G_collision;
    tinytype h_collision;
    build_collision_constraint(G_collision, h_collision);
    
    std::cout << "🔧 Adding RLT/McCormick bounds to tighten X ≈ x*x^T..." << std::endl;
    
    // RLT/McCormick bounds: Add secant inequalities for diagonal terms
    // X_ii <= (l_i + u_i)*x_i - l_i*u_i  (secant upper bound)
    // Combined with M ⪰ 0 (which gives X_ii >= x_i^2), this tightly constrains X_ii ≈ x_i^2
    
    // State bounds for RLT constraints - balance tightness with feasibility
    tinytype px_min = -15.0, px_max = 5.0;   // position x bounds (keep wide for initial condition)
    tinytype py_min = -2.0,  py_max = 2.0;   // position y bounds (tightened for effectiveness)
    tinytype vx_min = -5.0,  vx_max = 5.0;   // velocity x bounds
    tinytype vy_min = -5.0,  vy_max = 5.0;   // velocity y bounds
    
    // Build RLT constraints (4 diagonal secant bounds)
    Eigen::Matrix<tinytype, 4, NX_AUG> A_rlt;
    Eigen::Matrix<tinytype, 4, 1> b_rlt;
    A_rlt.setZero();
    
    // X_00 <= (px_min + px_max)*px - px_min*px_max  (px^2 secant)
    A_rlt(0, 4) = 1.0;                    // +1 * X_00
    A_rlt(0, 0) = -(px_min + px_max);     // -(l_px + u_px) * px
    b_rlt(0) = -px_min * px_max;          // RHS: -l_px * u_px
    
    // X_11 <= (py_min + py_max)*py - py_min*py_max  (py^2 secant)  
    A_rlt(1, 9) = 1.0;                    // +1 * X_11
    A_rlt(1, 1) = -(py_min + py_max);     // -(l_py + u_py) * py
    b_rlt(1) = -py_min * py_max;          // RHS: -l_py * u_py
    
    // X_22 <= (vx_min + vx_max)*vx - vx_min*vx_max  (vx^2 secant)
    A_rlt(2, 14) = 1.0;                   // +1 * X_22
    A_rlt(2, 2) = -(vx_min + vx_max);     // -(l_vx + u_vx) * vx
    b_rlt(2) = -vx_min * vx_max;          // RHS: -l_vx * u_vx
    
    // X_33 <= (vy_min + vy_max)*vy - vy_min*vy_max  (vy^2 secant)
    A_rlt(3, 19) = 1.0;                   // +1 * X_33
    A_rlt(3, 3) = -(vy_min + vy_max);     // -(l_vy + u_vy) * vy
    b_rlt(3) = -vy_min * vy_max;          // RHS: -l_vy * u_vy
    
    // Add McCormick bounds for off-diagonal position terms (px*py)
    // X_01 (px*py) is at index 4 + 0*4 + 1 = 5
    // X_10 (py*px) is at index 4 + 1*4 + 0 = 8
    
    // McCormick envelope constraints for X_01 = px*py:
    // Lower bounds:
    // X_01 >= px_min*py + py_min*px - px_min*py_min
    // X_01 >= px_max*py + py_max*px - px_max*py_max
    // Upper bounds:
    // X_01 <= px_max*py + py_min*px - px_max*py_min
    // X_01 <= px_min*py + py_max*px - px_min*py_max
    
    Eigen::Matrix<tinytype, 4, NX_AUG> A_mccormick;
    Eigen::Matrix<tinytype, 4, 1> b_mccormick;
    A_mccormick.setZero();
    
    // McCormick lower bound 1: -X_01 + px_min*py + py_min*px <= px_min*py_min
    A_mccormick(0, 5) = -1.0;      // -X_01
    A_mccormick(0, 0) = py_min;    // py_min * px
    A_mccormick(0, 1) = px_min;    // px_min * py
    b_mccormick(0) = px_min * py_min;
    
    // McCormick lower bound 2: -X_01 + px_max*py + py_max*px <= px_max*py_max
    A_mccormick(1, 5) = -1.0;      // -X_01
    A_mccormick(1, 0) = py_max;    // py_max * px
    A_mccormick(1, 1) = px_max;    // px_max * py
    b_mccormick(1) = px_max * py_max;
    
    // McCormick upper bound 1: X_01 - px_max*py - py_min*px <= -px_max*py_min
    A_mccormick(2, 5) = 1.0;       // X_01
    A_mccormick(2, 0) = -py_min;   // -py_min * px
    A_mccormick(2, 1) = -px_max;   // -px_max * py
    b_mccormick(2) = -px_max * py_min;
    
    // McCormick upper bound 2: X_01 - px_min*py - py_max*px <= -px_min*py_max
    A_mccormick(3, 5) = 1.0;       // X_01
    A_mccormick(3, 0) = -py_max;   // -py_max * px
    A_mccormick(3, 1) = -px_min;   // -px_min * py
    b_mccormick(3) = -px_min * py_max;
    
    // Note: X_10 = X_01 due to symmetry, so we don't need separate constraints
    
    // Combine collision constraint + RLT bounds + McCormick bounds
    int total_constraints = 1 + 4 + 4;  // 1 collision + 4 RLT + 4 McCormick
    Eigen::Matrix<tinytype, Eigen::Dynamic, NX_AUG> A_combined(total_constraints, NX_AUG);
    Eigen::Matrix<tinytype, Eigen::Dynamic, 1> b_combined(total_constraints);
    
    // First row: collision constraint (negated for correct inequality direction)
    A_combined.row(0) = -G_collision;
    b_combined(0) = h_collision;
    
    // Next 4 rows: RLT secant bounds
    A_combined.block<4, NX_AUG>(1, 0) = A_rlt;
    b_combined.segment<4>(1) = b_rlt;
    
    // Next 4 rows: McCormick bounds
    A_combined.block<4, NX_AUG>(5, 0) = A_mccormick;
    b_combined.segment<4>(5) = b_mccormick;
    
    // Replicate across horizon
    tinyMatrix A_lin_x = A_combined.replicate(NHORIZON, 1);
    Eigen::Matrix<tinytype, Eigen::Dynamic, 1> b_lin_x = b_combined.replicate(NHORIZON, 1);
    
    // No input constraints - use empty matrices
    tinyMatrix A_lin_u(0, NU_AUG);
    Eigen::Matrix<tinytype, 0, 1> b_lin_u_empty;
    
    status = tiny_set_linear_constraints(solver, A_lin_x, b_lin_x, A_lin_u, b_lin_u_empty);
    
    std::cout << "✅ Added " << total_constraints << " constraints per timestep:" << std::endl;
    std::cout << "  - 1 collision avoidance constraint" << std::endl;
    std::cout << "  - 4 RLT secant bounds (X_ii ≤ (l_i+u_i)*x_i - l_i*u_i)" << std::endl;
    std::cout << "  - 4 McCormick bounds for X_01 (px*py cross-term)" << std::endl;
    std::cout << "🔧 Enhanced SDP projection: Joint state-control moment matrices [1;x;u;X;XU;UX;UU] ⪰ 0" << std::endl;
    
    if (status != 0) {
        std::cout << "❌ Setting linear constraints failed with status: " << status << std::endl;
        return -1;
    }
    
    // Configure solver settings
    solver->settings->max_iter = 200;
    solver->settings->abs_pri_tol = 1e-3;
    solver->settings->abs_dua_tol = 1e-3;
    solver->settings->check_termination = 1;
    
    std::cout << "✅ TinyMPC solver initialized successfully!" << std::endl;
    std::cout << "✅ Augmented dynamics matrices built with Kronecker products" << std::endl;
    std::cout << "✅ Linear collision avoidance constraint added" << std::endl;
    
    // Sanity check: Test constraint at key points
    std::cout << "\n🔍 Sanity checking collision constraint..." << std::endl;
    
    // Test at obstacle center [-5, 0]
    Eigen::Matrix<tinytype, NX_AUG, 1> test_center = construct_augmented_state(Eigen::Matrix<tinytype, NX_PHYS, 1>(-5.0, 0.0, 0.0, 0.0));
    tinytype constraint_center = G_collision.dot(test_center);
    std::cout << "  At obstacle center [-5, 0]: phi = " << constraint_center << ", h = " << h_collision;
    std::cout << " → " << (constraint_center >= h_collision ? "✅ SATISFIED" : "❌ VIOLATED") << std::endl;
    
    // Test far away [-10, 0]  
    Eigen::Matrix<tinytype, NX_AUG, 1> test_far = construct_augmented_state(Eigen::Matrix<tinytype, NX_PHYS, 1>(-10.0, 0.0, 0.0, 0.0));
    tinytype constraint_far = G_collision.dot(test_far);
    std::cout << "  Far from obstacle [-10, 0]: phi = " << constraint_far << ", h = " << h_collision;
    std::cout << " → " << (constraint_far >= h_collision ? "✅ SATISFIED" : "❌ VIOLATED") << std::endl;

    // Create workspace pointer for brevity
    TinyWorkspace *work = solver->work;

    // Physical initial and goal states
    Eigen::Matrix<tinytype, NX_PHYS, 1> x0_phys(-10.0, 0.1, 0.0, 0.0);  // Start
    Eigen::Matrix<tinytype, NX_PHYS, 1> xg_phys(GOAL_X, GOAL_Y, 0.0, 0.0);  // Goal

    // Convert to augmented states
    tiny_VectorNx x0_aug = construct_augmented_state(x0_phys);
    tiny_VectorNx xg_aug = construct_augmented_state(xg_phys);

    std::cout << "\n🎯 Initial physical state: [" << x0_phys.transpose().format(CleanFmt) << "]" << std::endl;
    std::cout << "🎯 Goal physical state: [" << xg_phys.transpose().format(CleanFmt) << "]" << std::endl;

    // Set reference trajectory - only terminal goal
    work->Xref.setZero();
    work->Xref.col(NHORIZON-1) = xg_aug;
    
    // Zero reference inputs
    work->Uref.setZero();

    std::cout << "\n🚀 Solving with augmented state SDP formulation..." << std::endl;
    
    // Initialize trajectory with straight line in physical space
    for (int k = 0; k < NHORIZON; k++) {
        tinytype alpha = static_cast<tinytype>(k) / (NHORIZON - 1);
        Eigen::Matrix<tinytype, NX_PHYS, 1> x_interp = (1.0 - alpha) * x0_phys + alpha * xg_phys;
        work->x.col(k) = construct_augmented_state(x_interp);
    }
    
    // Set initial condition AFTER initialization (to avoid overwriting)
    work->x.col(0) = x0_aug;
    
    // Solve the MPC problem
    auto start_time = std::chrono::high_resolution_clock::now();
    
    status = tiny_solve(solver);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "⏱️  Solve time: " << duration.count() << " ms" << std::endl;
    std::cout << "🔄 Iterations: " << work->iter << std::endl;
    
    if (status == 0) {
        std::cout << "✅ Problem solved successfully!" << std::endl;
    } else {
        std::cout << "⚠️  Solver status: " << status << std::endl;
    }
    
    // Debug: Check initial condition enforcement
    std::cout << "\n🔍 Initial condition check:" << std::endl;
    Eigen::Matrix<tinytype, NX_PHYS, 1> x0_primal = extract_physical_state(work->x.col(0));
    Eigen::Matrix<tinytype, NX_PHYS, 1> x0_solution = extract_physical_state(solver->solution->x.col(0));
    std::cout << "  Primal x[0]:    [" << x0_primal.transpose().format(CleanFmt) << "]" << std::endl;
    std::cout << "  Solution x[0]:  [" << x0_solution.transpose().format(CleanFmt) << "]" << std::endl;
    std::cout << "  Expected x[0]:  [" << x0_phys.transpose().format(CleanFmt) << "]" << std::endl;

    // Analyze results
    std::cout << "\n========================================" << std::endl;
    std::cout << "📊 TRAJECTORY ANALYSIS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Extract physical trajectory and check obstacle violations
    int violations = 0;
    tinytype min_distance = 1000.0;
    
    // Use solution variables (vnew) which are the consensus variables that satisfy constraints
    const auto& X_solution = solver->solution->x;  // This contains vnew after solve
    
    for (int k = 0; k < NHORIZON; k++) {
        // Extract physical state from solution (constraint-satisfying variables)
        Eigen::Matrix<tinytype, NX_PHYS, 1> x_phys = extract_physical_state(X_solution.col(k));
        
        tinytype px = x_phys(0);
        tinytype py = x_phys(1);
        tinytype dist = sqrt((px - OBS_CENTER_X)*(px - OBS_CENTER_X) + 
                           (py - OBS_CENTER_Y)*(py - OBS_CENTER_Y));
        
        min_distance = std::min(min_distance, dist);
        
        if (dist < OBS_RADIUS) {
            violations++;
        }
    }
    
    std::cout << "🛡️  Safety violations: " << violations << "/" << NHORIZON << " time steps" << std::endl;
    std::cout << "📏 Minimum distance to obstacle: " << min_distance << std::endl;
    std::cout << "✅ Safe trajectory: " << (violations == 0 ? "YES" : "NO") << std::endl;
    
    // Check moment matrix consistency (PSD property)
    std::cout << "\n🔍 Checking moment matrix consistency..." << std::endl;
    int psd_violations = 0;
    
    for (int k = 0; k < NHORIZON; k++) {
        // Extract physical state from solution variables
        Eigen::Matrix<tinytype, NX_PHYS, 1> x_phys = extract_physical_state(X_solution.col(k));
        
        // Extract quadratic terms from augmented solution state
        Eigen::Matrix<tinytype, 16, 1> vec_xx = X_solution.col(k).segment<16>(NX_PHYS);
        
        // Reconstruct xx^T matrix
        Eigen::Matrix<tinytype, NX_PHYS, NX_PHYS> xx_reconstructed;
        int idx = 0;
        for (int j = 0; j < NX_PHYS; j++) {
            for (int i = 0; i < NX_PHYS; i++) {
                xx_reconstructed(i, j) = vec_xx(idx++);
            }
        }
        
        // Compare with x*x^T
        Eigen::Matrix<tinytype, NX_PHYS, NX_PHYS> xx_true = x_phys * x_phys.transpose();
        tinytype consistency_error = (xx_reconstructed - xx_true).norm();
        
        if (consistency_error > 1e-2) {
            psd_violations++;
        }
    }
    
    std::cout << "🔗 Moment matrix consistency violations: " << psd_violations << "/" << NHORIZON << std::endl;
    
    // Check constraint feasibility using solution variables
    std::cout << "\n📐 Checking constraint feasibility on solution..." << std::endl;
    tinytype max_violation = 0.0;
    
    // Debug: Check the first few time steps in detail
    for (int k = 0; k < std::min(3, NHORIZON); k++) {
        tinytype constraint_val = (-G_collision).dot(X_solution.col(k));
        tinytype violation = std::max(0.0, constraint_val - h_collision);
        max_violation = std::max(max_violation, violation);
        
        // Extract physical position for manual verification
        Eigen::Matrix<tinytype, NX_PHYS, 1> x_phys = extract_physical_state(X_solution.col(k));
        tinytype px = x_phys(0), py = x_phys(1);
        
        // Manual constraint calculation
        tinytype manual_val = px*px + py*py + 10.0*px;  // px² + py² + 10*px
        tinytype manual_violation = std::max(0.0, -manual_val - h_collision);  // Note the sign!
        
        std::cout << "  t=" << k << ": pos=[" << px << ", " << py << "]" << std::endl;
        std::cout << "    Solver constraint: " << constraint_val << " vs " << h_collision 
                  << " → violation=" << violation << std::endl;
        std::cout << "    Manual constraint: " << manual_val << " vs " << -h_collision 
                  << " → violation=" << manual_violation << std::endl;
    }
    
    for (int k = 3; k < NHORIZON; k++) {
        tinytype constraint_val = (-G_collision).dot(X_solution.col(k));
        tinytype violation = std::max(0.0, constraint_val - h_collision);
        max_violation = std::max(max_violation, violation);
    }
    
    std::cout << "📊 Maximum constraint violation: " << max_violation;
    std::cout << " → " << (max_violation < 1e-6 ? "✅ FEASIBLE" : "⚠️ INFEASIBLE") << std::endl;
    
    // Save trajectory data (physical states only) using solution variables
    std::ofstream file("obstacle_avoidance_sdp_lifted_trajectory.csv");
    file << "# Obstacle Avoidance with SDP State Lifting (Solution Variables)\n";
    file << "# time, pos_x, pos_y, vel_x, vel_y, input_x, input_y\n";
    
    const auto& U_solution = solver->solution->u;  // znew after solve
    
    // Debug: Check matrix dimensions
    std::cout << "📊 Matrix dimensions for CSV:" << std::endl;
    std::cout << "  X_solution: " << X_solution.rows() << "x" << X_solution.cols() << std::endl;
    std::cout << "  U_solution: " << U_solution.rows() << "x" << U_solution.cols() << std::endl;
    std::cout << "  NHORIZON: " << NHORIZON << std::endl;
    
    for (int k = 0; k < NHORIZON; k++) {
        Eigen::Matrix<tinytype, NX_PHYS, 1> x_phys;
        
        if (k == 0) {
            // For initial condition, use primal variables (where initial condition is enforced)
            x_phys = extract_physical_state(work->x.col(k));
        } else {
            // For other time steps, use solution variables (consensus)
            x_phys = extract_physical_state(X_solution.col(k));
        }
        
        file << k << ", " << x_phys(0) << ", " << x_phys(1) << ", " 
             << x_phys(2) << ", " << x_phys(3);
        
        if (k < NHORIZON - 1) {
            // Extract physical controls from solution
            Eigen::Matrix<tinytype, NU_PHYS, 1> u_phys = U_solution.col(k).head<NU_PHYS>();
            file << ", " << u_phys(0) << ", " << u_phys(1);
        } else {
            file << ", 0, 0";
        }
        file << "\n";
    }
    file.close();
    
    std::cout << "📊 Physical trajectory saved to obstacle_avoidance_sdp_lifted_trajectory.csv" << std::endl;
    std::cout << "\n🎉 SDP state lifting obstacle avoidance complete!" << std::endl;
    std::cout << "🎯 Augmented state formulation successfully implemented!" << std::endl;

    return 0;
}

} /* extern "C" */
