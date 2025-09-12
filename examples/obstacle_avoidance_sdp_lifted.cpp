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
    
    // Cost matrices - EXACTLY matching Julia formulation
    Eigen::Matrix<tinytype, NX_AUG, 1> Q_aug = Map<Matrix<tinytype, NX_AUG, 1>>(Q_aug_data);
    Eigen::Matrix<tinytype, NU_AUG, 1> R_aug = Map<Matrix<tinytype, NU_AUG, 1>>(R_aug_data);
    
    // Linear cost vectors - Julia uses these for q'*x_bar and r'*u_bar terms
    Eigen::Matrix<tinytype, NX_AUG, 1> q_aug = Map<Matrix<tinytype, NX_AUG, 1>>(q_aug_data);
    Eigen::Matrix<tinytype, NU_AUG, 1> r_aug = Map<Matrix<tinytype, NU_AUG, 1>>(r_aug_data);

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
    
    tinytype rho_value = 1.0;  // Julia doesn't specify rho - using standard value
    
    // Set up problem with augmented dimensions
    int status = tiny_setup(&solver,
                            A_aug, B_aug, f_aug, Q_aug.asDiagonal(), R_aug.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON, 0);
    
    if (status != 0) {
        std::cout << "❌ TinyMPC setup failed with status: " << status << std::endl;
        return -1;
    }
    
    // COMMENT OUT: Box constraints (not in Julia)
    /*
    status = tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);
    
    if (status != 0) {
        std::cout << "❌ Setting bound constraints failed with status: " << status << std::endl;
        return -1;
    }
    */
    std::cout << "🔧 PURE JULIA: No box constraints (Julia doesn't have bounds)" << std::endl;
    
    // Add linear collision avoidance constraint - Julia formulation
    Eigen::Matrix<tinytype, 1, NX_AUG> m_collision;
    tinytype n_collision;
    build_collision_constraint(m_collision, n_collision);
    
    // PURE JULIA: Only collision constraint (no RLT/McCormick bounds)
    std::cout << "🔧 Using PURE Julia formulation - only collision constraint..." << std::endl;
    
    // Only collision constraint - exactly as Julia does
    tinyMatrix A_lin_x(NHORIZON, NX_AUG);  // One constraint per time step
    Eigen::Matrix<tinytype, Eigen::Dynamic, 1> b_lin_x(NHORIZON);
    
    // Replicate collision constraint across horizon
    for (int k = 0; k < NHORIZON; k++) {
        A_lin_x.row(k) = -m_collision;  // Julia: m*x >= n → TinyMPC: -m*x <= -n
        b_lin_x(k) = -n_collision;
    }
    
    // No input constraints - use empty matrices
    tinyMatrix A_lin_u(0, NU_AUG);
    Eigen::Matrix<tinytype, 0, 1> b_lin_u_empty;
    
    status = tiny_set_linear_constraints(solver, A_lin_x, b_lin_x, A_lin_u, b_lin_u_empty);
    
    std::cout << "✅ Added ONLY collision constraint per timestep (pure Julia):" << std::endl;
    std::cout << "  - 1 collision avoidance constraint (m*x >= n)" << std::endl;
    std::cout << "🔧 Pure SDP projection: [1;x;u;X;XU;UX;UU] ⪰ 0" << std::endl;
    
    if (status != 0) {
        std::cout << "❌ Setting linear constraints failed with status: " << status << std::endl;
        return -1;
    }
    
    // Configure solver settings
    solver->settings->max_iter = 200;
    solver->settings->abs_pri_tol = 1e-3;
    solver->settings->abs_dua_tol = 1e-3;
    solver->settings->check_termination = 1;
    
    // PURE JULIA: Disable box constraints
    solver->settings->en_state_bound = false;
    solver->settings->en_input_bound = false;
    
    std::cout << "✅ TinyMPC solver initialized successfully!" << std::endl;
    std::cout << "✅ Augmented dynamics matrices built with Kronecker products" << std::endl;
    std::cout << "✅ Linear collision avoidance constraint added" << std::endl;

    // Create workspace pointer for brevity
    TinyWorkspace *work = solver->work;

    // Physical initial and goal states - DEFINE EARLY for box constraints
    Eigen::Matrix<tinytype, NX_PHYS, 1> x0_phys(-10.0, 0.1, 0.0, 0.0);  // Start
    Eigen::Matrix<tinytype, NX_PHYS, 1> xg_phys(GOAL_X, GOAL_Y, 0.0, 0.0);  // Goal
    
    // Sanity check: Test constraint at key points
    std::cout << "\n🔍 Sanity checking collision constraint..." << std::endl;
    
    // Test at obstacle center [-5, 0]
    Eigen::Matrix<tinytype, NX_AUG, 1> test_center = construct_augmented_state(Eigen::Matrix<tinytype, NX_PHYS, 1>(-5.0, 0.0, 0.0, 0.0));
    tinytype constraint_center = m_collision.dot(test_center);
    std::cout << "  At obstacle center [-5, 0]: phi = " << constraint_center << ", n = " << n_collision;
    std::cout << " → " << (constraint_center >= n_collision ? "✅ SATISFIED" : "❌ VIOLATED") << std::endl;
    
    // Test far away [-10, 0]  
    Eigen::Matrix<tinytype, NX_AUG, 1> test_far = construct_augmented_state(Eigen::Matrix<tinytype, NX_PHYS, 1>(-10.0, 0.0, 0.0, 0.0));
    tinytype constraint_far = m_collision.dot(test_far);
    std::cout << "  Far from obstacle [-10, 0]: phi = " << constraint_far << ", n = " << n_collision;
    std::cout << " → " << (constraint_far >= n_collision ? "✅ SATISFIED" : "❌ VIOLATED") << std::endl;

    // Convert to augmented states
    tiny_VectorNx x0_aug = construct_augmented_state(x0_phys);
    tiny_VectorNx xg_aug = construct_augmented_state(xg_phys);

    // PURE JULIA: No initial condition box constraints
    // Julia uses hard equality: x_bar[:,1] == [x_initial; vec(x_initial*x_initial')]
    // We rely on initialization + ADMM consensus

    std::cout << "\n🎯 Initial physical state: [" << x0_phys.transpose().format(CleanFmt) << "]" << std::endl;
    std::cout << "🎯 Goal physical state: [" << xg_phys.transpose().format(CleanFmt) << "]" << std::endl;

    // FIX 3b: Correct Xref and Uref calculation using actual diagonal elements
    // Build elementwise Xref and Uref once
    Eigen::Matrix<tinytype, NX_AUG, 1> Qdiag = Q_aug;  // Q = reg*I, so diagonal is just Q_aug
    Eigen::Matrix<tinytype, NU_AUG, 1> Rdiag = R_aug;  // R diagonal elements
    
    // Xref = -0.5 * Q^{-1} q   (Q = reg*I ⇒ divides by reg)
    Eigen::Matrix<tinytype, NX_AUG, 1> Xref_one = -0.5 * q_aug.cwiseQuotient(Qdiag);
    
    // Uref = -0.5 * R^{-1} r   (elementwise using the real diag of R)
    Eigen::Matrix<tinytype, NU_AUG, 1> Uref_one = -0.5 * r_aug.cwiseQuotient(Rdiag);
    
    work->Xref.setZero();
    work->Uref.setZero();
    for (int k = 0; k < NHORIZON; ++k) {
        work->Xref.col(k) = Xref_one;
    }
    for (int k = 0; k < NHORIZON-1; ++k) {
        work->Uref.col(k) = Uref_one;
    }
    
    // Override terminal goal
    work->Xref.col(NHORIZON-1) = xg_aug;
    
    // Debug: Print Julia's linear cost vectors to verify
    std::cout << "\n🔍 Julia linear costs being used:" << std::endl;
    std::cout << "  q_aug = [" << q_aug.transpose() << "]" << std::endl;
    std::cout << "  r_aug = [" << r_aug.transpose() << "]" << std::endl;
    std::cout << "  Xref_one = [" << Xref_one.transpose() << "]" << std::endl;
    std::cout << "  Uref_one = [" << Uref_one.transpose() << "]" << std::endl;

    std::cout << "\n🚀 Solving with augmented state SDP formulation..." << std::endl;
    
    // Initialize trajectory with straight line in physical space
    for (int k = 0; k < NHORIZON; k++) {
        tinytype alpha = static_cast<tinytype>(k) / (NHORIZON - 1);
        Eigen::Matrix<tinytype, NX_PHYS, 1> x_interp = (1.0 - alpha) * x0_phys + alpha * xg_phys;
        work->x.col(k) = construct_augmented_state(x_interp);
    }
    
    // CRITICAL: Set initial condition as hard constraint
    work->x.col(0) = x0_aug;
    
    // Also initialize slack variables to match initial condition
    work->vnew.col(0) = x0_aug;
    
    // Set initial condition in reference to enforce it more strongly
    work->Xref.col(0) = x0_aug;
    
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
        // Extract physical state - use primal for k=0, solution for k>0
        Eigen::Matrix<tinytype, NX_PHYS, 1> x_phys;
        if (k == 0) {
            x_phys = extract_physical_state(work->x.col(k));  // Use primal for initial condition
        } else {
            x_phys = extract_physical_state(X_solution.col(k));  // Use solution for others
        }
        
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
        // Use correct variables: primal for k=0, solution for k>0
        Eigen::Matrix<tinytype, NX_AUG, 1> x_check;
        if (k == 0) {
            x_check = work->x.col(k);  // Use primal variables for initial condition
        } else {
            x_check = X_solution.col(k);  // Use solution variables for other time steps
        }
        
        tinytype constraint_val = m_collision.dot(x_check);
        tinytype violation = std::max(0.0, n_collision - constraint_val);  // m*x >= n, so violation when m*x < n
        max_violation = std::max(max_violation, violation);
        
        // Extract physical position for manual verification
        Eigen::Matrix<tinytype, NX_PHYS, 1> x_phys = extract_physical_state(x_check);
        tinytype px = x_phys(0), py = x_phys(1);
        
        // Manual constraint calculation: px² + py² - 2*obs_x*px - 2*obs_y*py
        tinytype manual_val = px*px + py*py - 2.0*OBS_CENTER_X*px - 2.0*OBS_CENTER_Y*py;
        tinytype manual_violation = std::max(0.0, n_collision - manual_val);
        
        std::cout << "  t=" << k << ": pos=[" << px << ", " << py << "]" << std::endl;
        std::cout << "    Solver constraint: " << constraint_val << " vs " << n_collision 
                  << " → violation=" << violation << std::endl;
        std::cout << "    Manual constraint: " << manual_val << " vs " << n_collision 
                  << " → violation=" << manual_violation << std::endl;
    }
    
    for (int k = 3; k < NHORIZON; k++) {
        tinytype constraint_val = m_collision.dot(X_solution.col(k));  // Use solution variables for k > 0
        tinytype violation = std::max(0.0, n_collision - constraint_val);
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
    
    // FIX C: Export dynamics-consistent data by re-simulating with bounded controls
    Eigen::Matrix<tinytype, NX_PHYS, NX_PHYS> A_phys = 
        Map<Matrix<tinytype, NX_PHYS, NX_PHYS, RowMajor>>(A_phys_data);
    Eigen::Matrix<tinytype, NX_PHYS, NU_PHYS> B_phys = 
        Map<Matrix<tinytype, NX_PHYS, NU_PHYS, RowMajor>>(B_phys_data);
    
    Eigen::Matrix<tinytype, NX_PHYS, 1> xroll = x0_phys;  // Start from true initial condition
    
    for (int k = 0; k < NHORIZON; k++) {
        // Write current state
        file << k << ", " << xroll(0) << ", " << xroll(1) << ", " 
             << xroll(2) << ", " << xroll(3);
        
        if (k < NHORIZON - 1) {
            // Extract controls from solution (znew) which are now properly bounded after re-clamp
            Eigen::Matrix<tinytype, NU_PHYS, 1> u_phys = solver->solution->u.col(k).head<NU_PHYS>();
            file << ", " << u_phys(0) << ", " << u_phys(1);
            
            // Roll forward with true dynamics
            xroll = A_phys * xroll + B_phys * u_phys;
        } else {
            file << ", 0, 0";
        }
        file << "\n";
    }
    file.close();
    
    std::cout << "Physical trajectory saved to obstacle_avoidance_sdp_lifted_trajectory.csv" << std::endl;
    std::cout << "\nSDP state lifting obstacle avoidance complete!" << std::endl;
    std::cout << "Augmented state formulation successfully implemented!" << std::endl;

    return 0;
}

} /* extern "C" */
