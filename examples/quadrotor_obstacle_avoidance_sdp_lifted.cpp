// Quadrotor Obstacle Avoidance with SDP Constraints using State Lifting

#define NSTATES NX_AUG    // 156 (augmented states)
#define NINPUTS NU_AUG    // 116 (augmented controls)
#define NHORIZON 31
#define NTOTAL 50

#include <iostream>
#include <fstream>
#include <chrono>
#include <tinympc/tiny_api.hpp>
#include "problem_data/quadrotor_obstacle_avoidance_sdp_params.hpp"

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

extern "C"
{

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "QUADROTOR OBSTACLE AVOIDANCE WITH SDP STATE LIFTING" << std::endl;
    std::cout << "========================================" << std::endl;
    
    TinySolver *solver;
    
    // Initialize cost vectors
    initialize_costs();
    
    // Build augmented dynamics matrices (use dynamic allocation for large matrices)
    Eigen::MatrixXd A_aug(NX_AUG, NX_AUG);
    Eigen::MatrixXd B_aug(NX_AUG, NU_AUG);
    build_augmented_A(A_aug);
    build_augmented_B(B_aug);
    
    // Cast to tinytype
    tinyMatrix A_aug_tiny = A_aug.cast<tinytype>();
    tinyMatrix B_aug_tiny = B_aug.cast<tinytype>();
    
    // Zero dynamics offset for augmented system
    tinyVector f_aug = tinyVector::Zero(NX_AUG);
    
    // Cost matrices
    Eigen::Matrix<tinytype, NX_AUG, 1> Q_aug = Map<Matrix<tinytype, NX_AUG, 1>>(Q_aug_data);
    Eigen::Matrix<tinytype, NU_AUG, 1> R_aug = Map<Matrix<tinytype, NU_AUG, 1>>(R_aug_data);
    
    // Linear cost vectors
    Eigen::Matrix<tinytype, NX_AUG, 1> q_aug = Map<Matrix<tinytype, NX_AUG, 1>>(q_aug_data);
    Eigen::Matrix<tinytype, NU_AUG, 1> r_aug = Map<Matrix<tinytype, NU_AUG, 1>>(r_aug_data);

    std::cout << "Problem setup:" << std::endl;
    std::cout << "- Physical states: " << NX_PHYS << " [x,y,z,φ,θ,ψ,dx,dy,dz,dφ,dθ,dψ]" << std::endl;
    std::cout << "- Physical inputs: " << NU_PHYS << " [u1,u2,u3,u4]" << std::endl;
    std::cout << "- Augmented states: " << NX_AUG << " (x + vec(xx^T))" << std::endl;
    std::cout << "- Augmented inputs: " << NU_AUG << " (u + cross terms)" << std::endl;
    std::cout << "- Horizon: " << NHORIZON << std::endl;
    std::cout << "- Obstacle (XY only): [" << OBS_CENTER_X << ", " << OBS_CENTER_Y << "], r=" << OBS_RADIUS << std::endl;

    // Augmented box constraints
    tiny_VectorNx x_min_one, x_max_one;
    x_min_one.setConstant(-1000.0);  // Very loose bounds for quadratic terms
    x_max_one.setConstant(1000.0);
    
    // Bounds on physical states
    x_min_one(0) = -15.0;  // x_min
    x_min_one(1) = -2.0;   // y_min
    x_min_one(2) = 0.0;    // z_min (ground)
    x_max_one(0) = 5.0;    // x_max
    x_max_one(1) = 2.0;    // y_max
    x_max_one(2) = 3.0;    // z_max
    // Leave angles and angular velocities with loose bounds
    
    tinyMatrix x_min = x_min_one.replicate(1, NHORIZON);
    tinyMatrix x_max = x_max_one.replicate(1, NHORIZON);
    
    // Augmented control constraints
    tiny_MatrixNuNhm1 u_min, u_max;
    u_min.setConstant(-1000.0);  // Very loose bounds
    u_max.setConstant(1000.0);
    
    // Tighter bounds on physical controls only
    for (int k = 0; k < NHORIZON-1; k++) {
        u_min(0, k) = -0.4f;
        u_min(1, k) = -0.4f;
        u_min(2, k) = -0.4f;
        u_min(3, k) = -0.4f;
        u_max(0, k) = 0.4f;
        u_max(1, k) = 0.4f;
        u_max(2, k) = 0.4f;
        u_max(3, k) = 0.4f;
    }

    std::cout << "\n🔧 Setting up TinyMPC solver..." << std::endl;
    
    tinytype rho_value = 100.0;  // Higher rho for quadrotor
    
    // Set up problem with augmented dimensions
    int status = tiny_setup(&solver,
                            A_aug_tiny, B_aug_tiny, f_aug, Q_aug.asDiagonal(), R_aug.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON, 0);
    
    if (status != 0) {
        std::cout << "❌ TinyMPC setup failed with status: " << status << std::endl;
        return -1;
    }
    
    // Set box constraints
    status = tiny_set_bound_constraints(solver, x_min, x_max, u_min, u_max);
    
    if (status != 0) {
        std::cout << "❌ Setting bound constraints failed with status: " << status << std::endl;
        return -1;
    }
    
    // Add linear collision avoidance constraint
    Eigen::Matrix<tinytype, 1, NX_AUG> m_collision;
    tinytype n_collision;
    build_collision_constraint(m_collision, n_collision);
    
    std::cout << "🔧 Adding collision constraint..." << std::endl;
    
    // Collision constraint across horizon
    tinyMatrix A_lin_x(NHORIZON, NX_AUG);
    Eigen::Matrix<tinytype, Eigen::Dynamic, 1> b_lin_x(NHORIZON);
    
    for (int k = 0; k < NHORIZON; k++) {
        A_lin_x.row(k) = -m_collision;  // m*x >= n → -m*x <= -n
        b_lin_x(k) = -n_collision;
    }
    
    // No input constraints
    tinyMatrix A_lin_u(0, NU_AUG);
    Eigen::Matrix<tinytype, 0, 1> b_lin_u_empty;
    
    status = tiny_set_linear_constraints(solver, A_lin_x, b_lin_x, A_lin_u, b_lin_u_empty);
    
    std::cout << "✅ Added collision constraint (XY plane obstacle avoidance)" << std::endl;
    
    if (status != 0) {
        std::cout << "❌ Setting linear constraints failed with status: " << status << std::endl;
        return -1;
    }
    
    // Configure solver settings
    solver->settings->max_iter = 1000;  // More iterations for larger problem
    solver->settings->abs_pri_tol = 1e-3;
    solver->settings->abs_dua_tol = 1e-3;
    solver->settings->check_termination = 1;
    
    // Enable box and SDP constraints
    solver->settings->en_state_bound = true;
    solver->settings->en_input_bound = true;
    solver->settings->en_state_sdp = true;
    solver->settings->en_input_sdp = true;
    solver->settings->en_state_linear = true;
    
    // Forward blend parameter
    solver->settings->forward_blend_alpha = 0.8;
    
    std::cout << "✅ TinyMPC solver initialized successfully!" << std::endl;
    std::cout << "✅ SDP projections enabled for moment matrix constraints" << std::endl;

    // Create workspace pointer
    TinyWorkspace *work = solver->work;

    // Physical initial and goal states
    Eigen::Matrix<tinytype, NX_PHYS, 1> x0_phys;
    x0_phys << -10.0f, 0.1f, 1.0f,  // position: start at [-10, 0.1, 1]
               0.0f, 0.0f, 0.0f,     // angles: zeros
               0.0f, 0.0f, 0.0f,     // velocities: zeros
               0.0f, 0.0f, 0.0f;     // angular velocities: zeros
    
    Eigen::Matrix<tinytype, NX_PHYS, 1> xg_phys;
    xg_phys << GOAL_X, GOAL_Y, GOAL_Z,  // position: goal at [0, 0, 1]
               0.0f, 0.0f, 0.0f,          // angles: zeros
               0.0f, 0.0f, 0.0f,          // velocities: zeros
               0.0f, 0.0f, 0.0f;          // angular velocities: zeros
    
    // Sanity check: Test constraint at key points
    std::cout << "\n🔍 Sanity checking collision constraint..." << std::endl;
    
    Eigen::Matrix<tinytype, NX_AUG, 1> test_center = construct_augmented_state(
        (Eigen::Matrix<tinytype, NX_PHYS, 1>() << -5.0f, 0.0f, 1.0f, 0,0,0,0,0,0,0,0,0).finished());
    tinytype constraint_center = m_collision.dot(test_center);
    std::cout << "  At obstacle center [-5, 0, 1]: phi = " << constraint_center << ", n = " << n_collision;
    std::cout << " → " << (constraint_center >= n_collision ? "✅ SATISFIED" : "❌ VIOLATED") << std::endl;
    
    Eigen::Matrix<tinytype, NX_AUG, 1> test_far = construct_augmented_state(
        (Eigen::Matrix<tinytype, NX_PHYS, 1>() << -10.0f, 0.0f, 1.0f, 0,0,0,0,0,0,0,0,0).finished());
    tinytype constraint_far = m_collision.dot(test_far);
    std::cout << "  Far from obstacle [-10, 0, 1]: phi = " << constraint_far << ", n = " << n_collision;
    std::cout << " → " << (constraint_far >= n_collision ? "✅ SATISFIED" : "❌ VIOLATED") << std::endl;

    // Convert to augmented states
    tiny_VectorNx x0_aug = construct_augmented_state(x0_phys);
    tiny_VectorNx xg_aug = construct_augmented_state(xg_phys);

    std::cout << "\n🎯 Initial physical state: [" << x0_phys.head<6>().transpose().format(CleanFmt) << ", ...]" << std::endl;
    std::cout << "🎯 Goal physical state: [" << xg_phys.head<6>().transpose().format(CleanFmt) << ", ...]" << std::endl;

    // Build Xref and Uref
    Eigen::Matrix<tinytype, NX_AUG, 1> Qdiag = Q_aug;
    Eigen::Matrix<tinytype, NU_AUG, 1> Rdiag = R_aug;
    
    Eigen::Matrix<tinytype, NX_AUG, 1> Xref_one = -0.5 * q_aug.cwiseQuotient(Qdiag);
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

    std::cout << "\n🚀 Solving with augmented state SDP formulation..." << std::endl;
    
    // Initialize trajectory with straight line in physical space
    for (int k = 0; k < NHORIZON; k++) {
        tinytype alpha = static_cast<tinytype>(k) / (NHORIZON - 1);
        Eigen::Matrix<tinytype, NX_PHYS, 1> x_interp = (1.0 - alpha) * x0_phys + alpha * xg_phys;
        work->x.col(k) = construct_augmented_state(x_interp);
    }
    
    // Set initial condition
    work->x.col(0) = x0_aug;
    work->vnew.col(0) = x0_aug;
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
    
    // Analyze results
    std::cout << "\n========================================" << std::endl;
    std::cout << "📊 TRAJECTORY ANALYSIS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Extract physical trajectory and check obstacle violations (XY plane only)
    int violations = 0;
    tinytype min_distance = 1000.0;
    
    const auto& X_solution = solver->solution->x;
    
    for (int k = 0; k < NHORIZON; k++) {
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
    
    // Check moment matrix consistency
    std::cout << "\n🔍 Checking moment matrix consistency..." << std::endl;
    int psd_violations = 0;
    
    for (int k = 0; k < NHORIZON; k++) {
        Eigen::Matrix<tinytype, NX_PHYS, 1> x_phys = extract_physical_state(X_solution.col(k));
        Eigen::Matrix<tinytype, NX_PHYS*NX_PHYS, 1> vec_xx = X_solution.col(k).segment<NX_PHYS*NX_PHYS>(NX_PHYS);
        
        Eigen::Matrix<tinytype, NX_PHYS, NX_PHYS> xx_reconstructed;
        int idx = 0;
        for (int j = 0; j < NX_PHYS; j++) {
            for (int i = 0; i < NX_PHYS; i++) {
                xx_reconstructed(i, j) = vec_xx(idx++);
            }
        }
        
        Eigen::Matrix<tinytype, NX_PHYS, NX_PHYS> xx_true = x_phys * x_phys.transpose();
        tinytype consistency_error = (xx_reconstructed - xx_true).norm();
        
        if (consistency_error > 1e-2) {
            psd_violations++;
        }
    }
    
    std::cout << "🔗 Moment matrix consistency violations: " << psd_violations << "/" << NHORIZON << std::endl;
    
    // Save trajectory data (forward-simulated with true dynamics)
    std::ofstream file("quadrotor_obstacle_avoidance_sdp_lifted.csv");
    file << "# Quadrotor Obstacle Avoidance with SDP State Lifting\n";
    file << "# time, x, y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi, u1, u2, u3, u4\n";
    
    const auto& U_solution = solver->solution->u;
    
    Eigen::Matrix<tinytype, NX_PHYS, NX_PHYS> A_phys = 
        Map<Matrix<tinytype, NX_PHYS, NX_PHYS, RowMajor>>(A_phys_data);
    Eigen::Matrix<tinytype, NX_PHYS, NU_PHYS> B_phys = 
        Map<Matrix<tinytype, NX_PHYS, NU_PHYS, RowMajor>>(B_phys_data);
    
    Eigen::Matrix<tinytype, NX_PHYS, 1> xroll = x0_phys;
    
    for (int k = 0; k < NHORIZON; k++) {
        // Write current state
        file << k;
        for (int i = 0; i < NX_PHYS; i++) {
            file << ", " << xroll(i);
        }
        
        if (k < NHORIZON - 1) {
            // Extract controls
            Eigen::Matrix<tinytype, NU_PHYS, 1> u_phys = U_solution.col(k).head<NU_PHYS>();
            for (int i = 0; i < NU_PHYS; i++) {
                file << ", " << u_phys(i);
            }
            
            // Roll forward with true dynamics
            xroll = A_phys * xroll + B_phys * u_phys;
        } else {
            file << ", 0, 0, 0, 0";
        }
        file << "\n";
    }
    file.close();
    
    std::cout << "\n📊 Trajectory saved to quadrotor_obstacle_avoidance_sdp_lifted.csv" << std::endl;
    std::cout << "✅ SDP state lifting obstacle avoidance complete!" << std::endl;

    return 0;
}

} /* extern "C" */

