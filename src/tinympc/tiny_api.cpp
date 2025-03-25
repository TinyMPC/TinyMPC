#include "tiny_api.hpp"
#include "tiny_api_constants.hpp"

#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

using namespace Eigen;
IOFormat TinyApiFmt(4, 0, ", ", "\n", "[", "]");

static int check_dimension(std::string matrix_name, std::string rows_or_columns, int actual, int expected) {
    if (actual != expected) {
        std::cout << matrix_name << " has " << actual << " " << rows_or_columns << ". Expected " << expected << "." << std::endl;
        return 1;
    }
    return 0;
}

int tiny_setup(TinySolver** solverp,
                tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix Q, tinyMatrix R, 
                tinytype rho, int nx, int nu, int N,
                tinyMatrix x_min, tinyMatrix x_max, tinyMatrix u_min, tinyMatrix u_max,
                int verbose) {

    TinySolution *solution = new TinySolution();
    TinyCache *cache = new TinyCache();
    TinySettings *settings = new TinySettings();
    TinyWorkspace *work = new TinyWorkspace();
    TinySolver *solver = new TinySolver();

    solver->solution = solution;
    solver->cache = cache;
    solver->settings = settings;
    solver->work = work;

    *solverp = solver;

    // Initialize solution
    solution->iter = 0;
    solution->solved = 0;
    solution->x = tinyMatrix::Zero(nx, N);
    solution->u = tinyMatrix::Zero(nu, N-1);

    // Initialize settings
    tiny_set_default_settings(settings);

    // Initialize workspace
    work->nx = nx;
    work->nu = nu;
    work->N = N;

    // Make sure arguments are the correct shapes
    int status = 0;
    status |= check_dimension("State transition matrix (A)", "rows", Adyn.rows(), nx);
    status |= check_dimension("State transition matrix (A)", "columns", Adyn.cols(), nx);
    status |= check_dimension("Input matrix (B)", "rows",  Bdyn.rows(), nx);
    status |= check_dimension("Input matrix (B)", "columns",  Bdyn.cols(), nu);
    status |= check_dimension("State stage cost (Q)", "rows",  Q.rows(), nx);
    status |= check_dimension("State stage cost (Q)", "columns",  Q.cols(), nx);
    status |= check_dimension("State input cost (R)", "rows",  R.rows(), nu);
    status |= check_dimension("State input cost (R)", "columns",  R.cols(), nu);
    status |= check_dimension("Lower state bounds (x_min)", "rows", x_min.rows(), nx);
    status |= check_dimension("Lower state bounds (x_min)", "cols", x_min.cols(), N);
    status |= check_dimension("Lower state bounds (x_max)", "rows", x_max.rows(), nx);
    status |= check_dimension("Lower state bounds (x_max)", "cols", x_max.cols(), N);
    status |= check_dimension("Lower input bounds (u_min)", "rows", u_min.rows(), nu);
    status |= check_dimension("Lower input bounds (u_min)", "cols", u_min.cols(), N-1);
    status |= check_dimension("Lower input bounds (u_max)", "rows", u_max.rows(), nu);
    status |= check_dimension("Lower input bounds (u_max)", "cols", u_max.cols(), N-1);
    
    work->x = tinyMatrix::Zero(nx, N);
    work->u = tinyMatrix::Zero(nu, N-1);

    work->q = tinyMatrix::Zero(nx, N);
    work->r = tinyMatrix::Zero(nu, N-1);

    work->p = tinyMatrix::Zero(nx, N);
    work->d = tinyMatrix::Zero(nu, N-1);

    work->v = tinyMatrix::Zero(nx, N);
    work->vnew = tinyMatrix::Zero(nx, N);
    work->z = tinyMatrix::Zero(nu, N-1);
    work->znew = tinyMatrix::Zero(nu, N-1);
    
    work->g = tinyMatrix::Zero(nx, N);
    work->y = tinyMatrix::Zero(nu, N-1);

    work->Q = (Q + rho * tinyMatrix::Identity(nx, nx)).diagonal();
    work->R = (R + rho * tinyMatrix::Identity(nu, nu)).diagonal();
    work->Adyn = Adyn;
    work->Bdyn = Bdyn;

    work->x_min = x_min;
    work->x_max = x_max;
    work->u_min = u_min;
    work->u_max = u_max;

    work->Xref = tinyMatrix::Zero(nx, N);
    work->Uref = tinyMatrix::Zero(nu, N-1);

    work->Qu = tinyVector::Zero(nu);

    work->primal_residual_state = 0;
    work->primal_residual_input = 0;
    work->dual_residual_state = 0;
    work->dual_residual_input = 0;
    work->status = 0;
    work->iter = 0;

    // Initialize cache
    status = tiny_precompute_and_set_cache(cache, Adyn, Bdyn, work->Q.asDiagonal(), work->R.asDiagonal(), nx, nu, rho, verbose);
    if (status) {
        return status;
    }

    // // Initialize sensitivity matrices for adaptive rho
    // if (solver->settings->adaptive_rho) {
    //     tiny_initialize_sensitivity_matrices(solver);
    // }

    return 0;
}

int tiny_precompute_and_set_cache(TinyCache *cache,
                                  tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix Q, tinyMatrix R,
                                  int nx, int nu, tinytype rho, int verbose) {

    if (!cache) {
        std::cout << "Error in tiny_precompute_and_set_cache: cache is nullptr" << std::endl;
        return 1;
    }

    // Update by adding rho * identity matrix to Q, R
    tinyMatrix Q1 = Q + rho * tinyMatrix::Identity(nx, nx);
    tinyMatrix R1 = R + rho * tinyMatrix::Identity(nu, nu);

    // Printing
    if (verbose) {
        std::cout << "A = " << Adyn.format(TinyApiFmt) << std::endl;
        std::cout << "B = " << Bdyn.format(TinyApiFmt) << std::endl;
        std::cout << "Q = " << Q1.format(TinyApiFmt) << std::endl;
        std::cout << "R = " << R1.format(TinyApiFmt) << std::endl;
        std::cout << "rho = " << rho << std::endl;
    }

    // Riccati recursion to get Kinf, Pinf
    tinyMatrix Ktp1 = tinyMatrix::Zero(nu, nx);
    tinyMatrix Ptp1 = rho * tinyMatrix::Ones(nx, 1).array().matrix().asDiagonal();
    tinyMatrix Kinf = tinyMatrix::Zero(nu, nx);
    tinyMatrix Pinf = tinyMatrix::Zero(nx, nx);

    for (int i = 0; i < 1000; i++)
    {
        Kinf = (R1 + Bdyn.transpose() * Ptp1 * Bdyn).inverse() * Bdyn.transpose() * Ptp1 * Adyn;
        Pinf = Q1 + Adyn.transpose() * Ptp1 * (Adyn - Bdyn * Kinf);
        // if Kinf converges, break
        if ((Kinf - Ktp1).cwiseAbs().maxCoeff() < 1e-5)
        {
            if (verbose) {
                std::cout << "Kinf converged after " << i + 1 << " iterations" << std::endl;
            }
            break;
        }
        Ktp1 = Kinf;
        Ptp1 = Pinf;
    }

    // Compute cached matrices
    tinyMatrix Quu_inv = (R1 + Bdyn.transpose() * Pinf * Bdyn).inverse();
    tinyMatrix AmBKt = (Adyn - Bdyn * Kinf).transpose();

    if (verbose) {
        std::cout << "Kinf = " << Kinf.format(TinyApiFmt) << std::endl;
        std::cout << "Pinf = " << Pinf.format(TinyApiFmt) << std::endl;
        std::cout << "Quu_inv = " << Quu_inv.format(TinyApiFmt) << std::endl;
        std::cout << "AmBKt = " << AmBKt.format(TinyApiFmt) << std::endl;

        std::cout << "\nPrecomputation finished!\n" << std::endl;
    }

    cache->rho = rho;
    cache->Kinf = Kinf;
    cache->Pinf = Pinf;
    cache->Quu_inv = Quu_inv;
    cache->AmBKt = AmBKt;
    cache->C1 = Quu_inv;
    cache->C2 = AmBKt;
    
    // Compute sensitivity matrices
   compute_sensitivity_matrices(cache, Adyn, Bdyn, Q, R, nx, nu, rho, verbose);

    return 0; // return success
}

// New function to compute sensitivity matrices
void compute_sensitivity_matrices(TinyCache *cache,
                                 tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix Q, tinyMatrix R,
                                 int nx, int nu, tinytype rho, int verbose) {
    
    if (verbose) {
        std::cout << "Starting compute_sensitivity_matrices with dimensions:" << std::endl;
        std::cout << "nx = " << nx << ", nu = " << nu << std::endl;
        std::cout << "Adyn: " << Adyn.rows() << "x" << Adyn.cols() << std::endl;
        std::cout << "Bdyn: " << Bdyn.rows() << "x" << Bdyn.cols() << std::endl;
        std::cout << "Kinf: " << cache->Kinf.rows() << "x" << cache->Kinf.cols() << std::endl;
        std::cout << "Pinf: " << cache->Pinf.rows() << "x" << cache->Pinf.cols() << std::endl;
    }
    
    // Get cached values
    tinyMatrix Kinf = cache->Kinf;
    tinyMatrix Pinf = cache->Pinf;
    
    // Identity matrices
    tinyMatrix Inx = tinyMatrix::Identity(nx, nx);
    tinyMatrix Inu = tinyMatrix::Identity(nu, nu);
    
    // Compute intermediate matrices
    tinyMatrix BtP = Bdyn.transpose() * Pinf;
    tinyMatrix BtPB = BtP * Bdyn;
    tinyMatrix R_BtPB = R + rho * Inu + BtPB;
    tinyMatrix R_BtPB_inv = R_BtPB.inverse();
    tinyMatrix Acl = Adyn - Bdyn * Kinf;
    tinyMatrix Acl_t = Acl.transpose();
    
    // Compute dK/drho (derivative of Kinf with respect to rho)
    tinyMatrix temp1 = BtP * Bdyn;
    tinyMatrix temp2 = temp1 * R_BtPB_inv;
    tinyMatrix temp3 = Inu + temp2;
    tinyMatrix temp4 = R_BtPB_inv * temp3;
    tinyMatrix temp5 = BtP * Acl;
    tinyMatrix dK_drho = -temp4 * temp5;
    
    // Compute dP/drho using Lyapunov equation approach
    // The Lyapunov equation for dP/drho is:
    // dP/drho = Inx + Acl_t * (dP/drho) * Acl + dL/drho
    // where dL/drho contains terms related to dK/drho
    
    // First, compute dL/drho
    tinyMatrix dL_drho = Kinf.transpose() * Inu * Kinf;
    
    // Now solve the Lyapunov equation iteratively
    tinyMatrix dP_drho = Inx;  // Initial guess
    
    // Perform fixed-point iterations to solve the Lyapunov equation
    for (int i = 0; i < 500; i++) {
        dP_drho = dL_drho + Acl_t * dP_drho * Acl;
    }
    
    // Compute dC1/drho (derivative of Quu_inv)
    tinyMatrix dC1_drho = -R_BtPB_inv * temp3;
    
    // Compute dC2/drho (derivative of AmBKt)
    tinyMatrix dC2_drho = -(Bdyn * dK_drho).transpose();
    
    // Scale the matrices to avoid too large updates
    double scale_factor = 0.01;
    dK_drho *= scale_factor;
    dP_drho *= scale_factor;
    dC1_drho *= scale_factor;
    dC2_drho *= scale_factor;
    
    // Store the computed sensitivity matrices
    cache->dKinf_drho = dK_drho;
    cache->dPinf_drho = dP_drho;
    cache->dC1_drho = dC1_drho;
    cache->dC2_drho = dC2_drho;
    
    if (verbose) std::cout << "Sensitivity matrices computed successfully" << std::endl;
}

int tiny_solve(TinySolver* solver) {
    return solve(solver);
}

int tiny_update_settings(TinySettings* settings, tinytype abs_pri_tol, tinytype abs_dua_tol,
                    int max_iter, int check_termination, 
                    int en_state_bound, int en_input_bound) {
    if (!settings) {
        std::cout << "Error in tiny_update_settings: settings is nullptr" << std::endl;
        return 1;
    }
    settings->abs_pri_tol = abs_pri_tol;
    settings->abs_dua_tol = abs_dua_tol;
    settings->max_iter = max_iter;
    settings->check_termination = check_termination;
    settings->en_state_bound = en_state_bound;
    settings->en_input_bound = en_input_bound;
    return 0;
}

int tiny_set_default_settings(TinySettings* settings) {
    if (!settings) {
        std::cout << "Error in tiny_set_default_settings: settings is nullptr" << std::endl;
        return 1;
    }
    settings->abs_pri_tol = TINY_DEFAULT_ABS_PRI_TOL;
    settings->abs_dua_tol = TINY_DEFAULT_ABS_DUA_TOL;
    settings->max_iter = TINY_DEFAULT_MAX_ITER;
    settings->check_termination = TINY_DEFAULT_CHECK_TERMINATION;
    settings->en_state_bound = TINY_DEFAULT_EN_STATE_BOUND;
    settings->en_input_bound = TINY_DEFAULT_EN_INPUT_BOUND;
    
    // Default adaptive rho settings
    settings->adaptive_rho = 1;                // Disabled by default
    settings->adaptive_rho_min = 60.0;
    settings->adaptive_rho_max = 100.0;
    settings->adaptive_rho_enable_clipping = 0;
    
    return 0;
}

int tiny_set_x0(TinySolver* solver, tinyVector x0) {
    if (!solver) {
        std::cout << "Error in tiny_set_x0: solver is nullptr" << std::endl;
        return 1;
    }
    if (x0.rows() != solver->work->nx) {
        perror("Error in tiny_set_x0: x0 is not the correct length");
    }
    solver->work->x.col(0) = x0;
    return 0;
}

int tiny_set_x_ref(TinySolver* solver, tinyMatrix x_ref) {
    if (!solver) {
        std::cout << "Error in tiny_set_x_ref: solver is nullptr" << std::endl;
        return 1;
    }
    int status = 0;
    status |= check_dimension("State reference trajectory (x_ref)", "rows", x_ref.rows(), solver->work->nx);
    status |= check_dimension("State reference trajectory (x_ref)", "columns", x_ref.cols(), solver->work->N);
    solver->work->Xref = x_ref;
    return 0;
}

int tiny_set_u_ref(TinySolver* solver, tinyMatrix u_ref) {
    if (!solver) {
        std::cout << "Error in tiny_set_u_ref: solver is nullptr" << std::endl;
        return 1;
    }
    int status = 0;
    status |= check_dimension("Control/input reference trajectory (u_ref)", "rows", u_ref.rows(), solver->work->nu);
    status |= check_dimension("Control/input reference trajectory (u_ref)", "columns",u_ref.cols(), solver->work->N-1);
    solver->work->Uref = u_ref;
    return 0;
}

void tiny_initialize_sensitivity_matrices(TinySolver *solver) {

    int nu = solver->work->nu;
    int nx = solver->work->nx;
    // Initialize matrices with zeros
    solver->cache->dKinf_drho = tinyMatrix::Zero(nu, nx);
    solver->cache->dPinf_drho = tinyMatrix::Zero(nx, nx);
    solver->cache->dC1_drho = tinyMatrix::Zero(nu, nu);
    solver->cache->dC2_drho = tinyMatrix::Zero(nx, nx);

  
    // Pre-computed sensitivity matrices
    const float dKinf_drho[4][12] = {
        { 0.0001, -0.0000, -0.0016,  0.0002,  0.0005,  0.0033,  0.0001, -0.0000, -0.0009,  0.0000,  0.0001,  0.0010},
        {-0.0001,  0.0000, -0.0016, -0.0001, -0.0004, -0.0033, -0.0001,  0.0000, -0.0009, -0.0000, -0.0001, -0.0010},
        { 0.0000, -0.0000, -0.0016,  0.0001,  0.0004,  0.0033,  0.0000, -0.0000, -0.0009,  0.0000,  0.0001,  0.0010},
        {-0.0001,  0.0000, -0.0016, -0.0002, -0.0004, -0.0033, -0.0000,  0.0000, -0.0009, -0.0000, -0.0001, -0.0010}
    };

    const float dPinf_drho[12][12] = {
        { 0.0636, -0.0079, -0.0000,  0.0408,  0.2425,  0.4183,  0.0425, -0.0059, -0.0000,  0.0056,  0.0294,  0.1505},
        {-0.0079,  0.0589,  0.0000, -0.1954, -0.0409, -0.1666, -0.0059,  0.0378,  0.0000, -0.0217, -0.0056, -0.0600},
        { 0.0000,  0.0000,  9.0348,  0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  6.1357,  0.0000, -0.0000,  0.0000},
        { 0.0408, -0.1954, -0.0000,  0.7039,  0.3142,  1.8467,  0.0357, -0.1284, -0.0000,  0.0834,  0.0506,  0.7094},
        { 0.2425, -0.0409,  0.0000,  0.3142,  1.2380,  4.6235,  0.1788, -0.0358, -0.0000,  0.0507,  0.1764,  1.7752},
        { 0.4183, -0.1666,  0.0000,  1.8467,  4.6235, 34.2096,  0.4407, -0.1758,  0.0000,  0.3224,  0.8063, 12.9370},
        { 0.0425, -0.0059, -0.0000,  0.0357,  0.1788,  0.4407,  0.0293, -0.0046, -0.0000,  0.0053,  0.0231,  0.1643},
        {-0.0059,  0.0378,  0.0000, -0.1284, -0.0358, -0.1758, -0.0046,  0.0244,  0.0000, -0.0145, -0.0053, -0.0656},
        {-0.0000,  0.0000,  6.1357, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  4.2496, -0.0000,  0.0000,  0.0000},
        { 0.0056, -0.0217,  0.0000,  0.0834,  0.0507,  0.3224,  0.0053, -0.0145, -0.0000,  0.0109,  0.0086,  0.1258},
        { 0.0294, -0.0056,  0.0000,  0.0506,  0.1764,  0.8063,  0.0231, -0.0053,  0.0000,  0.0086,  0.0274,  0.3145},
        { 0.1505, -0.0600,  0.0000,  0.7094,  1.7752, 12.9370,  0.1643, -0.0656,  0.0000,  0.1258,  0.3145,  5.0369}
    };

    const float dC1_drho[4][4] = {
        {-0.0, -0.0, -0.0, -0.0},
        {-0.0, -0.0, -0.0, -0.0},
        {-0.0, -0.0, -0.0, -0.0},
        {-0.0, -0.0, -0.0, -0.0}
    };

    const float dC2_drho[12][12] = {
        { 0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000},
        {-0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000},
        { 0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000},
        { 0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000},
        { 0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000},
        { 0.0000, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000},
        { 0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000},
        {-0.0000,  0.0000,  0.0000, -0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000,  0.0000},
        { 0.0000,  0.0000,  0.0005, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0003,  0.0000,  0.0000, -0.0000},
        { 0.0000, -0.0002,  0.0000,  0.0008,  0.0001, -0.0001,  0.0000, -0.0002,  0.0000,  0.0001,  0.0000, -0.0000},
        { 0.0002, -0.0000,  0.0000,  0.0001,  0.0007, -0.0002,  0.0002, -0.0000, -0.0000,  0.0000,  0.0001, -0.0001},
        { 0.0000, -0.0000, -0.0000,  0.0000,  0.0001,  0.0011,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000,  0.0003}
    };

   
    // Map arrays to Eigen matrices
    solver->cache->dKinf_drho = Map<const Matrix<float, 4, 12>>(dKinf_drho[0]).cast<tinytype>();
    solver->cache->dPinf_drho = Map<const Matrix<float, 12, 12>>(dPinf_drho[0]).cast<tinytype>();
    solver->cache->dC1_drho = Map<const Matrix<float, 4, 4>>(dC1_drho[0]).cast<tinytype>();
    solver->cache->dC2_drho = Map<const Matrix<float, 12, 12>>(dC2_drho[0]).cast<tinytype>();
}

#ifdef __cplusplus
}
#endif