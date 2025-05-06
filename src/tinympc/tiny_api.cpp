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
                tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix fdyn, tinyMatrix Q, tinyMatrix R, 
                tinytype rho, int nx, int nu, int N, int verbose) {

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
    status |= check_dimension("Affine vector (f)", "rows", fdyn.rows(), nx);
    status |= check_dimension("Affine vector (f)", "columns", fdyn.cols(), 1);
    status |= check_dimension("State stage cost (Q)", "rows",  Q.rows(), nx);
    status |= check_dimension("State stage cost (Q)", "columns",  Q.cols(), nx);
    status |= check_dimension("State input cost (R)", "rows",  R.rows(), nu);
    status |= check_dimension("State input cost (R)", "columns",  R.cols(), nu);
    if (status) {
        return status;
    }
    
    work->x = tinyMatrix::Zero(nx, N);
    work->u = tinyMatrix::Zero(nu, N-1);

    work->q = tinyMatrix::Zero(nx, N);
    work->r = tinyMatrix::Zero(nu, N-1);

    work->p = tinyMatrix::Zero(nx, N);
    work->d = tinyMatrix::Zero(nu, N-1);

    // Bound constraint slack variables
    work->v = tinyMatrix::Zero(nx, N);
    work->vnew = tinyMatrix::Zero(nx, N);
    work->z = tinyMatrix::Zero(nu, N-1);
    work->znew = tinyMatrix::Zero(nu, N-1);
    
    // Bound constraint dual variables
    work->g = tinyMatrix::Zero(nx, N);
    work->y = tinyMatrix::Zero(nu, N-1);
    
    // Cone constraint slack variables
    work->vc = tinyMatrix::Zero(nx, N);
    work->vcnew = tinyMatrix::Zero(nx, N);
    work->zc = tinyMatrix::Zero(nu, N-1);
    work->zcnew = tinyMatrix::Zero(nu, N-1);
    
    // Cone constraint dual variables
    work->gc = tinyMatrix::Zero(nx, N);
    work->yc = tinyMatrix::Zero(nu, N-1);

    work->Q = (Q + rho * tinyMatrix::Identity(nx, nx)).diagonal();
    work->R = (R + rho * tinyMatrix::Identity(nu, nu)).diagonal();
    work->Adyn = Adyn; // State transition matrix
    work->Bdyn = Bdyn; // Input matrix
    work->fdyn = fdyn; // Affine offset vector

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
    status = tiny_precompute_and_set_cache(cache, Adyn, Bdyn, fdyn, work->Q.asDiagonal(), work->R.asDiagonal(), nx, nu, rho, verbose);
    return status;
}

int tiny_set_bound_constraints(TinySolver* solver,
                    tinyMatrix x_min, tinyMatrix x_max,
                    tinyMatrix u_min, tinyMatrix u_max) {
    if (!solver) {
        std::cout << "Error in tiny_set_bound_constraints: solver is nullptr" << std::endl;
        return 1;
    }

    // Make sure all bound constraint matrix sizes are self-consistent
    int status = 0;
    status |= check_dimension("Lower state bounds (x_min)", "rows", x_min.rows(), solver->work->nx);
    status |= check_dimension("Lower state bounds (x_min)", "cols", x_min.cols(), solver->work->N);
    status |= check_dimension("Lower state bounds (x_max)", "rows", x_max.rows(), solver->work->nx);
    status |= check_dimension("Lower state bounds (x_max)", "cols", x_max.cols(), solver->work->N);
    status |= check_dimension("Lower input bounds (u_min)", "rows", u_min.rows(), solver->work->nu);
    status |= check_dimension("Lower input bounds (u_min)", "cols", u_min.cols(), solver->work->N-1);
    status |= check_dimension("Lower input bounds (u_max)", "rows", u_max.rows(), solver->work->nu);
    status |= check_dimension("Lower input bounds (u_max)", "cols", u_max.cols(), solver->work->N-1);

    solver->work->x_min = x_min;
    solver->work->x_max = x_max;
    solver->work->u_min = u_min;
    solver->work->u_max = u_max;

    // Enable constraints
    solver->settings->en_state_bound = 1;
    solver->settings->en_input_bound = 1;


    return 0;
}

int tiny_set_cone_constraints(TinySolver* solver,
                              VectorXi Acx, VectorXi qcx, tinyVector cx,
                              VectorXi Acu, VectorXi qcu, tinyVector cu) {
    if (!solver) {
        std::cout << "Error in tiny_set_cone_constraints: solver is nullptr" << std::endl;
        return 1;
    }

    // Make sure all cone constraint vector sizes are self-consistent
    int num_state_cones = Acx.rows();
    int num_input_cones = Acu.rows();
    int status = 0;
    status |= check_dimension("Cone state size (qcx)", "rows", qcx.rows(), num_state_cones);
    status |= check_dimension("Cone mu value for state (cx)", "rows", cx.rows(), num_state_cones);
    status |= check_dimension("Cone input size (qcu)", "rows", qcu.rows(), num_input_cones);
    status |= check_dimension("Cone mu value for input (cu)", "rows", cu.rows(), num_input_cones);
    if (status) {
        return status;
    }

    solver->work->numStateCones = num_state_cones;
    solver->work->numInputCones = num_input_cones;

    solver->work->Acx = Acx;
    solver->work->qcx = qcx;
    solver->work->cx = cx;
    
    solver->work->Acu = Acu;
    solver->work->qcu = qcu;
    solver->work->cu = cu;

    // Enable constraints
    solver->settings->en_state_soc = 1;
    solver->settings->en_input_soc = 1;

     // Initialize sensitivity matrices for adaptive rho
    if (solver->settings->adaptive_rho) {
        tiny_initialize_sensitivity_matrices(solver);
    }


    return 0;
}

int tiny_precompute_and_set_cache(TinyCache *cache,
                                  tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix fdyn, tinyMatrix Q, tinyMatrix R,
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

    // Precomputation for affine term
    tinyVector APf = AmBKt*Pinf*fdyn;
    tinyVector BPf = Bdyn.transpose()*Pinf*fdyn;

    if (verbose) {
        std::cout << "Kinf = " << Kinf.format(TinyApiFmt) << std::endl;
        std::cout << "Pinf = " << Pinf.format(TinyApiFmt) << std::endl;
        std::cout << "Quu_inv = " << Quu_inv.format(TinyApiFmt) << std::endl;
        std::cout << "AmBKt = " << AmBKt.format(TinyApiFmt) << std::endl;
        std::cout << "APf = " << APf.format(TinyApiFmt) << std::endl;
        std::cout << "BPf = " << BPf.format(TinyApiFmt) << std::endl;

        std::cout << "\nPrecomputation finished!\n" << std::endl;
    }

    cache->rho = rho;
    cache->Kinf = Kinf;
    cache->Pinf = Pinf;
    cache->Quu_inv = Quu_inv;
    cache->AmBKt = AmBKt;
    cache->C1 = Quu_inv;
    cache->C2 = AmBKt;
    cache->APf = APf;
    cache->BPf = BPf;

    return 0; // return success
}


int tiny_solve(TinySolver* solver) {
    return solve(solver);
}

int tiny_update_settings(TinySettings* settings, tinytype abs_pri_tol, tinytype abs_dua_tol,
                    int max_iter, int check_termination, 
                    int en_state_bound, int en_input_bound,
                    int en_state_soc, int en_input_soc) {
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

    // Turn off constraints until they are set by tiny_set_bound_constraints or tiny_set_cone_constraints
    settings->en_state_bound = TINY_DEFAULT_EN_STATE_BOUND;
    settings->en_input_bound = TINY_DEFAULT_EN_INPUT_BOUND;
    settings->en_state_soc = TINY_DEFAULT_EN_STATE_SOC;
    settings->en_input_soc = TINY_DEFAULT_EN_INPUT_SOC;
    
    // Initialize adaptive rho settings
    settings->adaptive_rho = 0;  // Disabled by default
    settings->adaptive_rho_min = 1.0;
    settings->adaptive_rho_max = 100.0;
    settings->adaptive_rho_enable_clipping = 1;

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

    const float dKinf_drho[4][12] = {
        {  0.0001,  -0.0001,  -0.0025,   0.0003,   0.0007,   0.0050,   0.0001,  -0.0001,  -0.0008,   0.0000,   0.0001,   0.0008},
        { -0.0001,  -0.0000,  -0.0025,  -0.0001,  -0.0006,  -0.0050,  -0.0001,   0.0000,  -0.0008,  -0.0000,  -0.0001,  -0.0008},
        {  0.0000,   0.0000,  -0.0025,   0.0001,   0.0004,   0.0050,   0.0000,   0.0000,  -0.0008,   0.0000,   0.0000,   0.0008},
        { -0.0000,   0.0001,  -0.0025,  -0.0003,  -0.0004,  -0.0050,  -0.0000,   0.0001,  -0.0008,  -0.0000,  -0.0000,  -0.0008}
    };

    const float dPinf_drho[12][12] = {
        {  0.0494,  -0.0045,  -0.0000,   0.0110,   0.1300,  -0.0283,   0.0280,  -0.0026,  -0.0000,   0.0004,   0.0070,  -0.0094},
        { -0.0045,   0.0491,   0.0000,  -0.1320,  -0.0111,   0.0114,  -0.0026,   0.0279,   0.0000,  -0.0076,  -0.0004,   0.0038},
        { -0.0000,   0.0000,   2.4450,   0.0000,  -0.0000,  -0.0000,  -0.0000,   0.0000,   1.2593,   0.0000,   0.0000,   0.0000},
        {  0.0110,  -0.1320,   0.0000,   0.3913,   0.0592,   0.3108,   0.0080,  -0.0776,   0.0000,   0.0254,   0.0068,   0.0750},
        {  0.1300,  -0.0111,  -0.0000,   0.0592,   0.4420,   0.7771,   0.0797,  -0.0081,  -0.0000,   0.0068,   0.0350,   0.1875},
        { -0.0283,   0.0114,  -0.0000,   0.3108,   0.7771,  10.0441,   0.0272,  -0.0109,   0.0000,   0.0655,   0.1639,   2.6362},
        {  0.0280,  -0.0026,  -0.0000,   0.0080,   0.0797,   0.0272,   0.0163,  -0.0016,  -0.0000,   0.0005,   0.0047,   0.0032},
        { -0.0026,   0.0279,   0.0000,  -0.0776,  -0.0081,  -0.0109,  -0.0016,   0.0161,   0.0000,  -0.0046,  -0.0005,  -0.0013},
        { -0.0000,   0.0000,   1.2593,   0.0000,  -0.0000,   0.0000,  -0.0000,   0.0000,   0.9232,   0.0000,   0.0000,   0.0000},
        {  0.0004,  -0.0076,   0.0000,   0.0254,   0.0068,   0.0655,   0.0005,  -0.0046,   0.0000,   0.0022,   0.0017,   0.0244},
        {  0.0070,  -0.0004,   0.0000,   0.0068,   0.0350,   0.1639,   0.0047,  -0.0005,   0.0000,   0.0017,   0.0054,   0.0610},
        { -0.0094,   0.0038,   0.0000,   0.0750,   0.1875,   2.6362,   0.0032,  -0.0013,   0.0000,   0.0244,   0.0610,   0.9869}
    };

    const float dC1_drho[4][4] = {
        { -0.0000,   0.0000,  -0.0000,   0.0000},
        {  0.0000,  -0.0000,   0.0000,  -0.0000},
        { -0.0000,   0.0000,  -0.0000,   0.0000},
        {  0.0000,  -0.0000,   0.0000,  -0.0000}
    };

    const float dC2_drho[12][12] = {
        {  0.0000,  -0.0000,   0.0000,   0.0000,   0.0000,  -0.0000,   0.0000,  -0.0000,   0.0000,   0.0000,   0.0000,  -0.0000},
        { -0.0000,   0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000,  -0.0000,   0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000},
        { -0.0000,   0.0000,   0.0001,   0.0000,  -0.0000,  -0.0000,  -0.0000,   0.0000,   0.0000,   0.0000,  -0.0000,  -0.0000},
        {  0.0000,  -0.0000,  -0.0000,   0.0001,   0.0000,  -0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000,   0.0000,  -0.0000},
        {  0.0000,  -0.0000,  -0.0000,   0.0000,   0.0001,  -0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000,   0.0000,  -0.0000},
        { -0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000,   0.0001,  -0.0000,   0.0000,  -0.0000,   0.0000,   0.0000,   0.0000},
        {  0.0000,  -0.0000,   0.0000,   0.0000,   0.0000,  -0.0000,   0.0000,  -0.0000,   0.0000,   0.0000,   0.0000,  -0.0000},
        { -0.0000,   0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000,  -0.0000,   0.0000,   0.0000,  -0.0000,  -0.0000,   0.0000},
        { -0.0000,   0.0000,   0.0021,   0.0000,  -0.0000,  -0.0000,  -0.0000,   0.0000,   0.0006,   0.0000,  -0.0000,  -0.0000},
        {  0.0002,  -0.0027,  -0.0000,   0.0068,   0.0005,  -0.0005,   0.0001,  -0.0015,  -0.0000,   0.0004,   0.0000,  -0.0001},
        {  0.0027,  -0.0002,   0.0000,   0.0005,   0.0066,  -0.0011,   0.0015,  -0.0001,   0.0000,   0.0000,   0.0004,  -0.0002},
        { -0.0001,   0.0001,   0.0000,  -0.0000,   0.0000,   0.0041,  -0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0006}
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