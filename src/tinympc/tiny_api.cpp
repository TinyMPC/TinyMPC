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
    status |= check_dimension("Affine vector (f)", "rows", fdyn.rows(), nx);
    status |= check_dimension("Affine vector (f)", "columns", fdyn.cols(), 1);
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
    work->Adyn = Adyn; // State transition matrix
    work->Bdyn = Bdyn; // Input matrix
    work->fdyn = fdyn; // Affine offset vector

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
    status = tiny_precompute_and_set_cache(cache, Adyn, Bdyn, fdyn, work->Q.asDiagonal(), work->R.asDiagonal(), nx, nu, rho, verbose);
    if (status) {
        return status;
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
    cache->APf = AmBKt*Pinf*fdyn; // Precomputation for affine term
    cache->BPf = Bdyn.transpose()*Pinf*fdyn; // Precomputation for affine term

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
    settings->en_state_bound = TINY_DEFAULT_EN_STATE_BOUND;
    settings->en_input_bound = TINY_DEFAULT_EN_INPUT_BOUND;
    settings->en_state_soc = TINY_DEFAULT_EN_STATE_SOC;
    settings->en_input_soc = TINY_DEFAULT_EN_INPUT_SOC;
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



#ifdef __cplusplus
}
#endif