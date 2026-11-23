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

    // Linear constraint slack variables
    work->vl = tinyMatrix::Zero(nx, N);
    work->vlnew = tinyMatrix::Zero(nx, N);
    work->zl = tinyMatrix::Zero(nu, N-1);
    work->zlnew = tinyMatrix::Zero(nu, N-1);
    
    // Linear constraint dual variables
    work->gl = tinyMatrix::Zero(nx, N);
    work->yl = tinyMatrix::Zero(nu, N-1);

    // Time-varying Linear constraint slack variables
    work->vl_tv = tinyMatrix::Zero(nx, N);
    work->vlnew_tv = tinyMatrix::Zero(nx, N);
    work->zl_tv = tinyMatrix::Zero(nu, N-1);
    work->zlnew_tv = tinyMatrix::Zero(nu, N-1);

    // Time-varying Linear constraint dual variables
    work->gl_tv = tinyMatrix::Zero(nx, N);
    work->yl_tv = tinyMatrix::Zero(nu, N-1);

    work->Q = Q.diagonal(); // Original cost weights; rho is added once in tiny_precompute_and_set_cache
    work->R = R.diagonal();
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
    if (status) {
        return status;
    }

    // Initialize sensitivity matrices for adaptive rho
    if (solver->settings->adaptive_rho) {
        tiny_initialize_sensitivity_matrices(solver);
    }

    return 0;
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

    return 0;
}

int tiny_set_linear_constraints(TinySolver* solver,
                               tinyMatrix Alin_x, tinyVector blin_x,
                               tinyMatrix Alin_u, tinyVector blin_u) {
    if (!solver) {
        std::cout << "Error in tiny_set_linear_constraints: solver is nullptr" << std::endl;
        return 1;
    }

    // Make sure all linear constraint matrix sizes are self-consistent
    int num_state_linear = Alin_x.rows();
    int num_input_linear = Alin_u.rows();
    int status = 0;
    
    // Check state constraint dimensions
    if (num_state_linear > 0) {
        status |= check_dimension("State linear constraint matrix (Alin_x)", "rows", Alin_x.rows(), num_state_linear);
        status |= check_dimension("State linear constraint matrix (Alin_x)", "columns", Alin_x.cols(), solver->work->nx);
        status |= check_dimension("State linear constraint vector (blin_x)", "rows", blin_x.rows(), num_state_linear);
        status |= check_dimension("State linear constraint vector (blin_x)", "columns", blin_x.cols(), 1);
    }
    
    // Check input constraint dimensions
    if (num_input_linear > 0) {
        status |= check_dimension("Input linear constraint matrix (Alin_u)", "rows", Alin_u.rows(), num_input_linear);
        status |= check_dimension("Input linear constraint matrix (Alin_u)", "columns", Alin_u.cols(), solver->work->nu);
        status |= check_dimension("Input linear constraint vector (blin_u)", "rows", blin_u.rows(), num_input_linear);
        status |= check_dimension("Input linear constraint vector (blin_u)", "columns", blin_u.cols(), 1);
    }
    
    if (status) {
        return status;
    }

    solver->work->numStateLinear = num_state_linear;
    solver->work->numInputLinear = num_input_linear;

    solver->work->Alin_x = Alin_x;
    solver->work->blin_x = blin_x;
    solver->work->Alin_u = Alin_u;
    solver->work->blin_u = blin_u;

    return 0;
}

int tiny_set_tv_linear_constraints(TinySolver* solver,
                               tinyMatrix tv_Alin_x, tinyMatrix tv_blin_x,
                               tinyMatrix tv_Alin_u, tinyMatrix tv_blin_u) {
    if (!solver) {
        std::cout << "Error in tiny_set_linear_constraints: solver is nullptr" << std::endl;
        return 1;
    }

    // Make sure all linear constraint matrix sizes are self-consistent
    int num_tv_state_linear = tv_Alin_x.rows() / solver->work->N;
    int num_tv_input_linear = tv_Alin_u.rows() / (solver->work->N-1);
    int status = 0;
    
    // Check state constraint dimensions
    if (num_tv_state_linear > 0) {
        status |= check_dimension("State time-varying linear constraint matrix (tv_Alin_x)", "rows", 
                                                    tv_Alin_x.rows(), num_tv_state_linear * solver->work->N);
        status |= check_dimension("State time-varying linear constraint matrix (tv_Alin_x)", "columns", 
                                                    tv_Alin_x.cols(), solver->work->nx);
        status |= check_dimension("State time-varying linear constraint vector (tv_blin_x)", "rows", 
                                                    tv_blin_x.rows(), num_tv_state_linear);
        status |= check_dimension("State time-varying linear constraint vector (tv_blin_x)", "columns", 
                                                    tv_blin_x.cols(), solver->work->N);
    }
    
    // Check input constraint dimensions
    if (num_tv_input_linear > 0) {
        status |= check_dimension("Input time-varying linear constraint matrix (tv_Alin_u)", "rows", 
                                                tv_Alin_u.rows(), num_tv_input_linear * (solver->work->N-1));
        status |= check_dimension("Input time-varying linear constraint matrix (tv_Alin_u)", "columns", 
                                                tv_Alin_u.cols(), solver->work->nu);
        status |= check_dimension("Input time-varying linear constraint vector (tv_blin_u)", "rows", 
                                                tv_blin_u.rows(), num_tv_input_linear);
        status |= check_dimension("Input time-varying linear constraint vector (tv_blin_u)", "columns", 
                                                tv_blin_u.cols(), solver->work->N-1);
    }
    
    if (status) {
        return status;
    }

    solver->work->numtvStateLinear = num_tv_state_linear;
    solver->work->numtvInputLinear = num_tv_input_linear;

    solver->work->tv_Alin_x = tv_Alin_x;
    solver->work->tv_blin_x = tv_blin_x;
    solver->work->tv_Alin_u = tv_Alin_u;
    solver->work->tv_blin_u = tv_blin_u;

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


/**
 * Each enabled constraint family adds its own rho * ||x - slack + dual||^2 term to the
 * ADMM primal update (see update_linear_cost), and the bound-constraint term is always
 * present. The Riccati cache must therefore be computed for Q + m_x*rho*I and
 * R + m_u*rho*I. Recompute it if the enabled constraints changed since it was built.
 */
static void tiny_count_rho_terms(TinySolver* solver, int* mx, int* mu) {
    TinySettings *s = solver->settings;
    TinyWorkspace *w = solver->work;
    *mx = 1 + ((s->en_state_soc && w->numStateCones > 0) ? 1 : 0)
            + (s->en_state_linear ? 1 : 0) + (s->en_tv_state_linear ? 1 : 0);
    *mu = 1 + ((s->en_input_soc && w->numInputCones > 0) ? 1 : 0)
            + (s->en_input_linear ? 1 : 0) + (s->en_tv_input_linear ? 1 : 0);
}

int tiny_set_cache_rho(TinySolver* solver, tinytype rho) {
    TinyWorkspace *w = solver->work;
    TinyCache *c = solver->cache;
    int mx, mu;
    tiny_count_rho_terms(solver, &mx, &mu);

    int nx = w->nx;
    int nu = w->nu;
    // tiny_precompute_and_set_cache adds one rho itself
    tinyMatrix Qm = w->Q.asDiagonal();
    tinyMatrix Rm = w->R.asDiagonal();
    Qm += (mx - 1) * rho * tinyMatrix::Identity(nx, nx);
    Rm += (mu - 1) * rho * tinyMatrix::Identity(nu, nu);
    int status = tiny_precompute_and_set_cache(c, w->Adyn, w->Bdyn, w->fdyn, Qm, Rm, nx, nu, rho, 0);
    c->rho_mult_x = mx;
    c->rho_mult_u = mu;
    return status;
}

static int tiny_sync_cache_with_constraints(TinySolver* solver) {
    TinyCache *c = solver->cache;
    int mx, mu;
    tiny_count_rho_terms(solver, &mx, &mu);
    int cur_mx = c->rho_mult_x > 0 ? c->rho_mult_x : 1;
    int cur_mu = c->rho_mult_u > 0 ? c->rho_mult_u : 1;
    if (mx == cur_mx && mu == cur_mu) {
        return 0;
    }
    // Adaptive-rho sensitivities depend on the multipliers too; recompute lazily
    c->dKinf_drho.resize(0, 0);
    return tiny_set_cache_rho(solver, c->rho);
}

int tiny_solve(TinySolver* solver) {
    int status = tiny_sync_cache_with_constraints(solver);
    if (status) {
        return status;
    }
    return solve(solver);
}

int tiny_update_settings(TinySettings* settings, tinytype abs_pri_tol, tinytype abs_dua_tol,
                    int max_iter, int check_termination, 
                    int en_state_bound, int en_input_bound,
                    int en_state_soc, int en_input_soc,
                    int en_state_linear, int en_input_linear, 
                    int en_tv_state_linear, int en_tv_input_linear) {
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
    settings->en_state_soc = en_state_soc;
    settings->en_input_soc = en_input_soc;
    settings->en_state_linear = en_state_linear;
    settings->en_input_linear = en_input_linear;
    settings->en_tv_state_linear = en_tv_state_linear;
    settings->en_tv_input_linear = en_tv_input_linear;
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
    settings->en_state_linear = TINY_DEFAULT_EN_STATE_LINEAR;
    settings->en_input_linear = TINY_DEFAULT_EN_INPUT_LINEAR;
    settings->en_tv_state_linear = TINY_DEFAULT_EN_TV_STATE_LINEAR;
    settings->en_tv_input_linear = TINY_DEFAULT_EN_TV_INPUT_LINEAR;
    
    // Initialize adaptive rho settings
    settings->adaptive_rho = 0;  // Disabled by default
    settings->adaptive_rho_min = 0;  // <= 0: relative band around the base rho
    settings->adaptive_rho_max = 0;
    settings->adaptive_rho_enable_clipping = 1;
    settings->adaptive_rho_recache = 0;

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

// Riccati recursion for the ADMM-augmented cost (Q + mx*rho*I, R + mu*rho*I), run to
// tight convergence. Used to differentiate the cache with respect to rho.
static void tiny_compute_cache_terms(const tinyMatrix& Adyn, const tinyMatrix& Bdyn,
                                     const tinyMatrix& Q, const tinyMatrix& R, tinytype rho,
                                     int mx, int mu,
                                     tinyMatrix& Kinf, tinyMatrix& Pinf,
                                     tinyMatrix& Quu_inv, tinyMatrix& AmBKt) {
    int nx = Adyn.rows();
    int nu = Bdyn.cols();
    tinyMatrix Q1 = Q + mx * rho * tinyMatrix::Identity(nx, nx);
    tinyMatrix R1 = R + mu * rho * tinyMatrix::Identity(nu, nu);
    tinyMatrix P = rho * tinyMatrix::Identity(nx, nx);
    tinyMatrix K = tinyMatrix::Zero(nu, nx);
    for (int i = 0; i < 10000; i++) {
        tinyMatrix Kn = (R1 + Bdyn.transpose() * P * Bdyn).inverse() * Bdyn.transpose() * P * Adyn;
        tinyMatrix Pn = Q1 + Adyn.transpose() * P * (Adyn - Bdyn * Kn);
        tinytype dK = (Kn - K).cwiseAbs().maxCoeff();
        K = Kn;
        P = Pn;
        if (dK < 1e-12) break;
    }
    Kinf = K;
    Pinf = P;
    Quu_inv = (R1 + Bdyn.transpose() * P * Bdyn).inverse();
    AmBKt = (Adyn - Bdyn * K).transpose();
}

void tiny_initialize_sensitivity_matrices(TinySolver *solver) {
    // Compute d(cache)/d(rho) at the current rho by central finite differences of the
    // Riccati solution. Works for any system (previously hardcoded for one quadrotor).
    TinyWorkspace *work = solver->work;
    TinyCache *cache = solver->cache;
    tinyMatrix Q = work->Q.asDiagonal();
    tinyMatrix R = work->R.asDiagonal();
    tinytype rho = cache->rho;
    tinytype h = 1e-3 * (rho > 1 ? rho : 1);
    int mx, mu;
    tiny_count_rho_terms(solver, &mx, &mu);

    tinyMatrix Kp, Pp, Cp1, Cp2, Km, Pm, Cm1, Cm2;
    tiny_compute_cache_terms(work->Adyn, work->Bdyn, Q, R, rho + h, mx, mu, Kp, Pp, Cp1, Cp2);
    tiny_compute_cache_terms(work->Adyn, work->Bdyn, Q, R, rho - h, mx, mu, Km, Pm, Cm1, Cm2);
    cache->rho_base = rho;

    cache->dKinf_drho = (Kp - Km) / (2 * h);
    cache->dPinf_drho = (Pp - Pm) / (2 * h);
    cache->dC1_drho = (Cp1 - Cm1) / (2 * h);  // d(Quu_inv)/d(rho)
    cache->dC2_drho = (Cp2 - Cm2) / (2 * h);  // d(AmBKt)/d(rho)
}

#ifdef __cplusplus
}
#endif