// Quadrotor hover benchmark for adaptive rho (first-order adaptive caching,
// Mahajan & Plancher, arXiv:2507.03231).
//
// Runs the same closed-loop hover task three times and reports total ADMM
// iterations and tracking error:
//   1. fixed rho
//   2. adaptive rho, first-order Taylor cache update (the paper's method)
//   3. adaptive rho, exact cache recompute on every rho change (reference)

#define NSTATES 12
#define NINPUTS 4
#define NHORIZON 10
#define NSIM 100

#include <iostream>
#include <cstdio>
#include <tinympc/tiny_api.hpp>
#include "problem_data/quadrotor_20hz_params.hpp"

extern "C" {

typedef Matrix<tinytype, NINPUTS, NHORIZON-1> tiny_MatrixNuNhm1;
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;
typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;

struct RunResult {
    int total_iters;
    tinytype avg_err;
    tinytype final_err;
    tinytype final_rho;
};

static RunResult run(tinytype rho0, int adaptive, int recache)
{
    TinySolver *solver;

    tinyMatrix Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    tinyMatrix Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn_data);
    tinyVector fdyn = tiny_VectorNx::Zero();
    tinyVector Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
    tinyVector R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

    tiny_setup(&solver, Adyn, Bdyn, fdyn, Q.asDiagonal(), R.asDiagonal(),
               rho0, NSTATES, NINPUTS, NHORIZON, 0);
    tiny_set_bound_constraints(solver,
                               tiny_MatrixNxNh::Constant(-5), tiny_MatrixNxNh::Constant(5),
                               tiny_MatrixNuNhm1::Constant(-0.5), tiny_MatrixNuNhm1::Constant(0.5));
    solver->settings->max_iter = 100;

    // Adaptive rho: rho is kept in [0.7, 1.2] * rho0 by default (see
    // TINY_ADAPTIVE_RHO_*_SCALE); set adaptive_rho_min/max to override.
    // Sensitivities are computed automatically on the first solve.
    solver->settings->adaptive_rho = adaptive;
    solver->settings->adaptive_rho_recache = recache;

    TinyWorkspace *work = solver->work;

    tiny_VectorNx x0;
    x0 << 0, 1, 0, 0.2, 0, 0, 0.1, 0, 0, 0, 0, 0;
    tiny_VectorNx xref;
    xref << 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    work->Xref = xref.replicate<1, NHORIZON>();

    // Silence per-solve convergence messages
    std::streambuf *old_buf = std::cout.rdbuf(nullptr);

    RunResult res = {0, 0, 0, 0};
    for (int k = 0; k < NSIM; ++k) {
        res.avg_err += (x0 - xref).norm() / NSIM;
        tiny_set_x0(solver, x0);
        tiny_solve(solver);
        res.total_iters += solver->work->iter;
        x0 = work->Adyn * x0 + work->Bdyn * work->u.col(0);
    }

    std::cout.rdbuf(old_buf);
    res.final_err = (x0 - xref).norm();
    res.final_rho = solver->cache->rho;
    return res;
}

int main()
{
    const tinytype rho0 = 85.0;  // near-hover setting from the paper

    RunResult fixed   = run(rho0, 0, 0);
    RunResult taylor  = run(rho0, 1, 0);
    RunResult recache = run(rho0, 1, 1);

    printf("Quadrotor hover, %d MPC steps, initial rho = %.1f\n\n", NSIM, rho0);
    printf("%-28s %12s %12s %12s %10s\n", "method", "total iters", "avg error", "final error", "final rho");
    printf("%-28s %12d %12.4f %12.4f %10.2f\n", "fixed rho", fixed.total_iters, fixed.avg_err, fixed.final_err, fixed.final_rho);
    printf("%-28s %12d %12.4f %12.4f %10.2f\n", "adaptive (Taylor update)", taylor.total_iters, taylor.avg_err, taylor.final_err, taylor.final_rho);
    printf("%-28s %12d %12.4f %12.4f %10.2f\n", "adaptive (exact recompute)", recache.total_iters, recache.avg_err, recache.final_err, recache.final_rho);
    return 0;
}

} /* extern "C" */
