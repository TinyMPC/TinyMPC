#pragma once

#include <Eigen/Dense>
#include "constants.hpp"

using Eigen::Matrix;

#ifdef __cplusplus
extern "C" {
#endif

typedef float tinytype;

typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;
typedef Matrix<tinytype, NINPUTS, 1> tiny_VectorNu;
typedef Matrix<tinytype, NSTATE_CONSTRAINTS, 1> tiny_VectorNc;
typedef Matrix<tinytype, NSTATES, NSTATES> tiny_MatrixNxNx;
typedef Matrix<tinytype, NSTATES, NINPUTS> tiny_MatrixNxNu;
typedef Matrix<tinytype, NINPUTS, NSTATES> tiny_MatrixNuNx;
typedef Matrix<tinytype, NINPUTS, NINPUTS> tiny_MatrixNuNu;
typedef Matrix<tinytype, NSTATE_CONSTRAINTS, NSTATES> tiny_MatrixNcNx;

/**
 * Matrices that must be recomputed with changes in time step, rho, or model parameters
*/ 
struct tiny_cache {
    tiny_MatrixNxNx Adyn;
    tiny_MatrixNxNu Bdyn;
    tinytype rho;
    tiny_MatrixNuNx Kinf;
    tiny_MatrixNxNx Pinf;
    tiny_MatrixNuNu Quu_inv;
    tiny_MatrixNxNx AmBKt;
    tiny_MatrixNxNu coeff_d2p;
};

/**
 * Problem parameters
*/
struct tiny_params {
    tiny_MatrixNxNx Q;
    tiny_MatrixNxNx Qf;
    tiny_MatrixNuNu R;

    tiny_VectorNu u_min;
    tiny_VectorNu u_max;
    tiny_VectorNc x_min[NHORIZON];
    tiny_VectorNc x_max[NHORIZON];
    tiny_MatrixNcNx A_constraints[NHORIZON];

    tiny_VectorNx Xref[NHORIZON];
    tiny_VectorNu Uref[NHORIZON-1];

    struct tiny_cache cache;
};

/**
 * Problem variables
*/
struct tiny_problem {
    // State and input
    tiny_VectorNx x[NHORIZON];
    tiny_VectorNu u[NHORIZON-1];

    // Linear control cost terms
    tiny_VectorNx q[NHORIZON];
    tiny_VectorNu r[NHORIZON-1];

    // Linear Riccati backward pass terms
    tiny_VectorNx p[NHORIZON];
    tiny_VectorNu d[NHORIZON-1];

    // Auxiliary variables
    tiny_VectorNx v[NHORIZON];
    tiny_VectorNx vnew[NHORIZON];
    tiny_VectorNu z[NHORIZON-1];
    tiny_VectorNu znew[NHORIZON-1];

    // Dual variables
    tiny_VectorNx g[NHORIZON];
    tiny_VectorNu y[NHORIZON-1];

    tinytype primal_residual_state;
    tinytype primal_residual_input;
    tinytype dual_residual_state;
    tinytype dual_residual_input;
    tinytype abs_tol;
    int status;
    int iter;
    int max_iter;
    int iters_check_rho_update;
};

#ifdef __cplusplus
}
#endif
