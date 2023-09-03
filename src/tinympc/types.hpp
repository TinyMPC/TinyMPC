#pragma once

#include <Eigen.h>
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

// TODO: code review this since tiny_MatrixNuNhm1 naming is kind of gross
typedef Matrix<tinytype, NSTATES, NHORIZON, Eigen::ColMajor> tiny_MatrixNxNh;       // Nu x Nh
typedef Matrix<tinytype, NINPUTS, NHORIZON-1, Eigen::ColMajor> tiny_MatrixNuNhm1;   // Nu x Nh-1

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
    tiny_VectorNx Q;
    tiny_VectorNx Qf;
    tiny_VectorNu R;

    tiny_MatrixNuNhm1 u_min;
    tiny_MatrixNuNhm1 u_max;
    tiny_VectorNc x_min[NHORIZON];
    tiny_VectorNc x_max[NHORIZON];
    tiny_MatrixNcNx A_constraints[NHORIZON];

    // Turns out converting everything to big matrices is
    // slower than using for loops here - maybe this would
    // be different with fixed point
    // Only works with one constraint per knot point
    // but can be extended to multiple by making it
    // NHORIZON*NSTATE_CONSTRAINTS tall and keeping
    // track of indexing in the projection function
    // Matrix<tinytype, NHORIZON, NSTATE_CONSTRAINTS> x_min;
    // Matrix<tinytype, NHORIZON, NSTATE_CONSTRAINTS> x_max;
    // Matrix<tinytype, NHORIZON, NSTATES> A_constraints; 

    tiny_MatrixNxNh Xref;   // Nx x Nh
    tiny_MatrixNuNhm1 Uref; // Nu x Nh-1

    struct tiny_cache cache;
};

/**
 * Problem variables
*/
struct tiny_problem {
    // State and input
    tiny_MatrixNxNh x;
    tiny_MatrixNuNhm1 u;

    // Linear control cost terms
    tiny_MatrixNxNh q;
    tiny_MatrixNuNhm1 r;

    // Linear Riccati backward pass terms
    tiny_MatrixNxNh p;
    tiny_MatrixNuNhm1 d;

    // Auxiliary variables
    tiny_MatrixNxNh v;
    tiny_MatrixNxNh vnew;
    tiny_MatrixNuNhm1 z;
    tiny_MatrixNuNhm1 znew;

    // Dual variables
    tiny_MatrixNxNh g;
    tiny_MatrixNuNhm1 y;

    tinytype primal_residual_state;
    tinytype primal_residual_input;
    tinytype dual_residual_state;
    tinytype dual_residual_input;
    tinytype abs_tol;
    int status;
    int iter;
    int max_iter;
    int iters_check_rho_update;

    // Temporaries for algorithm efficiency
    tiny_MatrixNxNh xg;
    tinytype dist;
    Matrix<tinytype, NHORIZON, NSTATE_CONSTRAINTS> dists;
    Matrix<tinytype, 3, 1> xyz_new;
    tiny_VectorNu Qu;
    tiny_VectorNx Ax; // Stores result of sparse Adyn*x vector product computation

    // Helper variables
    int intersect;
};

#ifdef __cplusplus
}
#endif
