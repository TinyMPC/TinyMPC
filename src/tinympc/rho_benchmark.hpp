#pragma once
#include <cstdint>
#include "types.hpp"

struct RhoAdapter {
    tinytype rho_min;
    tinytype rho_max;
    bool clip;
    bool matrices_initialized;
    
    // Pre-allocated matrices for formatting
    tinyMatrix A_matrix;
    tinyMatrix z_vector;
    tinyMatrix y_vector;
    tinyMatrix x_decision;
    tinyMatrix P_matrix;
    tinyMatrix q_vector;
    
    // Pre-allocated matrices for residual computation
    tinyMatrix Ax_vector;
    tinyMatrix r_prim_vector;
    tinyMatrix r_dual_vector;
    tinyMatrix Px_vector;
    tinyMatrix ATy_vector;
    
    // Dimensions
    int format_nx;
    int format_nu;
    int format_N;
};

struct RhoBenchmarkResult {
    uint32_t time_us;
    tinytype initial_rho;
    tinytype final_rho;
    tinytype pri_res;
    tinytype dual_res;
    tinytype pri_norm;
    tinytype dual_norm;
};

// Initialize matrices for formatting
void initialize_format_matrices(RhoAdapter* adapter, int nx, int nu, int N);

// Format matrices for residual computation
void format_matrices(
    RhoAdapter* adapter,
    const tinyMatrix& x_prev,
    const tinyMatrix& u_prev,
    const tinyMatrix& v_prev,
    const tinyMatrix& z_prev,
    const tinyMatrix& g_prev,
    const tinyMatrix& y_prev,
    TinyCache* cache,
    TinyWorkspace* work,
    int N
);

// Compute residuals
void compute_residuals(
    RhoAdapter* adapter,
    tinytype* pri_res,
    tinytype* dual_res,
    tinytype* pri_norm,
    tinytype* dual_norm
);

// Predict new rho value
tinytype predict_rho(
    RhoAdapter* adapter,
    tinytype pri_res,
    tinytype dual_res,
    tinytype pri_norm,
    tinytype dual_norm,
    tinytype current_rho
);

// Update matrices using derivatives
void update_matrices_with_derivatives(TinyCache* cache, tinytype new_rho);

// Main benchmark function
void benchmark_rho_adaptation(
    RhoAdapter* adapter,
    const tinyMatrix& x_prev,
    const tinyMatrix& u_prev,
    const tinyMatrix& v_prev,
    const tinyMatrix& z_prev,
    const tinyMatrix& g_prev,
    const tinyMatrix& y_prev,
    TinyCache* cache,
    TinyWorkspace* work,
    int N,
    RhoBenchmarkResult* result
);