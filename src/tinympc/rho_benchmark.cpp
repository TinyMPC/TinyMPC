#include "rho_benchmark.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#ifdef ARDUINO
#include <Arduino.h>
#else
// For non-Arduino platforms
uint32_t micros() {
    return 0; // Replace with appropriate timing function
}
#endif

void initialize_format_matrices(RhoAdapter* adapter, int nx, int nu, int N) {
    // Calculate dimensions
    int x_decision_size = nx * N + nu * (N-1);
    int constraint_rows = (nx + nu) * (N-1);
    
    // Pre-allocate matrices
    adapter->A_matrix = tinyMatrix::Zero(constraint_rows, x_decision_size);
    adapter->z_vector = tinyMatrix::Zero(constraint_rows, 1);
    adapter->y_vector = tinyMatrix::Zero(constraint_rows, 1);
    adapter->x_decision = tinyMatrix::Zero(x_decision_size, 1);
    
    // Pre-compute P matrix structure
    adapter->P_matrix = tinyMatrix::Zero(x_decision_size, x_decision_size);
    adapter->q_vector = tinyMatrix::Zero(x_decision_size, 1);
    
    // Pre-allocate residual computation matrices
    adapter->Ax_vector = tinyMatrix::Zero(constraint_rows, 1);
    adapter->r_prim_vector = tinyMatrix::Zero(constraint_rows, 1);
    adapter->r_dual_vector = tinyMatrix::Zero(x_decision_size, 1);
    adapter->Px_vector = tinyMatrix::Zero(x_decision_size, 1);
    adapter->ATy_vector = tinyMatrix::Zero(x_decision_size, 1);
    
    // Store dimensions
    adapter->format_nx = nx;
    adapter->format_nu = nu;
    adapter->format_N = N;
    
    adapter->matrices_initialized = true;
}

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
) {
    if (!adapter->matrices_initialized) {
        initialize_format_matrices(adapter, x_prev.rows(), u_prev.rows(), N);
    }
    
    int nx = adapter->format_nx;
    int nu = adapter->format_nu;
    
    // Fill x_decision
    int x_idx = 0;
    for (int i = 0; i < N; i++) {
        adapter->x_decision.block(x_idx, 0, nx, 1) = x_prev.col(i);
        x_idx += nx;
        if (i < N-1) {
            adapter->x_decision.block(x_idx, 0, nu, 1) = u_prev.col(i);
            x_idx += nu;
        }
    }
    
    // Clear A matrix for reuse
    adapter->A_matrix.setZero();
    
    // Fill A matrix with dynamics and input constraints
    for (int i = 0; i < N-1; i++) {
        // Input constraints
        int row_start = i * nu;
        int col_start = i * (nx+nu) + nx;
        adapter->A_matrix.block(row_start, col_start, nu, nu) = tinyMatrix::Identity(nu, nu);
        
        // Dynamics constraints
        row_start = (N-1) * nu + i * nx;
        col_start = i * (nx+nu);
        adapter->A_matrix.block(row_start, col_start, nx, nx) = work->Adyn;
        adapter->A_matrix.block(row_start, col_start+nx, nx, nu) = work->Bdyn;
        
        int next_state_idx = col_start + nx + nu;
        if (next_state_idx < adapter->A_matrix.cols()) {
            adapter->A_matrix.block(row_start, next_state_idx, nx, nx) = -tinyMatrix::Identity(nx, nx);
        }
    }
    
    // Fill z and y vectors
    for (int i = 0; i < N-1; i++) {
        adapter->z_vector.block(i*nu, 0, nu, 1) = z_prev.col(i);
        adapter->z_vector.block((N-1)*nu+i*nx, 0, nx, 1) = v_prev.col(i+1);
        
        adapter->y_vector.block(i*nu, 0, nu, 1) = y_prev.col(i);
        adapter->y_vector.block((N-1)*nu+i*nx, 0, nx, 1) = g_prev.col(i+1);
    }
    
    // Build P matrix (cost matrix)
    adapter->P_matrix.setZero();
    
    // Fill diagonal blocks
    x_idx = 0;
    for (int i = 0; i < N; i++) {
        // State cost
        if (i == N-1) {
            adapter->P_matrix.block(x_idx, x_idx, nx, nx) = cache->Pinf;
        } else {
            adapter->P_matrix.block(x_idx, x_idx, nx, nx) = work->Q.asDiagonal();
        }
        x_idx += nx;
        
        // Input cost
        if (i < N-1) {
            adapter->P_matrix.block(x_idx, x_idx, nu, nu) = work->R.asDiagonal();
            x_idx += nu;
        }
    }
    
    // Create q vector (linear cost vector)
    x_idx = 0;
    for (int i = 0; i < N; i++) {
        // For simplicity, we'll use zero reference for now
        // In a real implementation, you'd use your reference trajectory
        tinyMatrix x_ref = tinyMatrix::Zero(nx, 1);
        tinyMatrix delta_x = x_prev.col(i) - x_ref;
        adapter->q_vector.block(x_idx, 0, nx, 1) = work->Q.asDiagonal() * delta_x;
        x_idx += nx;
        
        if (i < N-1) {
            // For simplicity, we'll use zero reference for now
            tinyMatrix u_ref = tinyMatrix::Zero(nu, 1);
            tinyMatrix delta_u = u_prev.col(i) - u_ref;
            adapter->q_vector.block(x_idx, 0, nu, 1) = work->R.asDiagonal() * delta_u;
            x_idx += nu;
        }
    }
}

void compute_residuals(
    RhoAdapter* adapter,
    tinytype* pri_res,
    tinytype* dual_res,
    tinytype* pri_norm,
    tinytype* dual_norm
) {
    // Compute Ax
    adapter->Ax_vector = adapter->A_matrix * adapter->x_decision;
    
    // Compute primal residual
    adapter->r_prim_vector = adapter->Ax_vector - adapter->z_vector;
    *pri_res = adapter->r_prim_vector.cwiseAbs().maxCoeff();
    *pri_norm = std::max(adapter->Ax_vector.cwiseAbs().maxCoeff(), adapter->z_vector.cwiseAbs().maxCoeff());
    
    // Compute dual residual components
    adapter->Px_vector = adapter->P_matrix * adapter->x_decision;
    adapter->ATy_vector = adapter->A_matrix.transpose() * adapter->y_vector;
    
    // Compute full dual residual
    adapter->r_dual_vector = adapter->Px_vector + adapter->q_vector + adapter->ATy_vector;
    *dual_res = adapter->r_dual_vector.cwiseAbs().maxCoeff();
    
    // Compute normalization
    *dual_norm = std::max(std::max(adapter->Px_vector.cwiseAbs().maxCoeff(), 
                                  adapter->ATy_vector.cwiseAbs().maxCoeff()), 
                         adapter->q_vector.cwiseAbs().maxCoeff());
}

tinytype predict_rho(
    RhoAdapter* adapter,
    tinytype pri_res,
    tinytype dual_res,
    tinytype pri_norm,
    tinytype dual_norm,
    tinytype current_rho
) {
    const tinytype eps = 1e-10;
    
    tinytype normalized_pri = pri_res / (pri_norm + eps);
    tinytype normalized_dual = dual_res / (dual_norm + eps);
    
    tinytype ratio = normalized_pri / (normalized_dual + eps);
    
    tinytype new_rho = current_rho * std::sqrt(ratio);
    
    if (adapter->clip) {
        new_rho = std::min(std::max(new_rho, adapter->rho_min), adapter->rho_max);
    }
    
    return new_rho;
}

void update_matrices_with_derivatives(TinyCache* cache, tinytype new_rho) {
    tinytype delta_rho = new_rho - cache->rho;
    
    // //Direct updates only when new rho is 5 times greater or less than current rho
    // if (new_rho > 2 * cache->rho || new_rho < cache->rho / 2) {
        
    //     cache->Kinf += delta_rho * cache->dKinf_drho;
    //     cache->Pinf += delta_rho * cache->dPinf_drho;
    //     cache->C1 += delta_rho * cache->dC1_drho;
    //     cache->C2 += delta_rho * cache->dC2_drho;
    //     std::cout << "Rho updated: " << cache->rho << " -> " << new_rho 
    //           << " (delta: " << delta_rho << ")" << std::endl;
    // }

    cache->Kinf += delta_rho * cache->dKinf_drho;
    cache->Pinf += delta_rho * cache->dPinf_drho;
    cache->C1 += delta_rho * cache->dC1_drho;
    cache->C2 += delta_rho * cache->dC2_drho;

    cache->rho = new_rho;
    

}

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
) {
    uint32_t start_time = micros();
    
    // Format matrices
    format_matrices(adapter, x_prev, u_prev, v_prev, z_prev, g_prev, y_prev, cache, work, N);
    
    // Compute residuals
    tinytype pri_res, dual_res, pri_norm, dual_norm;
    compute_residuals(adapter, &pri_res, &dual_res, &pri_norm, &dual_norm);
    
    // Predict new rho
    tinytype new_rho = predict_rho(adapter, pri_res, dual_res, pri_norm, dual_norm, cache->rho);
    
    // Update matrices
   update_matrices_with_derivatives(cache, new_rho);
    
    // Store results
    result->time_us = micros() - start_time;
    result->initial_rho = cache->rho;
    result->final_rho = new_rho;
    result->pri_res = pri_res;
    result->dual_res = dual_res;
    result->pri_norm = pri_norm;
    result->dual_norm = dual_norm;
}