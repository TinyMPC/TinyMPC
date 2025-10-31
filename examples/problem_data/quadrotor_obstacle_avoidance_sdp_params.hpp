#pragma once

#include <tinympc/types.hpp>
#include <Eigen/Core>

using Eigen::Map;
using Eigen::Matrix;
using Eigen::RowMajor;

// Kronecker product implementation
template<typename Derived1, typename Derived2>
Eigen::Matrix<typename Derived1::Scalar, Eigen::Dynamic, Eigen::Dynamic>
kroneckerProduct(const Eigen::MatrixBase<Derived1>& A, const Eigen::MatrixBase<Derived2>& B) {
    int rows_A = A.rows(), cols_A = A.cols();
    int rows_B = B.rows(), cols_B = B.cols();
    
    Eigen::Matrix<typename Derived1::Scalar, Eigen::Dynamic, Eigen::Dynamic> result(rows_A * rows_B, cols_A * cols_B);
    
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_A; ++j) {
            result.block(i * rows_B, j * cols_B, rows_B, cols_B) = A(i, j) * B;
        }
    }
    
    return result;
}

// Physical dimensions
#define NX_PHYS 12  // Quadrotor states: [x, y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi]
#define NU_PHYS 4   // Quadrotor controls: [u1, u2, u3, u4]

// Augmented dimensions
#define NX_AUG (NX_PHYS + NX_PHYS*NX_PHYS)  // 12 + 144 = 156
#define NU_AUG (NU_PHYS + NX_PHYS*NU_PHYS + NU_PHYS*NX_PHYS + NU_PHYS*NU_PHYS)  // 4 + 48 + 48 + 16 = 116

// Cost weights (matching Julia/double integrator pattern)
#define REG_VALUE 1e-6f
#define Q_XX_WEIGHT 0.1f
#define R_XX_WEIGHT 500.0f
#define R_XX_LINEAR 10.0f

// Obstacle parameters (XY plane only)
#define OBS_CENTER_X -5.0f
#define OBS_CENTER_Y 0.0f
#define OBS_RADIUS 2.0f

// Goal position
#define GOAL_X 0.0f
#define GOAL_Y 0.0f
#define GOAL_Z 1.0f  // Hover at 1m altitude

// Physical dynamics matrices (from quadrotor_20hz_params.hpp)
tinytype A_phys_data[NX_PHYS * NX_PHYS] = {
    1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0245250, 0.0000000, 0.0500000, 0.0000000, 0.0000000, 0.0000000, 0.0002044, 0.0000000,
    0.0000000, 1.0000000, 0.0000000, -0.0245250, 0.0000000, 0.0000000, 0.0000000, 0.0500000, 0.0000000, -0.0002044, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0500000, 0.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.9810000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0122625, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, -0.9810000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, -0.0122625, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000};

tinytype B_phys_data[NX_PHYS * NU_PHYS] = {
    -0.0007069, 0.0007773, 0.0007091, -0.0007795,
    0.0007034, 0.0007747, -0.0007042, -0.0007739,
    0.0052554, 0.0052554, 0.0052554, 0.0052554,
    -0.1720966, -0.1895213, 0.1722891, 0.1893288,
    -0.1729419, 0.1901740, 0.1734809, -0.1907131,
    0.0123423, -0.0045148, -0.0174024, 0.0095748,
    -0.0565520, 0.0621869, 0.0567283, -0.0623632,
    0.0562756, 0.0619735, -0.0563386, -0.0619105,
    0.2102143, 0.2102143, 0.2102143, 0.2102143,
    -13.7677303, -15.1617018, 13.7831318, 15.1463003,
    -13.8353509, 15.2139209, 13.8784751, -15.2570451,
    0.9873856, -0.3611820, -1.3921880, 0.7659845};

// Build augmented A matrix: A_aug = [A, 0; 0, kron(A,A)]
void build_augmented_A(Eigen::MatrixXd& A_aug) {
    // Load physical A matrix
    Eigen::Matrix<tinytype, NX_PHYS, NX_PHYS> A_phys = 
        Map<Matrix<tinytype, NX_PHYS, NX_PHYS, RowMajor>>(A_phys_data);
    
    // Initialize augmented matrix to zero
    A_aug.setZero();
    
    // Top-left block: A (12x12)
    A_aug.block(0, 0, NX_PHYS, NX_PHYS) = A_phys.cast<double>();
    
    // Bottom-right block: kron(A, A) (144x144)
    Eigen::MatrixXd kron_AA = kroneckerProduct(A_phys.cast<double>(), A_phys.cast<double>());
    A_aug.block(NX_PHYS, NX_PHYS, NX_PHYS*NX_PHYS, NX_PHYS*NX_PHYS) = kron_AA;
}

// Build augmented B matrix: B_aug = [B, 0, 0, 0; 0, kron(B,A), kron(A,B), kron(B,B)]
void build_augmented_B(Eigen::MatrixXd& B_aug) {
    // Load physical matrices
    Eigen::Matrix<tinytype, NX_PHYS, NX_PHYS> A_phys = 
        Map<Matrix<tinytype, NX_PHYS, NX_PHYS, RowMajor>>(A_phys_data);
    Eigen::Matrix<tinytype, NX_PHYS, NU_PHYS> B_phys = 
        Map<Matrix<tinytype, NX_PHYS, NU_PHYS, RowMajor>>(B_phys_data);
    
    // Initialize augmented matrix to zero
    B_aug.setZero();
    
    // Top-left block: B (12x4)
    B_aug.block(0, 0, NX_PHYS, NU_PHYS) = B_phys.cast<double>();
    
    // Bottom blocks: [kron(B,A), kron(A,B), kron(B,B)]
    Eigen::MatrixXd kron_BA = 
        kroneckerProduct(B_phys.cast<double>(), A_phys.cast<double>());  // 144x48
    Eigen::MatrixXd kron_AB = 
        kroneckerProduct(A_phys.cast<double>(), B_phys.cast<double>());  // 144x48
    Eigen::MatrixXd kron_BB = 
        kroneckerProduct(B_phys.cast<double>(), B_phys.cast<double>());  // 144x16
    
    B_aug.block(NX_PHYS, NU_PHYS, NX_PHYS*NX_PHYS, NU_PHYS*NX_PHYS) = kron_BA;
    B_aug.block(NX_PHYS, NU_PHYS + NU_PHYS*NX_PHYS, NX_PHYS*NX_PHYS, NX_PHYS*NU_PHYS) = kron_AB;
    B_aug.block(NX_PHYS, NU_PHYS + NU_PHYS*NX_PHYS + NX_PHYS*NU_PHYS, NX_PHYS*NX_PHYS, NU_PHYS*NU_PHYS) = kron_BB;
}

// Q_aug = reg*I (156x156)
// Q is diagonal, so we store just the diagonal
tinytype Q_aug_data[NX_AUG];

// q_aug: zeros on physical states, q_xx on diagonal entries of vec(XX)
tinytype q_aug_data[NX_AUG];

// R_aug = reg*I on most, R_xx on diagonal of vec(uu^T)
tinytype R_aug_data[NU_AUG];

// r_aug: zeros except r_xx on diagonal of vec(uu^T)
tinytype r_aug_data[NU_AUG];

// Initialize cost vectors
void initialize_costs() {
    // Q_aug: all reg
    for (int i = 0; i < NX_AUG; i++) {
        Q_aug_data[i] = REG_VALUE;
    }
    
    // q_aug: zeros on physical (0-11), q_xx on diagonal of XX (12 entries starting at index 12)
    for (int i = 0; i < NX_PHYS; i++) {
        q_aug_data[i] = 0.0f;
    }
    for (int i = 0; i < NX_PHYS; i++) {
        for (int j = 0; j < NX_PHYS; j++) {
            int idx = NX_PHYS + i + j * NX_PHYS;  // column-major index into vec(XX)
            if (i == j) {
                q_aug_data[idx] = Q_XX_WEIGHT;
            } else {
                q_aug_data[idx] = 0.0f;
            }
        }
    }
    
    // R_aug: reg everywhere, except R_xx on diagonal of vec(uu^T)
    for (int i = 0; i < NU_PHYS; i++) {
        R_aug_data[i] = REG_VALUE;
    }
    for (int i = 0; i < NU_PHYS * NX_PHYS; i++) {
        R_aug_data[NU_PHYS + i] = REG_VALUE;  // vec(xu^T)
    }
    for (int i = 0; i < NX_PHYS * NU_PHYS; i++) {
        R_aug_data[NU_PHYS + NU_PHYS*NX_PHYS + i] = REG_VALUE;  // vec(ux^T)
    }
    // vec(uu^T): diagonal gets R_xx + reg, off-diagonal gets reg
    for (int i = 0; i < NU_PHYS; i++) {
        for (int j = 0; j < NU_PHYS; j++) {
            int idx = NU_PHYS + NU_PHYS*NX_PHYS + NX_PHYS*NU_PHYS + i + j*NU_PHYS;
            if (i == j) {
                R_aug_data[idx] = R_XX_WEIGHT + REG_VALUE;
            } else {
                R_aug_data[idx] = REG_VALUE;
            }
        }
    }
    
    // r_aug: zeros except r_xx on diagonal of vec(uu^T)
    for (int i = 0; i < NU_PHYS; i++) {
        r_aug_data[i] = 0.0f;
    }
    for (int i = 0; i < NU_PHYS * NX_PHYS; i++) {
        r_aug_data[NU_PHYS + i] = 0.0f;  // vec(xu^T)
    }
    for (int i = 0; i < NX_PHYS * NU_PHYS; i++) {
        r_aug_data[NU_PHYS + NU_PHYS*NX_PHYS + i] = 0.0f;  // vec(ux^T)
    }
    // vec(uu^T): diagonal gets r_xx, off-diagonal gets 0
    for (int i = 0; i < NU_PHYS; i++) {
        for (int j = 0; j < NU_PHYS; j++) {
            int idx = NU_PHYS + NU_PHYS*NX_PHYS + NX_PHYS*NU_PHYS + i + j*NU_PHYS;
            if (i == j) {
                r_aug_data[idx] = R_XX_LINEAR;
            } else {
                r_aug_data[idx] = 0.0f;
            }
        }
    }
}

// Collision constraint: px² + py² - 2*cx*px - 2*cy*py >= cx² + cy² - r²
// In augmented state: m^T x_aug >= n
void build_collision_constraint(Eigen::Matrix<tinytype, 1, NX_AUG>& m, tinytype& n) {
    m.setZero();
    
    // Coefficients on physical state
    m(0, 0) = -2.0f * OBS_CENTER_X;  // px coefficient
    m(0, 1) = -2.0f * OBS_CENTER_Y;  // py coefficient
    // All other physical states (z, angles, velocities) have zero coefficient
    
    // Coefficients on quadratic terms vec(XX)
    // px² is at index: NX_PHYS + (0 + 0*NX_PHYS) = 12 + 0 = 12
    // py² is at index: NX_PHYS + (1 + 1*NX_PHYS) = 12 + (1 + 12) = 12 + 13 = 25
    m(0, NX_PHYS + 0) = 1.0f;   // px² coefficient
    m(0, NX_PHYS + 1 + 1*NX_PHYS) = 1.0f;  // py² coefficient
    
    // RHS: -(cx² + cy²) + r²
    n = -(OBS_CENTER_X * OBS_CENTER_X + OBS_CENTER_Y * OBS_CENTER_Y) + OBS_RADIUS * OBS_RADIUS;
}

// Zero dynamics offset
tinytype fdyn_aug_data[NX_AUG] = {0.0f};  // Will be initialized with zeros

