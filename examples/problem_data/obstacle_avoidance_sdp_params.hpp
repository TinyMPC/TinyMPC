// Compare A_bar and B_bar with Julia
// Make the zeros/non-zeros exactly as Julia 

#pragma once

#include <tinympc/types.hpp>

// SDP Obstacle Avoidance with State Lifting Parameters
// Based on mathematical formulation from the meeting notes

#define NX_PHYS 4  // Physical states [px, py, vx, vy]
#define NU_PHYS 2  // Physical controls [ax, ay]

// Augmented dimensions
#define NX_AUG 20  // x̄ = [x, vec(xx^T)] ∈ R^20  (4 + 16 = 20)
#define NU_AUG 22  // ū = [u, vec(xu^T), vec(ux^T), vec(uu^T)] ∈ R^22  (2 + 8 + 8 + 4 = 22)

tinytype rho_value = 100.0;

// Physical dynamics matrices (dt = 1.0 from math document)
// A = [[1, 0, 1, 0],
//      [0, 1, 0, 1], 
//      [0, 0, 1, 0],
//      [0, 0, 0, 1]]
tinytype A_phys_data[NX_PHYS * NX_PHYS] = {
    1.0f, 0.0f, 1.0f, 0.0f,  // px_{k+1} = px_k + vx_k
    0.0f, 1.0f, 0.0f, 1.0f,  // py_{k+1} = py_k + vy_k
    0.0f, 0.0f, 1.0f, 0.0f,  // vx_{k+1} = vx_k + ax_k
    0.0f, 0.0f, 0.0f, 1.0f   // vy_{k+1} = vy_k + ay_k
};

// B = [[0.5, 0],
//      [0, 0.5],
//      [1, 0],
//      [0, 1]]
tinytype B_phys_data[NX_PHYS * NU_PHYS] = {
    0.5f, 0.0f,  // px += 0.5*ax
    0.0f, 0.5f,  // py += 0.5*ay
    1.0f, 0.0f,  // vx += ax  
    0.0f, 1.0f   // vy += ay
};

// Custom Kronecker product implementation
template<int M, int N, int P, int Q>
Eigen::Matrix<tinytype, M*P, N*Q> kronecker_product(
    const Eigen::Matrix<tinytype, M, N>& A,
    const Eigen::Matrix<tinytype, P, Q>& B) {
    
    Eigen::Matrix<tinytype, M*P, N*Q> result;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            result.template block<P, Q>(i*P, j*Q) = A(i, j) * B;
        }
    }
    return result;
}

// Augmented dynamics matrices will be constructed using Kronecker products
// Ā = [A_d,    0_{4×16}]
//     [0_{16×4}, kron(A_d, A_d)]
//
// B̄ = [B_d,    0_{4×20}]
//     [0_{16×2}, kron(B_d, A_d), kron(A_d, B_d), kron(B_d, B_d)]

// Function to build augmented A matrix (20x20)
void build_augmented_A(Eigen::Matrix<tinytype, NX_AUG, NX_AUG>& A_aug) {
    using namespace Eigen;
    
    // Physical A matrix
    Matrix<tinytype, NX_PHYS, NX_PHYS, RowMajor> A_phys = 
        Map<Matrix<tinytype, NX_PHYS, NX_PHYS, RowMajor>>(A_phys_data);
    
    A_aug.setZero();
    
    // Top-left: A_d (4x4)
    A_aug.block<NX_PHYS, NX_PHYS>(0, 0) = A_phys;
    
    // Bottom-right: kron(A_d, A_d) (16x16)
    auto kron_AA = kronecker_product<NX_PHYS, NX_PHYS, NX_PHYS, NX_PHYS>(A_phys, A_phys);
    A_aug.block<16, 16>(NX_PHYS, NX_PHYS) = kron_AA;
}

// Function to build augmented B matrix (20x22)
void build_augmented_B(Eigen::Matrix<tinytype, NX_AUG, NU_AUG>& B_aug) {
    using namespace Eigen;
    
    // Physical matrices
    Matrix<tinytype, NX_PHYS, NX_PHYS, RowMajor> A_phys = 
        Map<Matrix<tinytype, NX_PHYS, NX_PHYS, RowMajor>>(A_phys_data);
    Matrix<tinytype, NX_PHYS, NU_PHYS, RowMajor> B_phys = 
        Map<Matrix<tinytype, NX_PHYS, NU_PHYS, RowMajor>>(B_phys_data);
    
    B_aug.setZero();
    
    // Top-left: B_d (4x2)
    B_aug.block<NX_PHYS, NU_PHYS>(0, 0) = B_phys;
    
    // Bottom part: Kronecker products for quadratic terms
    // From math: vec(x_{k+1} x_{k+1}^T) = vec(A x_k x_k^T A^T + A x_k u_k^T B^T + B u_k x_k^T A^T + B u_k u_k^T B^T)
    // This gives: kron(A,A)*vec(xx^T) + kron(B,A)*vec(xu^T) + kron(A,B)*vec(ux^T) + kron(B,B)*vec(uu^T)
    
    // Structure: [0_{16×2}, kron(B,A), kron(A,B), kron(B,B)]
    // Dimensions: B_phys is 4x2, A_phys is 4x4
    auto kron_BA = kronecker_product<NX_PHYS, NU_PHYS, NX_PHYS, NX_PHYS>(B_phys, A_phys);  // (4×2)⊗(4×4) = 16×8
    auto kron_AB = kronecker_product<NX_PHYS, NX_PHYS, NX_PHYS, NU_PHYS>(A_phys, B_phys);  // (4×4)⊗(4×2) = 16×8  
    auto kron_BB = kronecker_product<NX_PHYS, NU_PHYS, NX_PHYS, NU_PHYS>(B_phys, B_phys);  // (4×2)⊗(4×2) = 16×4
    
    // Bottom-left: zeros (16×2) - already set by setZero()
    B_aug.block<16, 8>(NX_PHYS, NU_PHYS) = kron_BA;           // kron(B,A) for vec(xu^T) terms
    B_aug.block<16, 8>(NX_PHYS, NU_PHYS + 8) = kron_AB;       // kron(A,B) for vec(ux^T) terms  
    B_aug.block<16, 4>(NX_PHYS, NU_PHYS + 16) = kron_BB;      // kron(B,B) for vec(uu^T) terms
}

// Julia weight parameters - EXACT VALUES
#define Q_XX_WEIGHT 0.1f     // Julia: q_xx = 0.1
#define R_XX_WEIGHT 500.0f   // Julia: R_xx = 500.0
#define R_XX_LINEAR 10.0f    // Julia: r_xx = 10.0
#define REG_VALUE 1e-6f      // Julia: reg = 1e-6

// Cost matrices for augmented system - EXACTLY matching Julia
// Q = zeros(20,20) + reg*I(20,20) (diagonal only, off-diagonal is zero)
tinytype Q_aug_data[NX_AUG] = {
    // Physical state costs: reg*I (first 4) - Julia uses zeros + reg*I
    REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE,
    // Quadratic term costs: reg*I (remaining 16) - Julia uses zeros + reg*I
    REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE,
    REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE, 
    REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE,
    REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE
};

// Linear cost vector q = [zeros(4); vec(q_xx*I(4,4))]
tinytype q_aug_data[NX_AUG] = {
    // Physical states: zeros(4)
    0.0f, 0.0f, 0.0f, 0.0f,
    // Quadratic terms: vec(q_xx*I(4,4)) = [q_xx, 0, 0, 0, 0, q_xx, 0, 0, 0, 0, q_xx, 0, 0, 0, 0, q_xx]
    Q_XX_WEIGHT, 0.0f, 0.0f, 0.0f,  // first column of q_xx*I
    0.0f, Q_XX_WEIGHT, 0.0f, 0.0f,  // second column of q_xx*I
    0.0f, 0.0f, Q_XX_WEIGHT, 0.0f,  // third column of q_xx*I
    0.0f, 0.0f, 0.0f, Q_XX_WEIGHT   // fourth column of q_xx*I
};

// R = Diagonal([zeros(2); zeros(16); vec(R_xx*I(2,2))]) + reg*I(22,22)
tinytype R_aug_data[NU_AUG] = {
    // Physical controls: zeros(2) + reg
    REG_VALUE, REG_VALUE,
    // Cross terms xu^T: zeros(8) + reg
    REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE,
    // Cross terms ux^T: zeros(8) + reg  
    REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE, REG_VALUE,
    // uu^T terms: vec(R_xx*I(2,2)) + reg = [R_xx+reg, reg, reg, R_xx+reg]
    R_XX_WEIGHT + REG_VALUE, REG_VALUE, REG_VALUE, R_XX_WEIGHT + REG_VALUE
};

// Linear cost vector r = [zeros(18); vec(r_xx*I(2,2))]
tinytype r_aug_data[NU_AUG] = {
    // Physical controls + cross terms: zeros(18)
    0.0f, 0.0f,  // u
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // xu^T
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // ux^T
    // uu^T terms: vec(r_xx*I(2,2)) = [r_xx, 0, 0, r_xx]
    R_XX_LINEAR, 0.0f, 0.0f, R_XX_LINEAR
};

// Obstacle parameters
#define OBS_CENTER_X -5.0f
#define OBS_CENTER_Y 0.0f
#define OBS_RADIUS 2.0f

// Goal position
#define GOAL_X 0.0f
#define GOAL_Y 0.0f

// Linear collision avoidance constraint parameters - EXACTLY matching Julia
// Julia: m = [-2*x_obs[1]; -2*x_obs[2]; zeros(2); 1; zeros(4); 1; zeros(10)]'
// Julia: n = -x_obs'*x_obs + r_obs^2
// Julia: m*x_bar[:,k] >= n

void build_collision_constraint(Eigen::Matrix<tinytype, 1, NX_AUG>& m, tinytype& n) {
    m.setZero();
    
    // Julia's m vector exactly:
    // [-2*x_obs[1]; -2*x_obs[2]; zeros(2); 1; zeros(4); 1; zeros(10)]'
    
    m(0, 0) = -2.0 * OBS_CENTER_X;  // -2*x_obs[1] (px coefficient)
    m(0, 1) = -2.0 * OBS_CENTER_Y;  // -2*x_obs[2] (py coefficient)
    // m(0, 2) = 0.0;  // zeros(2) - vx coefficient
    // m(0, 3) = 0.0;  // zeros(2) - vy coefficient
    
    // Quadratic terms vec(xx^T): [X00, X10, X20, X30, X01, X11, X21, X31, X02, X12, X22, X32, X03, X13, X23, X33]
    // We want coefficients for X00 (px^2) and X11 (py^2)
    m(0, 4) = 1.0;   // X00 = px^2 (position 4 in augmented state)
    // m(0, 5) = 0.0;   // X10 = px*py (zeros(4) part)
    // m(0, 6) = 0.0;   // X20 = px*vx
    // m(0, 7) = 0.0;   // X30 = px*vy
    // m(0, 8) = 0.0;   // X01 = py*px
    m(0, 9) = 1.0;   // X11 = py^2 (position 9 in augmented state)
    // Remaining 10 elements are zeros(10)
    
    // Julia's n value: -x_obs'*x_obs + r_obs^2
    n = -(OBS_CENTER_X * OBS_CENTER_X + OBS_CENTER_Y * OBS_CENTER_Y) + OBS_RADIUS * OBS_RADIUS;
}

// Zero dynamics offset
tinytype fdyn_aug_data[NX_AUG] = {0.0f};  // Will be initialized with zeros
