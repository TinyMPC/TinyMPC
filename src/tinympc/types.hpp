#pragma once


#include <Eigen.h>
// #include <Eigen/Core>
// #include <Eigen/LU>

using namespace Eigen;


#ifdef __cplusplus
extern "C" {
#endif

typedef double tinytype;  // should be double if you want to generate code
typedef Matrix<tinytype, Dynamic, Dynamic> tinyMatrix;
typedef Matrix<tinytype, Dynamic, 1> tinyVector;

// typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;
// typedef Matrix<tinytype, NINPUTS, 1> tiny_VectorNu;
// typedef Matrix<tinytype, NSTATES, NSTATES> tiny_MatrixNxNx;
// typedef Matrix<tinytype, NSTATES, NINPUTS> tiny_MatrixNxNu;
// typedef Matrix<tinytype, NINPUTS, NSTATES> tiny_MatrixNuNx;
// typedef Matrix<tinytype, NINPUTS, NINPUTS> tiny_MatrixNuNu;

// typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;       // Nu x Nh
// typedef Matrix<tinytype, NINPUTS, NHORIZON - 1> tiny_MatrixNuNhm1; // Nu x Nh-1

/**
 * Solution
 */
typedef struct {
    int iter;
    int solved;
    tinyMatrix x; // nx x N
    tinyMatrix u; // nu x N-1
} TinySolution;


/**
* Matrices that must be recomputed with changes in time step, rho
*/
typedef struct {
    tinytype rho;
    tinyMatrix Kinf;       // nu x nx
    tinyMatrix Pinf;       // nx x nx
    tinyMatrix Quu_inv;    // nu x nu
    tinyMatrix AmBKt;      // nx x nx
    tinyVector APf;        // nx x 1
    tinyVector BPf;        // nu x 1
    tinyMatrix C1;         // From adaptive rho
    tinyMatrix C2;         // From adaptive rho
    
    // Sensitivity matrices for adaptive rho
    tinyMatrix dKinf_drho;
    tinyMatrix dPinf_drho;
    tinyMatrix dC1_drho;
    tinyMatrix dC2_drho;
} TinyCache;
/**
* User settings
*/
typedef struct {
    tinytype abs_pri_tol;
    tinytype abs_dua_tol;
    int max_iter;
    int check_termination;
    int en_state_bound;
    int en_input_bound;
    int en_state_soc;
    int en_input_soc;
    int en_state_linear;
    int en_input_linear;
        
    // Add adaptive rho parameters
    int adaptive_rho;                  // Enable/disable adaptive rho (1/0)
    tinytype adaptive_rho_min;         // Minimum value for rho
    tinytype adaptive_rho_max;         // Maximum value for rho
    int adaptive_rho_enable_clipping;  // Enable/disable clipping of rho (1/0)
} TinySettings;


/**
 * Problem variables
 */
typedef struct {
    int nx;          // Number of states
    int nu;          // Number of control inputs
    int N;           // Number of knotpoints in the horizon

    // State and input
    tinyMatrix x;    // nx x N
    tinyMatrix u;    // nu x N-1

    // Linear control cost terms
    tinyMatrix q;    // nx x N
    tinyMatrix r;    // nu x N-1

    // Linear Riccati backward pass terms
    tinyMatrix p;    // nx x N
    tinyMatrix d;    // nu x N-1

    // Bound constraint variables
    // Slack variables
    tinyMatrix v;    // nx x N
    tinyMatrix vnew; // nx x N
    tinyMatrix z;    // nu x N-1
    tinyMatrix znew; // nu x N-1

    // Dual variables
    tinyMatrix g;    // nx x N
    tinyMatrix y;    // nu x N-1
    
    // State and input bounds
    tinyMatrix x_min;   // nx x N
    tinyMatrix x_max;   // nx x N
    tinyMatrix u_min;   // nu x N-1
    tinyMatrix u_max;   // nu x N-1

    // Cone constraint variables
    // Variables to keep track of general cone information
    int numStateCones; // Number of cone constraints on states at each time step
    int numInputCones; // Number of cone constraints on inputs at each time step
    tinyVector cx; // One coefficient for each state cone
    tinyVector cu; // One coefficient for each input cone
    VectorXi Acx;  // Start indices for each state cone
    VectorXi Acu;  // Start indices for each input cone
    VectorXi qcx;  // Dimension for each state cone
    VectorXi qcu;  // Dimension for each input cone

    // Slack variables
    tinyMatrix vc; // nx x N
    tinyMatrix vcnew; // nx x N
    tinyMatrix zc; // nu x N-1
    tinyMatrix zcnew; // nu x N-1

    // Dual variables
    tinyMatrix gc; // nx x N
    tinyMatrix yc; // nu x N-1

    // Linear constraint variables
    // Variables to keep track of general linear constraint information
    int numStateLinear; // Number of linear constraints on states at each time step
    int numInputLinear; // Number of linear constraints on inputs at each time step
    
    // Constraint matrices and vectors
    tinyMatrix Alin_x; // Normal vectors for state linear constraints (numStateLinear x nx)
    tinyVector blin_x; // Offset values for state linear constraints (numStateLinear x 1)
    tinyMatrix Alin_u; // Normal vectors for input linear constraints (numInputLinear x nu)
    tinyVector blin_u; // Offset values for input linear constraints (numInputLinear x 1)

    // Slack variables for linear constraints
    tinyMatrix vl; // nx x N
    tinyMatrix vlnew; // nx x N
    tinyMatrix zl; // nu x N-1
    tinyMatrix zlnew; // nu x N-1

    // Dual variables for linear constraints
    tinyMatrix gl; // nx x N
    tinyMatrix yl; // nu x N-1

    

    // Q, R, A, B, f given by user
    tinyVector Q;       // nx x 1
    tinyVector R;       // nu x 1
    tinyMatrix Adyn;    // nx x nx (state transition matrix)
    tinyMatrix Bdyn;    // nx x nu (control matrix)
    tinyVector fdyn;    // nx x 1 (affine vector)

    // Reference trajectory to track for one horizon
    tinyMatrix Xref;    // nx x N
    tinyMatrix Uref;    // nu x N-1

    // Temporaries
    tinyVector Qu;      // nu x 1



    // Variables for keeping track of solve status
    tinytype primal_residual_state;
    tinytype primal_residual_input;
    tinytype dual_residual_state;
    tinytype dual_residual_input;
    int status;
    int iter;
} TinyWorkspace;

/**
 * Main TinyMPC solver structure that holds all information.
 */
typedef struct {
    TinySolution *solution; // Solution
    TinySettings *settings; // Problem settings
    TinyCache *cache;       // Problem cache
    TinyWorkspace *work;    // Solver workspace
} TinySolver;


// Add at the top with other definitions
#define BENCH_NX 12
#define BENCH_NU 4

#ifdef __cplusplus
}
#endif
