#include <tinympc/tiny_data_workspace.hpp>

#ifdef __cplusplus
extern "C"
{
    /* Define the settings structure */
    TinySettings settings = {
        (tinytype)0.001, // abs_pri_tol
        (tinytype)0.001, // abs_dua_tol
        1000,            // max_iter
        10,              // check_termination
        1,               // en_state_bound
        1                // en_input_bound
    };

    /* Define the cache structure */
    TinyCache cache = {
        (tinytype)0.1,                                                                 // rho
        (tiny_MatrixNxNu() << 0.2207, 0.2699, 0.1022, 1.112).finished(),               // Kinf
        (tiny_MatrixNxNx() << 1.227, 0.3854, 0.3854, 4.795).finished(),                // Pinf
        (tiny_MatrixNuNu() << 0.02849, -0.05066, -0.05066, 0.1395).finished(),         // Quu_inv
        (tiny_MatrixNxNu() << 0.03112, 0.0148, 0.8547, -0.1916).finished(),            // AmBKt
        (tiny_MatrixNxNu() << -1.416e-07, -3.076e-08, 2.592e-06, 5.631e-07).finished() // coeff_d2p
    };

    // Do the same thing with workspace structure. But I am not sure if this is allocated on the stack or heap, at compile time or run time.

    /* Define the workspace structure */
    TinyWorkspace work;
    // TinyWorkspace work = {
    //     Eigen::Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>::Zero(),      // Adyn
    //     Eigen::Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>::Zero(),      // Bdyn
    //     Eigen::Matrix<tinytype, NSTATES, NHORIZON, Eigen::ColMajor>::Zero(),     // x
    //     Eigen::Matrix<tinytype, NINPUTS, NHORIZON - 1, Eigen::ColMajor>::Zero(), // u
    //     Eigen::Matrix<tinytype, NSTATES, NHORIZON, Eigen::ColMajor>::Zero(),     // Xref
    //     Eigen::Matrix<tinytype, NINPUTS, NHORIZON - 1, Eigen::ColMajor>::Zero(), // Uref
    //     Eigen::Matrix<tinytype, NSTATES, NHORIZON, Eigen::ColMajor>::Zero(),     // q
    //     Eigen::Matrix<tinytype, NINPUTS, NHORIZON - 1, Eigen::ColMajor>::Zero(), // r
    //     Eigen::Matrix<tinytype, NSTATES, NHORIZON, Eigen::ColMajor>::Zero(),     // p
    //     Eigen::Matrix<tinytype, NSTATES, NHORIZON, Eigen::ColMajor>::Zero(),     // v
    //     Eigen::Matrix<tinytype, NSTATES, NHORIZON, Eigen::ColMajor>::Zero(),     // vnew
    //     Eigen::Matrix<tinytype, NSTATES, NHORIZON, Eigen::ColMajor>::Zero(),     // g
    //     Eigen::Matrix<tinytype, NINPUTS, NHORIZON - 1, Eigen::ColMajor>::Zero(), // d
    //     Eigen::Matrix<tinytype, NINPUTS, NHORIZON - 1, Eigen::ColMajor>::Zero(), // z
    //     Eigen::Matrix<tinytype, NINPUTS, NHORIZON - 1, Eigen::ColMajor>::Zero(), // znew
    //     Eigen::Matrix<tinytype, NINPUTS, NHORIZON - 1, Eigen::ColMajor>::Zero(), // y
    //     Eigen::Matrix<tinytype, NSTATES, NHORIZON, Eigen::ColMajor>::Zero(),     // x_min
    //     Eigen::Matrix<tinytype, NSTATES, NHORIZON, Eigen::ColMajor>::Zero(),     // x_max
    //     Eigen::Matrix<tinytype, NINPUTS, NHORIZON - 1, Eigen::ColMajor>::Zero(), // u_min
    //     Eigen::Matrix<tinytype, NINPUTS, NHORIZON - 1, Eigen::ColMajor>::Zero(), // u_max
    //     (tinytype)0,                                                             // primal_residual_state
    //     (tinytype)0,                                                             // primal_residual_input
    //     (tinytype)0,                                                             // dual_residual_state
    //     (tinytype)0,                                                             // dual_residual_input
    //     0,                                                                       // status
    //     0                                                                        // iter
    // };

    /* Define the solver structure */
    TinySolver tiny_data_solver = {
        &settings,
        &cache,
        &work};
}
#endif