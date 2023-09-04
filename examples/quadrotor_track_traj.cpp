#include <iostream>

#include <tinympc/admm.hpp>
#include "problem_data/quadrotor_50hz_params.hpp"
#include "trajectory_data/quadrotor_100hz_ref_hover.hpp"

using Eigen::Matrix;

#define DT 1/100

extern "C" {

int main() {

    // Copy data from problem_data/quadrotor*.hpp
    struct tiny_cache cache;
    cache.Adyn = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Adyn_data);
    cache.Bdyn = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(Bdyn_data);
    cache.rho = rho_value;
    cache.Kinf = Eigen::Map<Matrix<tinytype, NINPUTS, NSTATES, Eigen::RowMajor>>(Kinf_data);
    cache.Pinf = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Pinf_data);
    cache.Quu_inv = Eigen::Map<Matrix<tinytype, NINPUTS, NINPUTS, Eigen::RowMajor>>(Quu_inv_data);
    cache.AmBKt = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(AmBKt_data);
    cache.coeff_d2p = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(coeff_d2p_data);

    struct tiny_params params;
    params.Q = Eigen::Map<tiny_VectorNx>(Q_data);
    params.Qf = Eigen::Map<tiny_VectorNx>(Qf_data);
    params.R = Eigen::Map<tiny_VectorNu>(R_data);
    params.u_min = tiny_MatrixNuNhm1::Constant(-0.5);
    params.u_max = tiny_MatrixNuNhm1::Constant(0.5);
    for (int i=0; i<NHORIZON; i++) {
        params.x_min[i] = tiny_VectorNc::Constant(-99999); // Currently unused
        params.x_max[i] = tiny_VectorNc::Constant(99999);
        params.A_constraints[i] = tiny_MatrixNcNx::Zero();
    }
    params.Xref = tiny_MatrixNxNh::Zero();
    params.Uref = tiny_MatrixNuNhm1::Zero();
    params.cache = cache;

    struct tiny_problem problem;
    problem.x = tiny_MatrixNxNh::Zero();
    problem.q = tiny_MatrixNxNh::Zero();
    problem.p = tiny_MatrixNxNh::Zero();
    problem.v = tiny_MatrixNxNh::Zero();
    problem.vnew = tiny_MatrixNxNh::Zero();
    problem.g = tiny_MatrixNxNh::Zero();

    problem.u = tiny_MatrixNuNhm1::Zero();
    problem.r = tiny_MatrixNuNhm1::Zero();
    problem.d = tiny_MatrixNuNhm1::Zero();
    problem.z = tiny_MatrixNuNhm1::Zero();
    problem.znew = tiny_MatrixNuNhm1::Zero();
    problem.y = tiny_MatrixNuNhm1::Zero();

    problem.primal_residual_state = 0;
    problem.primal_residual_input = 0;
    problem.dual_residual_state = 0;
    problem.dual_residual_input = 0;
    problem.abs_tol = 0.001;
    problem.status = 0;
    problem.iter = 0;
    problem.max_iter = 20;
    problem.iters_check_rho_update = 10;

    // Copy reference trajectory into Eigen matrix
    // Matrix<tinytype, NSTATES, NTOTAL, Eigen::ColMajor> Xref_total = Eigen::Map<Matrix<tinytype, NTOTAL, NSTATES, Eigen::RowMajor>>(Xref_data).transpose();
    Matrix<tinytype, NSTATES, 1> Xref_origin;
    Xref_origin << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    // params.Xref = Xref_total.block<NSTATES, NHORIZON>(0,0);
    params.Xref = Xref_origin.replicate<1,NHORIZON>();
    // problem.x.col(0) = params.Xref.col(0);
    problem.x.col(0) << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    // std::cout << params.Xref << std::endl;

    solve_admm(&problem, &params);
    // std::cout << problem.x.block<NSTATES, 5>(0,0) << std::endl;
    // std::cout << problem.x.block<NSTATES, 5>(0,NHORIZON-5) << std::endl;

    // std::cout << problem.iter << std::endl;
    // std::cout << problem.u.col(0)(0) << std::endl;
    // std::cout << problem.u.col(0)(1) << std::endl;
    // std::cout << problem.u.col(0)(2) << std::endl;
    // std::cout << problem.u.col(0)(3) << std::endl;

    // Matrix<tinytype, 3, 1> obstacle_center = {0.0, 2.0, 0.5};
    // tinytype obstacle_velocity = 1 * DT;

    // tinytype r_obstacle = 0.75;
    // tinytype b;
    // Matrix<tinytype, 3, 1> xc;
    // Matrix<tinytype, 3, 1> a;
    // Matrix<tinytype, 3, 1> q_c;
    // for (int i=0; i<NHORIZON; i++) {
    //     xc = obstacle_center - params.Xref.col(i).head(3);
    //     a = xc/norm(xc);
    //     params.A_constraints[i].head(3) = a.transpose();

    //     q_c = obstacle_center - r_obstacle*a;
    //     b = a.transpose() * q_c;
    //     params.x_max[i](0) = b;
    //     // std::cout << params.A_constraints[i].head(3) << std::endl;
    //     // std::cout << params.x_max[i](0) << "\n" << std::endl;
    // }


    // std::cout << params.Xref << std::endl;
    // std::cout << params.Q << std::endl;
    // problem.q = params.Xref.array().colwise() * params.Q.array();
    // std::cout << problem.q << std::endl;

    // std::cout << NHORIZON << std::endl;
    // std::cout << params.u_min << std::endl;
    // std::cout << params.u_max << std::endl;
    // for (int i=0; i<NHORIZON; i++) {
    //     std::cout << params.A_constraints[i] << std::endl;
    //     std::cout << params.x_min[i] << std::endl;
    //     std::cout << params.x_max[i] << std::endl;
    // }

    // std::cout << params.Xref << std::endl;
    // std::cout << params.Uref << std::endl;
    // std::cout << params.cache.Adyn << std::endl;
    // std::cout << params.cache.Bdyn << std::endl;
    // std::cout << params.cache.rho << std::endl;
    // std::cout << params.cache.Kinf << std::endl;
    // std::cout << params.cache.Pinf << std::endl;
    // std::cout << params.cache.Quu_inv << std::endl;
    // std::cout << params.cache.AmBKt << std::endl;
    // std::cout << params.cache.coeff_d2p << std::endl;

    return 0;
}

} /* extern "C" */