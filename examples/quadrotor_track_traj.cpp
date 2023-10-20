#include <iostream>

#include <tinympc/admm.hpp>
#include "work_data/quadrotor_20hz_work.hpp"
#include "trajectory_data/quadrotor_100hz_ref_hover.hpp"

using Eigen::Matrix;

#define DT 1/100

extern "C" {

int main() {

    // Copy data from work_data/quadrotor*.hpp
    TinyCache cache;
    cache.rho = rho_value;
    cache.Kinf = Eigen::Map<Matrix<tinytype, NINPUTS, NSTATES, Eigen::RowMajor>>(Kinf_data);
    cache.Pinf = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Pinf_data);
    cache.Quu_inv = Eigen::Map<Matrix<tinytype, NINPUTS, NINPUTS, Eigen::RowMajor>>(Quu_inv_data);
    cache.AmBKt = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(AmBKt_data);
    cache.coeff_d2p = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(coeff_d2p_data);

    TinyWorkspace work;
    work.Adyn = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Adyn_data);
    cache.Bdyn = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(Bdyn_data);
    work.Q = Eigen::Map<tiny_VectorNx>(Q_data);
    work.Qf = Eigen::Map<tiny_VectorNx>(Qf_data);
    work.R = Eigen::Map<tiny_VectorNu>(R_data);
    work.u_min = tiny_MatrixNuNhm1::Constant(-0.5);
    work.u_max = tiny_MatrixNuNhm1::Constant(0.5);
    work.x_min = tiny_MatrixNxNh::Constant(-1000);
    work.x_max = tiny_MatrixNxNh::Constant(1000);

    work.Xref = tiny_MatrixNxNh::Zero();
    work.Uref = tiny_MatrixNuNhm1::Zero();

    work.x = tiny_MatrixNxNh::Zero();
    work.q = tiny_MatrixNxNh::Zero();
    work.p = tiny_MatrixNxNh::Zero();
    work.v = tiny_MatrixNxNh::Zero();
    work.vnew = tiny_MatrixNxNh::Zero();
    work.g = tiny_MatrixNxNh::Zero();

    work.u = tiny_MatrixNuNhm1::Zero();
    work.r = tiny_MatrixNuNhm1::Zero();
    work.d = tiny_MatrixNuNhm1::Zero();
    work.z = tiny_MatrixNuNhm1::Zero();
    work.znew = tiny_MatrixNuNhm1::Zero();
    work.y = tiny_MatrixNuNhm1::Zero();

    work.primal_residual_state = 0;
    work.primal_residual_input = 0;
    work.dual_residual_state = 0;
    work.dual_residual_input = 0;

    TinySettings settings;
    settings.abs_pri_tol = 0.001;
    settings.abs_dua_tol = 0.001;
    settings.status = 0;
    settings.iter = 0;
    settings.max_iter = 100;
    settings.check_termination = 10;

    // Copy reference trajectory into Eigen matrix
    // Matrix<tinytype, NSTATES, NTOTAL, Eigen::ColMajor> Xref_total = Eigen::Map<Matrix<tinytype, NTOTAL, NSTATES, Eigen::RowMajor>>(Xref_data).transpose();
    tiny_VectorNx Xref_origin;
    Xref_origin << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    // work.Xref = Xref_total.block<NSTATES, NHORIZON>(0,0);
    work.Xref = Xref_origin.replicate<1,NHORIZON>();
    // work.x.col(0) = work.Xref.col(0);
    work.x.col(0) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    std::cout << work.Xref << std::endl;

    solve_admm(&work, &work);
    std::cout << work.iter << std::endl;
    std::cout << work.u.col(0)(0) << std::endl;
    std::cout << work.u.col(0)(1) << std::endl;
    std::cout << work.u.col(0)(2) << std::endl;
    std::cout << work.u.col(0)(3) << std::endl;

    // Matrix<tinytype, 3, 1> obstacle_center = {0.0, 2.0, 0.5};
    // tinytype obstacle_velocity = 1 * DT;

    // tinytype r_obstacle = 0.75;
    // tinytype b;
    // Matrix<tinytype, 3, 1> xc;
    // Matrix<tinytype, 3, 1> a;
    // Matrix<tinytype, 3, 1> q_c;
    // for (int i=0; i<NHORIZON; i++) {
    //     xc = obstacle_center - work.Xref.col(i).head(3);
    //     a = xc/norm(xc);
    //     work.A_constraints[i].head(3) = a.transpose();

    //     q_c = obstacle_center - r_obstacle*a;
    //     b = a.transpose() * q_c;
    //     work.x_max[i](0) = b;
    //     // std::cout << work.A_constraints[i].head(3) << std::endl;
    //     // std::cout << work.x_max[i](0) << "\n" << std::endl;
    // }


    // std::cout << work.Xref << std::endl;
    // std::cout << work.Q << std::endl;
    // work.q = work.Xref.array().colwise() * work.Q.array();
    // std::cout << work.q << std::endl;

    // std::cout << NHORIZON << std::endl;
    // std::cout << work.u_min << std::endl;
    // std::cout << work.u_max << std::endl;
    // for (int i=0; i<NHORIZON; i++) {
    //     std::cout << work.A_constraints[i] << std::endl;
    //     std::cout << work.x_min[i] << std::endl;
    //     std::cout << work.x_max[i] << std::endl;
    // }

    // std::cout << work.Xref << std::endl;
    // std::cout << work.Uref << std::endl;
    // std::cout << work.cache.Adyn << std::endl;
    // std::cout << work.cache.Bdyn << std::endl;
    // std::cout << work.cache.rho << std::endl;
    // std::cout << work.cache.Kinf << std::endl;
    // std::cout << work.cache.Pinf << std::endl;
    // std::cout << work.cache.Quu_inv << std::endl;
    // std::cout << work.cache.AmBKt << std::endl;
    // std::cout << work.cache.coeff_d2p << std::endl;

    return 0;
}

} /* extern "C" */