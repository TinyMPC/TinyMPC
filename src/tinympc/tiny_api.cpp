#include "tiny_api.hpp"
#include "tiny_api_constants.hpp"

#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

using namespace Eigen;
IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

void tiny_precompute_and_set_cache(TinyCache *cache, tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix Q, tinyMatrix R, int nx, int nu, double rho) {
    
    // Update by adding rho * identity matrix to Q, R
    Q = Q + rho * tinyMatrix::Ones(nx, 1);
    R = R + rho * tinyMatrix::Ones(nu, 1);
    tinyMatrix Q1 = Q.array().matrix().asDiagonal();
    tinyMatrix R1 = R.array().matrix().asDiagonal();

    // Printing
    std::cout << "A = " << Adyn.format(CleanFmt) << std::endl;
    std::cout << "B = " << Bdyn.format(CleanFmt) << std::endl;
    std::cout << "Q = " << Q1.format(CleanFmt) << std::endl;
    std::cout << "R = " << R1.format(CleanFmt) << std::endl;
    std::cout << "rho = " << rho << std::endl;

    // Riccati recursion to get Kinf, Pinf
    tinyMatrix Ktp1 = tinyMatrix::Zero(nu, nx);
    tinyMatrix Ptp1 = rho * tinyMatrix::Ones(nx, 1).array().matrix().asDiagonal();
    tinyMatrix Kinf = tinyMatrix::Zero(nu, nx);
    tinyMatrix Pinf = tinyMatrix::Zero(nx, nx);

    for (int i = 0; i < 1000; i++)
    {
        Kinf = (R1 + Bdyn.transpose() * Ptp1 * Bdyn).inverse() * Bdyn.transpose() * Ptp1 * Adyn;
        Pinf = Q1 + Adyn.transpose() * Ptp1 * (Adyn - Bdyn * Kinf);
        // if Kinf converges, break
        if ((Kinf - Ktp1).cwiseAbs().maxCoeff() < 1e-5)
        {
            std::cout << "Kinf converged after " << i + 1 << " iterations" << std::endl;
            break;
        }
        Ktp1 = Kinf;
        Ptp1 = Pinf;
    }

    std::cout << "Precomputing finished" << std::endl;

    // Compute cached matrices
    tinyMatrix Quu_inv = (R1 + Bdyn.transpose() * Pinf * Bdyn).inverse();
    tinyMatrix AmBKt = (Adyn - Bdyn * Kinf).transpose();

    std::cout << "Kinf = " << Kinf.format(CleanFmt) << std::endl;
    std::cout << "Pinf = " << Pinf.format(CleanFmt) << std::endl;
    std::cout << "Quu_inv = " << Quu_inv.format(CleanFmt) << std::endl;
    std::cout << "AmBKt = " << AmBKt.format(CleanFmt) << std::endl;

    cache->rho = rho;
    cache->Kinf = Kinf;
    cache->Pinf = Pinf;
    cache->Quu_inv = Quu_inv;
    cache->AmBKt = AmBKt;
}

void tiny_update_settings(TinySettings* settings, tinytype abs_pri_tol, tinytype abs_dua_tol,
                    int max_iter, int check_termination, 
                    int en_state_bound, int en_input_bound) {
    settings->abs_pri_tol = abs_pri_tol;
    settings->abs_dua_tol = abs_dua_tol;
    settings->max_iter = max_iter;
    settings->check_termination = check_termination;
    settings->en_state_bound = en_state_bound;
    settings->en_input_bound = en_input_bound;
}

void tiny_set_default_settings(TinySettings* settings) {
    settings->abs_pri_tol = TINY_DEFAULT_ABS_PRI_TOL;
    settings->abs_dua_tol = TINY_DEFAULT_ABS_DUA_TOL;
    settings->max_iter = TINY_DEFAULT_MAX_ITER;
    settings->check_termination = TINY_DEFAULT_CHECK_TERMINATION;
    settings->en_state_bound = TINY_DEFAULT_EN_STATE_BOUND;
    settings->en_input_bound = TINY_DEFAULT_EN_INPUT_BOUND;
}

#ifdef __cplusplus
}
#endif


// #include "tinympc/tiny_wrapper.hpp"

// extern "C"
// {
//     void set_x0(float *x0, int verbose)
//     {
//         for (int i = 0; i < tiny_data_solver.work->nx; i++)
//         {
//             tiny_data_solver.work->x(i, 0) = x0[i];
//         }

//         if (verbose != 0)
//         {
//             for (int i = 0; i < tiny_data_solver.work->nx; i++)
//             {
//                 printf("set_x0 result:  %f\n", tiny_data_solver.work->x(i, 0));
//             }
//         }
//     }

//     void set_xref(float *xref, int verbose)
//     {
//         for (int j = 0; j < tiny_data_solver.work->N; j++)
//         {
//             for (int i = 0; i < tiny_data_solver.work->nx; i++)
//             {
//                 tiny_data_solver.work->Xref(i, j) = xref[j * tiny_data_solver.work->nx + i];
//             }
//         }

//         if (verbose != 0)
//         {
//             for (int j = 0; j < tiny_data_solver.work->N; j++)
//             {
//                 for (int i = 0; i < tiny_data_solver.work->nx; i++)
//                 {
//                     printf("set_xref result:  %f\n", tiny_data_solver.work->Xref(i, j));
//                 }
//             }
//         }
//     }

//     void set_uref(float *xref, int verbose)
//     {
//         for (int j = 0; j < tiny_data_solver.work->N - 1; j++)
//         {
//             for (int i = 0; i < tiny_data_solver.work->nu; i++)
//             {
//                 tiny_data_solver.work->Uref(i, j) = xref[j * tiny_data_solver.work->nu + i];
//             }
//         }

//         if (verbose != 0)
//         {
//             for (int j = 0; j < tiny_data_solver.work->N - 1; j++)
//             {
//                 for (int i = 0; i < tiny_data_solver.work->nu; i++)
//                 {
//                     printf("set_xref result:  %f\n", tiny_data_solver.work->Uref(i, j));
//                 }
//             }
//         }
//     }

//     void set_umin(float *umin, int verbose)
//     {
//         for (int j = 0; j < tiny_data_solver.work->N - 1; j++)
//         {
//             for (int i = 0; i < tiny_data_solver.work->nu; i++)
//             {
//                 tiny_data_solver.work->u_min(i, j) = umin[j * tiny_data_solver.work->nu + i];
//             }
//         }

//         if (verbose != 0)
//         {
//             for (int j = 0; j < tiny_data_solver.work->N - 1; j++)
//             {
//                 for (int i = 0; i < tiny_data_solver.work->nu; i++)
//                 {
//                     printf("set_umin result:  %f\n", tiny_data_solver.work->u_min(i, j));
//                 }
//             }
//         }
//     }

//     void set_umax(float *umax, int verbose)
//     {
//         for (int j = 0; j < tiny_data_solver.work->N - 1; j++)
//         {
//             for (int i = 0; i < tiny_data_solver.work->nu; i++)
//             {
//                 tiny_data_solver.work->u_max(i, j) = umax[j * tiny_data_solver.work->nu + i];
//             }
//         }

//         if (verbose != 0)
//         {
//             for (int j = 0; j < tiny_data_solver.work->N - 1; j++)
//             {
//                 for (int i = 0; i < tiny_data_solver.work->nu; i++)
//                 {
//                     printf("set_umax result:  %f\n", tiny_data_solver.work->u_max(i, j));
//                 }
//             }
//         }
//     }

//     void set_xmin(float *xmin, int verbose)
//     {
//         for (int j = 0; j < tiny_data_solver.work->N; j++)
//         {
//             for (int i = 0; i < tiny_data_solver.work->nx; i++)
//             {
//                 tiny_data_solver.work->x_min(i, j) = xmin[j * tiny_data_solver.work->nx + i];
//             }
//         }

//         if (verbose != 0)
//         {
//             for (int j = 0; j < tiny_data_solver.work->N; j++)
//             {
//                 for (int i = 0; i < tiny_data_solver.work->nx; i++)
//                 {
//                     printf("set_xmin result:  %f\n", tiny_data_solver.work->x_min(i, j));
//                 }
//             }
//         }
//     }

//     void set_xmax(float *xmax, int verbose)
//     {
//         for (int j = 0; j < tiny_data_solver.work->N; j++)
//         {
//             for (int i = 0; i < tiny_data_solver.work->nx; i++)
//             {
//                 tiny_data_solver.work->x_max(i, j) = xmax[j * tiny_data_solver.work->nx + i];
//             }
//         }

//         if (verbose != 0)
//         {
//             for (int j = 0; j < tiny_data_solver.work->N; j++)
//             {
//                 for (int i = 0; i < tiny_data_solver.work->nx; i++)
//                 {
//                     printf("set_xmax result:  %f\n", tiny_data_solver.work->x_max(i, j));
//                 }
//             }
//         }
//     }

//     void reset_dual_variables(int verbose)
//     {
//         tiny_data_solver.work->y = tiny_MatrixNuNhm1::Zero();
//         tiny_data_solver.work->g = tiny_MatrixNxNh::Zero();

//         if (verbose != 0)
//         {
//             std::cout << "reset duals finished" << std::endl;
//         }
//     }

//     void call_tiny_solve(int verbose)
//     {
//         tiny_solve(&tiny_data_solver);

//         if (verbose != 0)
//         {
//             std::cout << "tiny solve finished" << std::endl;
//         }
//     }

//     void get_x(float *x_soln, int verbose)
//     {
//         Eigen::Map<tiny_MatrixNxNh>(x_soln, tiny_data_solver.work->x.rows(), tiny_data_solver.work->x.cols()) = tiny_data_solver.work->x;

//         if (verbose != 0)
//         {
//             for (int i = 0; i < tiny_data_solver.work->N; i++)
//             {
//                 for (int j = 0; j < tiny_data_solver.work->nx; j++) {
//                     printf("x_soln:  %f\n", x_soln[i*tiny_data_solver.work->nx + j]);
//                 }
//             }
//         }
//     }

//     void get_u(float *u_soln, int verbose)
//     {
//         Eigen::Map<tiny_MatrixNuNhm1>(u_soln, tiny_data_solver.work->u.rows(), tiny_data_solver.work->u.cols()) = tiny_data_solver.work->u;

//         if (verbose != 0)
//         {
//             for (int i = 0; i < tiny_data_solver.work->N - 1; i++)
//             {
//                 printf("u_soln:  %f\n", u_soln[i]);
//             }
//         }
//     }
// }
