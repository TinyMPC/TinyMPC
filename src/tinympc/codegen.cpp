#include <stdio.h>
#include <ctype.h>
#include <time.h>

#include <iostream>
#include <Eigen/Dense>

#include "types.hpp"
#include "codegen.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/* Define the maximum allowed length of a variable name */
#define MAX_VAR_LENGTH 255

/* Define the maximum allowed length of the path (directory + filename + extension) */
#define PATH_LENGTH 1024

/* Define the maximum allowed length of the filename (no extension)*/
#define FILE_LENGTH 100

using namespace Eigen;
IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

static void print_matrix(FILE *f, MatrixXf mat) {
    fprintf(f, "\t{\n");
    for (float x : mat.reshaped()) {
        fprintf(f, "\t\t(tinytype)%.20f,\n", x);
    }
    fprintf(f, "\t},\n");
}

static void print_vector(FILE *f, VectorXf mat) {
    fprintf(f, "\t{\n");
    for (float x : mat.reshaped()) {
        fprintf(f, "\t\t(tinytype)%.20f,\n", x);
    }
    fprintf(f, "\t},\n");
}

int tiny_codegen(int nx, int nu, int N,
                 tinytype *Adyn, tinytype *Bdyn, tinytype *Q, tinytype *Qf, tinytype *R,
                 tinytype *x_min, tinytype *x_max, tinytype *u_min, tinytype *u_max,
                 tinytype rho, tinytype abs_pri_tol, tinytype abs_dua_tol, int max_iters, int check_termination,
                 const char* output_dir, const char* file_prefix)
{
    TinyCache cache;
    TinySettings settings;
    TinyWorkspace work;
    TinySolver solver{&settings, &cache, &work};

    settings.abs_dua_tol = abs_dua_tol;
    settings.abs_pri_tol = abs_pri_tol;
    settings.check_termination = check_termination;
    settings.max_iter = max_iters;

    if (x_min != nullptr && x_max != nullptr)
    {
        settings.en_state_bound = 1;
    }
    else
    {
        settings.en_state_bound = 0;
    }
    if (u_min != nullptr && u_max != nullptr)
    {
        settings.en_input_bound = 1;
    }
    else
    {
        settings.en_input_bound = 0;
    }


    cache.rho = rho;
    work.Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn);
    work.Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn);
    work.Q = Map<tiny_VectorNx>(Q);
    work.Qf = Map<tiny_VectorNx>(Qf);
    work.R = Map<tiny_VectorNu>(R);
    work.x_min = Map<tiny_MatrixNxNh>(x_min); // x_min is col-major
    work.x_max = Map<tiny_MatrixNxNh>(x_max);
    work.u_min = Map<tiny_MatrixNuNhm1>(u_min); // u_min is col-major
    work.u_max = Map<tiny_MatrixNuNhm1>(u_max);

    // Update by adding rho * identity matrix to Q, Qf, R
    work.Q = work.Q + rho * tiny_VectorNx::Ones();
    work.Qf = work.Qf + rho * tiny_VectorNx::Ones();
    work.R = work.R + rho * tiny_VectorNu::Ones();
    tiny_MatrixNxNx Q1 = work.Q.array().matrix().asDiagonal();
    tiny_MatrixNxNx Qf1 = work.Qf.array().matrix().asDiagonal();
    tiny_MatrixNuNu R1 = work.R.array().matrix().asDiagonal();

    // Printing
    std::cout << "A = " << work.Adyn.format(CleanFmt) << std::endl;
    std::cout << "B = " << work.Bdyn.format(CleanFmt) << std::endl;
    std::cout << "Q = " << Q1.format(CleanFmt) << std::endl;
    std::cout << "Qf = " << Qf1.format(CleanFmt) << std::endl;
    std::cout << "R = " << R1.format(CleanFmt) << std::endl;
    std::cout << "rho = " << cache.rho << std::endl;

    // Riccati recursion to get Kinf, Pinf 
    tiny_MatrixNuNx Ktp1;
    tiny_MatrixNxNx Ptp1 = Qf1;
    Ktp1 = tiny_MatrixNuNx::Zero();

    for (int i = 0; i < 100; i++)
    {
        cache.Kinf = (R1 + work.Bdyn.transpose() * Ptp1 * work.Bdyn).inverse() * work.Bdyn.transpose() * Ptp1 * work.Adyn;
        cache.Pinf = Q1 + work.Adyn.transpose() * Ptp1 * (work.Adyn - work.Bdyn * cache.Kinf);
        // if Kinf converges, break
        if ((cache.Kinf - Ktp1).cwiseAbs().maxCoeff() < 1e-6)
        {
            std::cout << "Kinf converged after " << i+1 << " iterations" << std::endl;
            break;
        }
        Ktp1 = cache.Kinf;
        Ptp1 = cache.Pinf;
    }

    // Compute cached matrices
    cache.Quu_inv = (R1 + work.Bdyn.transpose() * cache.Pinf * work.Bdyn).inverse();
    cache.AmBKt = (work.Adyn - work.Bdyn * cache.Kinf).transpose();
    cache.coeff_d2p = cache.Kinf.transpose() * R1 - cache.AmBKt * cache.Pinf * work.Bdyn;

    std::cout << "Kinf = " << cache.Kinf.format(CleanFmt) << std::endl;
    std::cout << "Pinf = " << cache.Pinf.format(CleanFmt) << std::endl;
    std::cout << "Quu_inv = " << cache.Quu_inv.format(CleanFmt) << std::endl;
    std::cout << "AmBKt = " << cache.AmBKt.format(CleanFmt) << std::endl;
    std::cout << "coeff_d2p = " << cache.coeff_d2p.format(CleanFmt) << std::endl;

    // TODO(sschoedel): Write to files (check OSQP for references https://github.com/osqp/osqp/blob/master/src/codegen.c)
    // Write caches
    // Write settings
    // Write workspace
    // Write solver

    // Codegen workspace file
    char workspace_fname[PATH_LENGTH], workspace_cfname[PATH_LENGTH];
    FILE *data_f;
    time_t start_time;

    sprintf(workspace_fname, "%s%sworkspace", output_dir, file_prefix);
    sprintf(workspace_cfname, "%s.h", workspace_fname);

    // Open source file
    data_f = fopen(workspace_cfname, "w+");
    // data_f = fopen("/tmp/test.c", "w+");
    if (data_f == NULL)
        printf("ERROR OPENING FILE\n");
        // return tiny_error(TINY_FOPEN_ERROR);


    time(&start_time);
    fprintf(data_f, "/*\n");
    fprintf(data_f, " * This file was autogenerated by TinyMPC on %s", ctime(&start_time));
    fprintf(data_f, " * \n");
    fprintf(data_f, " * This file contains a sample solver to run the embedded code.\n");
    fprintf(data_f, " */\n\n");

    // Write cache to source file
    fprintf(data_f, "TinyCache tinycache = {\n");
    fprintf(data_f, "\t(tinytype)%.20f,\n", cache.rho);
    print_matrix(data_f, cache.Kinf);
    print_matrix(data_f, cache.Pinf);
    print_matrix(data_f, cache.Quu_inv);
    print_matrix(data_f, cache.AmBKt);
    print_matrix(data_f, cache.coeff_d2p);
    fprintf(data_f, "};\n\n");

    // Write settings to source file
    fprintf(data_f, "TinySettings tinysettings = {\n");
    fprintf(data_f, "\t(tinytype)%.20f,\n", settings.abs_pri_tol);
    fprintf(data_f, "\t(tinytype)%.20f,\n", settings.abs_dua_tol);
    fprintf(data_f, "\t%d,\n", settings.max_iter);
    fprintf(data_f, "\t%d,\n", settings.check_termination);
    fprintf(data_f, "\t%d,\n", settings.en_state_bound);
    fprintf(data_f, "\t%d\n", settings.en_input_bound);
    fprintf(data_f, "};\n\n");

    // Write workspace (problem variables) to source file
    fprintf(data_f, "TinyProblem tinyproblem = {\n");

    print_matrix(data_f, tiny_MatrixNxNh::Zero()); // x
    print_matrix(data_f, tiny_MatrixNuNhm1::Zero()); // u

    print_matrix(data_f, tiny_MatrixNxNh::Zero()); // q
    print_matrix(data_f, tiny_MatrixNuNhm1::Zero()); // r

    print_matrix(data_f, tiny_MatrixNxNh::Zero()); // p
    print_matrix(data_f, tiny_MatrixNuNhm1::Zero()); // d

    print_matrix(data_f, tiny_MatrixNxNh::Zero()); // v
    print_matrix(data_f, tiny_MatrixNxNh::Zero()); // vnew
    print_matrix(data_f, tiny_MatrixNuNhm1::Zero()); // z
    print_matrix(data_f, tiny_MatrixNuNhm1::Zero()); // znew

    print_matrix(data_f, tiny_MatrixNxNh::Zero()); // g
    print_matrix(data_f, tiny_MatrixNuNhm1::Zero()); // y

    fprintf(data_f, "\t(tinytype)%.20f,\n", work.primal_residual_state);
    fprintf(data_f, "\t(tinytype)%.20f,\n", work.primal_residual_input);
    fprintf(data_f, "\t(tinytype)%.20f,\n", work.dual_residual_state);
    fprintf(data_f, "\t(tinytype)%.20f,\n", work.dual_residual_input);
    fprintf(data_f, "\t%d,\n", work.status);
    fprintf(data_f, "\t%d,\n", work.iter);

    print_vector(data_f, work.Q);
    print_vector(data_f, work.Qf);
    print_vector(data_f, work.R);
    print_matrix(data_f, work.Adyn);
    print_matrix(data_f, work.Bdyn);

    print_matrix(data_f, work.u_min);
    print_matrix(data_f, work.u_max);
    print_matrix(data_f, work.x_min);
    print_matrix(data_f, work.x_max);
    print_matrix(data_f, work.Xref);
    print_matrix(data_f, work.Uref);

    print_vector(data_f, work.Qu);
    fprintf(data_f, "};\n\n");

    // Close codegen data file
    fclose(data_f);
    printf("workspace generated in %s\n", workspace_cfname);

    // Write to 

    // TODO: add error codes
    return 1;
}


#ifdef __cplusplus
}
#endif