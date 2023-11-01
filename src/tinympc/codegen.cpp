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

#define BUF_SIZE 65536

using namespace Eigen;
IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

static void print_matrix(FILE *f, MatrixXf mat, int num_elements) {
    for (int i=0; i<num_elements; i++) {
        fprintf(f, "(tinytype)%.16f", mat.reshaped()[i]);
        if (i < num_elements-1)
            fprintf(f, ",");
    }
    // for (tinytype x : mat.reshaped()) {
    //     fprintf(f, "(tinytype)%.20f, ", x);
    // }
}

static void print_vector(FILE *f, VectorXf mat, int num_elements) {
    for (int i=0; i<num_elements; i++) {
        fprintf(f, "(tinytype)%.16f", mat.reshaped()[i]);
        if (i < num_elements-1)
            fprintf(f, ",");
    }
    // for (tinytype x : mat.reshaped()) {
    //     fprintf(f, "\t\t(tinytype)%.20f,\n", x);
    // }
}

int tiny_codegen(int nx, int nu, int N,
                 tinytype *Adyn, tinytype *Bdyn, tinytype *Q, tinytype *Qf, tinytype *R,
                 tinytype *x_min, tinytype *x_max, tinytype *u_min, tinytype *u_max,
                 tinytype rho, tinytype abs_pri_tol, tinytype abs_dua_tol, int max_iters, int check_termination,
                 const char* tinympc_dir, const char* output_dir)
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

    sprintf(workspace_fname, "%s%stiny_data_workspace", tinympc_dir, output_dir);
    sprintf(workspace_cfname, "%s.cpp", workspace_fname);

    // Open source file
    data_f = fopen(workspace_cfname, "w+");
    if (data_f == NULL)
        printf("ERROR OPENING FILE\n");
        // return tiny_error(TINY_FOPEN_ERROR);

    time(&start_time);
    fprintf(data_f, "/*\n");
    fprintf(data_f, " * This file was autogenerated by TinyMPC on %s", ctime(&start_time));
    fprintf(data_f, " * \n");
    fprintf(data_f, " * This file contains a sample solver to run the embedded code.\n");
    fprintf(data_f, " */\n\n");

    // Open extern C
    fprintf(data_f, "#include <tinympc/tiny_data_workspace.hpp>\n\n");
    fprintf(data_f, "#ifdef __cplusplus\n");
    fprintf(data_f, "extern \"C\" {\n");
    fprintf(data_f, "#endif\n\n");

    // Write settings to workspace file
    fprintf(data_f, "/* User settings */\n");
    fprintf(data_f, "TinySettings settings = {\n");
    fprintf(data_f, "\t(tinytype)%.16f,\t// primal tolerance\n", settings.abs_pri_tol);
    fprintf(data_f, "\t(tinytype)%.16f,\t// dual tolerance\n", settings.abs_dua_tol);
    fprintf(data_f, "\t%d,\t\t// max iterations\n", settings.max_iter);
    fprintf(data_f, "\t%d,\t\t// iterations per termination check\n", settings.check_termination);
    fprintf(data_f, "\t%d,\t\t// enable state constraints\n", settings.en_state_bound);
    fprintf(data_f, "\t%d\t\t// enable input constraints\n", settings.en_input_bound);
    fprintf(data_f, "};\n\n");
    
    // Write cache to workspace file
    fprintf(data_f, "/* Matrices that must be recomputed with changes in time step, rho */\n");
    fprintf(data_f, "TinyCache cache = {\n");
    fprintf(data_f, "\t(tinytype)%.16f,\t// rho (step size/penalty)\n", cache.rho);
    fprintf(data_f, "\t(tiny_MatrixNuNx() << "); print_matrix(data_f, cache.Kinf, NINPUTS*NSTATES); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNx() << "); print_matrix(data_f, cache.Pinf, NSTATES*NSTATES); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNuNu() << "); print_matrix(data_f, cache.Quu_inv, NINPUTS*NINPUTS); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNu() << "); print_matrix(data_f, cache.AmBKt, NSTATES*NINPUTS); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNu() << "); print_matrix(data_f, cache.coeff_d2p, NSTATES*NINPUTS); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "};\n\n");

    // Write workspace (problem variables) to workspace file
    fprintf(data_f, "/* Problem variables */\n");
    fprintf(data_f, "TinyWorkspace work = {\n");

    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, tiny_MatrixNxNh::Zero(), NSTATES*NHORIZON); fprintf(data_f, ").finished(),\n"); // x
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, tiny_MatrixNuNhm1::Zero(), NINPUTS*(NHORIZON-1)); fprintf(data_f, ").finished(),\n"); // u

    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, tiny_MatrixNxNh::Zero(), NSTATES*NHORIZON); fprintf(data_f, ").finished(),\n"); // q
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, tiny_MatrixNuNhm1::Zero(), NINPUTS*(NHORIZON-1)); fprintf(data_f, ").finished(),\n"); // r

    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, tiny_MatrixNxNh::Zero(), NSTATES*NHORIZON); fprintf(data_f, ").finished(),\n"); // p
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, tiny_MatrixNuNhm1::Zero(), NINPUTS*(NHORIZON-1)); fprintf(data_f, ").finished(),\n"); // d

    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, tiny_MatrixNxNh::Zero(), NSTATES*NHORIZON); fprintf(data_f, ").finished(),\n"); // v
    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, tiny_MatrixNxNh::Zero(), NSTATES*NHORIZON); fprintf(data_f, ").finished(),\n"); // vnew
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, tiny_MatrixNuNhm1::Zero(), NINPUTS*(NHORIZON-1)); fprintf(data_f, ").finished(),\n"); // z
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, tiny_MatrixNuNhm1::Zero(), NINPUTS*(NHORIZON-1)); fprintf(data_f, ").finished(),\n"); // znew

    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, tiny_MatrixNxNh::Zero(), NSTATES*NHORIZON); fprintf(data_f, ").finished(),\n"); // g
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, tiny_MatrixNuNhm1::Zero(), NINPUTS*(NHORIZON-1)); fprintf(data_f, ").finished(),\n"); // y

    fprintf(data_f, "\t(tinytype)%.16f,\t// state primal residual\n", work.primal_residual_state);
    fprintf(data_f, "\t(tinytype)%.16f,\t// input primal residual\n", work.primal_residual_input);
    fprintf(data_f, "\t(tinytype)%.16f,\t// state dual residual\n", work.dual_residual_state);
    fprintf(data_f, "\t(tinytype)%.16f,\t// input dual residual\n", work.dual_residual_input);
    fprintf(data_f, "\t%d,\t// solve status\n", work.status);
    fprintf(data_f, "\t%d,\t// solve iteration\n", work.iter);

    fprintf(data_f, "\t(tiny_VectorNx() << "); print_vector(data_f, work.Q, NSTATES); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_VectorNx() << "); print_vector(data_f, work.Qf, NSTATES); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_VectorNu() << "); print_vector(data_f, work.R, NINPUTS); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNx() << "); print_matrix(data_f, work.Adyn, NSTATES*NSTATES); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNu() << "); print_matrix(data_f, work.Bdyn, NSTATES*NINPUTS); fprintf(data_f, ").finished(),\n");

    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, work.u_min, NINPUTS*(NHORIZON-1)); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, work.u_max, NINPUTS*(NHORIZON-1)); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, work.x_min, NSTATES*NHORIZON); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, work.x_max, NSTATES*NHORIZON); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, work.Xref, NSTATES*NHORIZON); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, work.Uref, NINPUTS*(NHORIZON-1)); fprintf(data_f, ").finished(),\n");

    fprintf(data_f, "\t(tiny_VectorNu() << "); print_vector(data_f, work.Qu, NINPUTS); fprintf(data_f, ").finished()\n");
    fprintf(data_f, "};\n\n");

    // Write solver struct definition to workspace file
    fprintf(data_f, "TinySolver tiny_data_solver = {&settings, &cache, &work};\n\n");

    // Close extern C
    fprintf(data_f, "#ifdef __cplusplus\n");
    fprintf(data_f, "}\n");
    fprintf(data_f, "#endif\n\n");

    // Close codegen data file
    fclose(data_f);
    printf("workspace generated in %s\n", workspace_cfname);

    // Copy solver header into codegen output directory
    char solver_src_cfname[PATH_LENGTH];
    char solver_dst_cfname[PATH_LENGTH];
    FILE *src_f, *dst_f;
    size_t in, out;
    char buf[BUF_SIZE];

    sprintf(solver_src_cfname, "%ssrc/tinympc/admm.hpp", tinympc_dir);
    sprintf(solver_dst_cfname, "%s%sadmm.hpp", tinympc_dir, output_dir);

    src_f = fopen(solver_src_cfname, "r");
    if (src_f == NULL)
        printf("ERROR OPENING SOLVER SOURCE FILE\n");
        // return tiny_error(TINY_FOPEN_ERROR);
    dst_f = fopen(solver_dst_cfname, "w+");
    if (dst_f == NULL)
        printf("ERROR OPENING SOLVER DESTINATION FILE\n");
        // return tiny_error(TINY_FOPEN_ERROR);


    // Copy contents of solver to code generated solver file
    while (1) {
        in = fread(buf, 1, BUF_SIZE, src_f);
        if (in == 0) break;
        out = fread(buf, 1, in, dst_f);
        if (out == 0); break;
    }

    fclose(src_f);
    fclose(dst_f);

    // TODO: add error codes
    return 1;
}


#ifdef __cplusplus
}
#endif