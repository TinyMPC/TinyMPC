#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

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

int tiny_codegen(const int nx, const int nu, const int N,
                 tinytype *Adyn_data, tinytype *Bdyn_data, tinytype *Q_data, tinytype *Qf_data, tinytype *R_data,
                 tinytype *x_min_data, tinytype *x_max_data, tinytype *u_min_data, tinytype *u_max_data,
                 tinytype rho, tinytype abs_pri_tol, tinytype abs_dua_tol, int max_iters, int check_termination,
                 const char* tinympc_dir, const char* output_dir)
{
    int en_state_bound = 0;
    int en_input_bound = 0;

    if (x_min_data != nullptr && x_max_data != nullptr) {
        en_state_bound = 1;
    }
    else {
        en_state_bound = 0;
    }

    if (u_min_data != nullptr && u_max_data != nullptr) {
        en_input_bound = 1;
    }
    else {
        en_input_bound = 0;
    }


    MatrixXf Adyn = MatrixXf::Map(Adyn_data, nx, nx);
    MatrixXf Bdyn = MatrixXf::Map(Bdyn_data, nx, nu);
    MatrixXf Q = MatrixXf::Map(Q_data, nx, 1);
    MatrixXf Qf = MatrixXf::Map(Qf_data, nx, 1);
    MatrixXf R = MatrixXf::Map(R_data, nx, 1);
    MatrixXf x_min = MatrixXf::Map(x_min_data, N, nx).transpose(); // x_min is col-major
    MatrixXf x_max = MatrixXf::Map(x_max_data, N, nx).transpose();
    MatrixXf u_min = MatrixXf::Map(u_min_data, N-1, nu).transpose(); // u_min is col-major
    MatrixXf u_max = MatrixXf::Map(u_max_data, N-1, nu).transpose();

    // Update by adding rho * identity matrix to Q, Qf, R
    Q = Q + rho * MatrixXf::Ones(nx, 1);
    Qf = Qf + rho * MatrixXf::Ones(nx, 1);
    R = R + rho * MatrixXf::Ones(nx, 1);
    MatrixXf Q1 = Q.array().matrix().asDiagonal();
    MatrixXf Qf1 = Qf.array().matrix().asDiagonal();
    MatrixXf R1 = R.array().matrix().asDiagonal();

    // Printing
    std::cout << "A = " << Adyn.format(CleanFmt) << std::endl;
    std::cout << "B = " << Bdyn.format(CleanFmt) << std::endl;
    std::cout << "Q = " << Q1.format(CleanFmt) << std::endl;
    std::cout << "Qf = " << Qf1.format(CleanFmt) << std::endl;
    std::cout << "R = " << R1.format(CleanFmt) << std::endl;
    std::cout << "rho = " << rho << std::endl;

    // Riccati recursion to get Kinf, Pinf 
    MatrixXf Ktp1 = MatrixXf::Zero(nu, nx);
    MatrixXf Ptp1 = Qf1;
    MatrixXf Kinf = MatrixXf::Zero(nu, nx);
    MatrixXf Pinf = MatrixXf::Zero(nx, nx);

    for (int i = 0; i < 100; i++)
    {
        Kinf = (R1 + Bdyn.transpose() * Ptp1 * Bdyn).inverse() * Bdyn.transpose() * Ptp1 * Adyn;
        Pinf = Q1 + Adyn.transpose() * Ptp1 * (Adyn - Bdyn * Kinf);
        // if Kinf converges, break
        if ((Kinf - Ktp1).cwiseAbs().maxCoeff() < 1e-6)
        {
            std::cout << "Kinf converged after " << i+1 << " iterations" << std::endl;
            break;
        }
        Ktp1 = Kinf;
        Ptp1 = Pinf;
    }

    // Compute cached matrices
    MatrixXf Quu_inv = (R1 + Bdyn.transpose() * Pinf * Bdyn).inverse();
    MatrixXf AmBKt = (Adyn - Bdyn * Kinf).transpose();
    MatrixXf coeff_d2p = Kinf.transpose() * R1 - AmBKt * Pinf * Bdyn;

    std::cout << "Kinf = " << Kinf.format(CleanFmt) << std::endl;
    std::cout << "Pinf = " << Pinf.format(CleanFmt) << std::endl;
    std::cout << "Quu_inv = " << Quu_inv.format(CleanFmt) << std::endl;
    std::cout << "AmBKt = " << AmBKt.format(CleanFmt) << std::endl;
    std::cout << "coeff_d2p = " << coeff_d2p.format(CleanFmt) << std::endl;


    // Make code gen output directory structure
    char workspace_dname[PATH_LENGTH];
    char workspace_src_dname[PATH_LENGTH];
    char workspace_tinympc_dname[PATH_LENGTH];
    char workspace_include_dname[PATH_LENGTH];

    sprintf(workspace_dname, "%s%s", tinympc_dir, output_dir);
    sprintf(workspace_src_dname, "%s/src", workspace_dname);
    sprintf(workspace_tinympc_dname, "%s/tinympc", workspace_dname);
    sprintf(workspace_include_dname, "%s/include", workspace_dname);

    struct stat st = {0};

    if (stat(workspace_dname, &st) == -1) {
        printf("Creating generated code directory at %s\n", workspace_dname);
        mkdir(workspace_dname, 0700);           // workspace/
        mkdir(workspace_src_dname, 0700);       // workspace/src
        mkdir(workspace_tinympc_dname, 0700);   // workspace/tinympc
        mkdir(workspace_include_dname, 0700);   // workspace/include
    }

    // Codegen workspace file
    char data_workspace_fname[PATH_LENGTH], glob_opts_fname[PATH_LENGTH];
    FILE *data_f, *glob_opts_f;
    time_t start_time;

    sprintf(data_workspace_fname, "%s/tiny_data_workspace.cpp", workspace_src_dname);
    sprintf(glob_opts_fname, "%s/glob_opts.hpp", workspace_tinympc_dname);

    // Open source file
    data_f = fopen(data_workspace_fname, "w+");
    if (data_f == NULL)
        printf("ERROR OPENING DATA WORKSPACE FILE\n");
        // return tiny_error(TINY_FOPEN_ERROR);

    // Preamble
    time(&start_time);
    fprintf(data_f, "/*\n");
    fprintf(data_f, " * This file was autogenerated by TinyMPC on %s", ctime(&start_time));
    fprintf(data_f, " */\n\n");

    // Open extern C
    fprintf(data_f, "#include <tinympc/tiny_data_workspace.hpp>\n\n");
    fprintf(data_f, "#ifdef __cplusplus\n");
    fprintf(data_f, "extern \"C\" {\n");
    fprintf(data_f, "#endif\n\n");

    // Write settings to workspace file
    fprintf(data_f, "/* User settings */\n");
    fprintf(data_f, "TinySettings settings = {\n");
    fprintf(data_f, "\t(tinytype)%.16f,\t// primal tolerance\n", abs_pri_tol);
    fprintf(data_f, "\t(tinytype)%.16f,\t// dual tolerance\n", abs_dua_tol);
    fprintf(data_f, "\t%d,\t\t// max iterations\n", max_iters);
    fprintf(data_f, "\t%d,\t\t// iterations per termination check\n", check_termination);
    fprintf(data_f, "\t%d,\t\t// enable state constraints\n", en_state_bound);
    fprintf(data_f, "\t%d\t\t// enable input constraints\n", en_input_bound);
    fprintf(data_f, "};\n\n");
    
    // Write cache to workspace file
    fprintf(data_f, "/* Matrices that must be recomputed with changes in time step, rho */\n");
    fprintf(data_f, "TinyCache cache = {\n");
    fprintf(data_f, "\t(tinytype)%.16f,\t// rho (step size/penalty)\n", rho);
    fprintf(data_f, "\t(tiny_MatrixNuNx() << "); print_matrix(data_f, Kinf, nu*nx); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNx() << "); print_matrix(data_f, Pinf, nx*nx); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNuNu() << "); print_matrix(data_f, Quu_inv, nu*nu); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNu() << "); print_matrix(data_f, AmBKt, nx*nu); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNu() << "); print_matrix(data_f, coeff_d2p, nx*nu); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "};\n\n");

    // Write workspace (problem variables) to workspace file
    fprintf(data_f, "/* Problem variables */\n");
    fprintf(data_f, "TinyWorkspace work = {\n");

    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, MatrixXf::Zero(nx, N), nx*N); fprintf(data_f, ").finished(),\n"); // x
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, MatrixXf::Zero(nu, N-1), nu*(N-1)); fprintf(data_f, ").finished(),\n"); // u

    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, MatrixXf::Zero(nx, N), nx*N); fprintf(data_f, ").finished(),\n"); // q
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, MatrixXf::Zero(nu, N-1), nu*(N-1)); fprintf(data_f, ").finished(),\n"); // r

    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, MatrixXf::Zero(nx, N), nx*N); fprintf(data_f, ").finished(),\n"); // p
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, MatrixXf::Zero(nu, N-1), nu*(N-1)); fprintf(data_f, ").finished(),\n"); // d

    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, MatrixXf::Zero(nx, N), nx*N); fprintf(data_f, ").finished(),\n"); // v
    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, MatrixXf::Zero(nx, N), nx*N); fprintf(data_f, ").finished(),\n"); // vnew
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, MatrixXf::Zero(nu, N-1), nu*(N-1)); fprintf(data_f, ").finished(),\n"); // z
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, MatrixXf::Zero(nu, N-1), nu*(N-1)); fprintf(data_f, ").finished(),\n"); // znew

    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, MatrixXf::Zero(nx, N), nx*N); fprintf(data_f, ").finished(),\n"); // g
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, MatrixXf::Zero(nu, N-1), nu*(N-1)); fprintf(data_f, ").finished(),\n"); // u

    fprintf(data_f, "\t(tinytype)%.16f,\t// state primal residual\n", 0.0);
    fprintf(data_f, "\t(tinytype)%.16f,\t// input primal residual\n", 0.0);
    fprintf(data_f, "\t(tinytype)%.16f,\t// state dual residual\n", 0.0);
    fprintf(data_f, "\t(tinytype)%.16f,\t// input dual residual\n", 0.0);
    fprintf(data_f, "\t%d,\t// solve status\n", 0);
    fprintf(data_f, "\t%d,\t// solve iteration\n", 0);

    fprintf(data_f, "\t(tiny_VectorNx() << "); print_matrix(data_f, Q, nx); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_VectorNx() << "); print_matrix(data_f, Qf, nx); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_VectorNu() << "); print_matrix(data_f, R, nu); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNx() << "); print_matrix(data_f, Adyn, nx*nx); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNu() << "); print_matrix(data_f, Bdyn, nx*nu); fprintf(data_f, ").finished(),\n");

    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, u_min, nu*(N-1)); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, u_max, nu*(N-1)); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, x_min, nx*N); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, x_max, nx*N); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNxNh() << "); print_matrix(data_f, MatrixXf::Zero(nx, N), nx*N); fprintf(data_f, ").finished(),\n");
    fprintf(data_f, "\t(tiny_MatrixNuNhm1() << "); print_matrix(data_f, MatrixXf::Zero(nu, N-1), nu*(N-1)); fprintf(data_f, ").finished(),\n");

    fprintf(data_f, "\t(tiny_VectorNu() << "); print_matrix(data_f, MatrixXf::Zero(nu, 1), nu); fprintf(data_f, ").finished()\n");
    fprintf(data_f, "};\n\n");

    // Write solver struct definition to workspace file
    fprintf(data_f, "TinySolver tiny_data_solver = {&settings, &cache, &work};\n\n");

    // Close extern C
    fprintf(data_f, "#ifdef __cplusplus\n");
    fprintf(data_f, "}\n");
    fprintf(data_f, "#endif\n\n");

    // Close codegen data file
    fclose(data_f);
    printf("Data generated in %s\n", data_workspace_fname);

    // Open global options file
    glob_opts_f = fopen(glob_opts_fname, "w+");
    if (glob_opts_f == NULL)
        printf("ERROR OPENING GLOBAL OPTIONS FILE\n");
        // return tiny_error(TINY_FOPEN_ERROR);

    // Preamble
    time(&start_time);
    fprintf(data_f, "/*\n");
    fprintf(data_f, " * This file was autogenerated by TinyMPC on %s", ctime(&start_time));
    fprintf(data_f, " */\n\n");
    
    // Write global options
    fprintf(glob_opts_f, "#pragma once\n\n");
    fprintf(glob_opts_f, "typedef float tinytype;\n\n");
    fprintf(glob_opts_f, "#define NSTATES %d\n", nx);
    fprintf(glob_opts_f, "#define NINPUTS %d\n", nu);
    fprintf(glob_opts_f, "#define NHORIZON %d", N);
    
    // Close codegen global options file
    fclose(glob_opts_f);
    printf("Global options generated in %s\n", glob_opts_fname);



    // // Copy solver header into codegen output directory
    // char solver_src_cfname[PATH_LENGTH];
    // char solver_dst_cfname[PATH_LENGTH];
    // FILE *src_f, *dst_f;
    // size_t in, out;
    // char buf[BUF_SIZE];

    // sprintf(solver_src_cfname, "%ssrc/tinympc/admm.hpp", tinympc_dir);
    // sprintf(solver_dst_cfname, "%s%sadmm.hpp", tinympc_dir, output_dir);

    // src_f = fopen(solver_src_cfname, "r");
    // if (src_f == NULL)
    //     printf("ERROR OPENING SOLVER SOURCE FILE\n");
    //     // return tiny_error(TINY_FOPEN_ERROR);
    // dst_f = fopen(solver_dst_cfname, "w+");
    // if (dst_f == NULL)
    //     printf("ERROR OPENING SOLVER DESTINATION FILE\n");
    //     // return tiny_error(TINY_FOPEN_ERROR);


    // // Copy contents of solver to code generated solver file
    // while (1) {
    //     in = fread(buf, 1, BUF_SIZE, src_f);
    //     if (in == 0) break;
    //     out = fread(buf, 1, in, dst_f);
    //     if (out == 0); break;
    // }

    // fclose(src_f);
    // fclose(dst_f);

    // TODO: add error codes
    return 1;
}


#ifdef __cplusplus
}
#endif