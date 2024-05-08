#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <Eigen/Dense>

// #include "types.hpp"
#include "codegen.hpp"

#ifdef __cplusplus
extern "C"
{
#endif

/* Define the maximum allowed length of the path (directory + filename + extension) */
#define PATH_LENGTH 2048

/* Define the maximum allowed length of the dirname */
#define DIR_NAME_LENGTH 100

/* Define the maximum allowed length of the filename (no extension)*/
#define FILE_NAME_LENGTH 100

#define BUF_SIZE 65536

using namespace Eigen;
IOFormat TinyFmt(4, 0, ", ", "\n", "[", "]");

// static void copy_file(char *src_file_name, char *dest_file_name)
// {
//     FILE *src_f, *dst_f;

//     src_f = fopen(src_file_name, "r");
//     if (src_f == NULL)
//         printf("ERROR OPENING SOLVER SOURCE FILE\n");
//     // return tiny_error(TINY_FOPEN_ERROR);
//     dst_f = fopen(dest_file_name, "w+");
//     if (dst_f == NULL)
//         printf("ERROR OPENING SOLVER DESTINATION FILE\n");
//     // return tiny_error(TINY_FOPEN_ERROR);

//     char c;

//     // Copy contents of solver to code generated solver file
//     c = fgetc(src_f);
//     while (c != EOF)
//     {
//         fputc(c, dst_f);
//         c = fgetc(src_f);
//     }

//     fclose(src_f);
//     fclose(dst_f);
// }

// static void copy_dir(char *src_dir_name, char *dest_dir_name)
// {
//     // Open folder at given directory path
//     DIR *src_dir = opendir(src_dir_name);
//     if (src_dir == NULL)
//     {
//         printf("SOURCE DIRECTORY NAME DOES NOT EXIST\n");
//         return;
//     }
//     struct dirent *dir_entry;

//     // printf("reading source directory %s\n", src_dir_name);

//     char dir_entry_full_name[PATH_LENGTH + DIR_NAME_LENGTH];
//     char codegen_dest_file_name[PATH_LENGTH + DIR_NAME_LENGTH];
//     struct stat buffer;
//     while (dir_entry = readdir(src_dir))
//     {
//         // printf("dir_entry->d_name: %s\n", dir_entry->d_name);

//         sprintf(dir_entry_full_name, "%s/%s", src_dir_name, dir_entry->d_name);
//         stat(dir_entry_full_name, &buffer);
//         // printf("dir_entry_full_name: %s\n", dir_entry_full_name);

//         sprintf(codegen_dest_file_name, "%s/%s", dest_dir_name, dir_entry->d_name);
//         // printf("codegen_dest_file_name: %s\n", codegen_dest_file_name);

//         // Check if file is directory or not
//         if (S_ISDIR(buffer.st_mode) && strcmp(dir_entry->d_name, "..") != 0 && strcmp(dir_entry->d_name, ".") != 0)
//         {
//             // If directory entry is another directory, create that folder
//             //   in the codegen directory and call copy_dir again
//             // printf("%s is a directory\n", dir_entry->d_name);
//             struct stat st = {0};
//             if (stat(codegen_dest_file_name, &st) == -1)
//             {
//                 // printf("Creating new folder %s\n", codegen_dest_file_name);
//                 mkdir(codegen_dest_file_name, 0700);
//             }
//             copy_dir(dir_entry_full_name, codegen_dest_file_name);
//         }
//         else if (!S_ISDIR(buffer.st_mode))
//         {
//             // Otherwise, copy file into code gen directory
//             // printf("%s is not a directory\n", dir_entry->d_name);
//             copy_file(dir_entry_full_name, codegen_dest_file_name);
//         }
//     }

//     closedir(src_dir);
// }

// int tiny_codegen(const int nx, const int nu, const int N,
//                     double *Adyn_data, double *Bdyn_data, double *Q_data, double *R_data,
//                     double *x_min_data, double *x_max_data, double *u_min_data, double *u_max_data,
//                     double rho, double abs_pri_tol, double abs_dua_tol, int max_iters, int check_termination, int gen_wrapper,
//                     const char *tinympc_dir, const char *output_dir)
// {
//     int en_state_bound = 0;
//     int en_input_bound = 0;

//     if (x_min_data != nullptr && x_max_data != nullptr)
//     {
//         en_state_bound = 1;
//     }
//     else
//     {
//         en_state_bound = 0;
//     }

//     if (u_min_data != nullptr && u_max_data != nullptr)
//     {
//         en_input_bound = 1;
//     }
//     else
//     {
//         en_input_bound = 0;
//     }

//     MatrixXd Adyn = MatrixXd::Map(Adyn_data, nx, nx);
//     MatrixXd Bdyn = MatrixXd::Map(Bdyn_data, nx, nu);
//     MatrixXd Q = MatrixXd::Map(Q_data, nx, 1);
//     MatrixXd R = MatrixXd::Map(R_data, nu, 1);
//     MatrixXd x_min = MatrixXd::Map(x_min_data, nx, N);
//     MatrixXd x_max = MatrixXd::Map(x_max_data, nx, N);
//     MatrixXd u_min = MatrixXd::Map(u_min_data, nu, N - 1);
//     MatrixXd u_max = MatrixXd::Map(u_max_data, nu, N - 1);

//     // Update by adding rho * identity matrix to Q, R
//     Q = Q + rho * MatrixXd::Ones(nx, 1);
//     R = R + rho * MatrixXd::Ones(nu, 1);
//     MatrixXd Q1 = Q.array().matrix().asDiagonal();
//     MatrixXd R1 = R.array().matrix().asDiagonal();

//     // Printing
//     std::cout << "A = " << Adyn.format(TinyFmt) << std::endl;
//     std::cout << "B = " << Bdyn.format(TinyFmt) << std::endl;
//     std::cout << "Q = " << Q1.format(TinyFmt) << std::endl;
//     std::cout << "R = " << R1.format(TinyFmt) << std::endl;
//     std::cout << "rho = " << rho << std::endl;

//     // Riccati recursion to get Kinf, Pinf
//     MatrixXd Ktp1 = MatrixXd::Zero(nu, nx);
//     MatrixXd Ptp1 = rho * MatrixXd::Ones(nx, 1).array().matrix().asDiagonal();
//     MatrixXd Kinf = MatrixXd::Zero(nu, nx);
//     MatrixXd Pinf = MatrixXd::Zero(nx, nx);

//     for (int i = 0; i < 1000; i++)
//     {
//         Kinf = (R1 + Bdyn.transpose() * Ptp1 * Bdyn).inverse() * Bdyn.transpose() * Ptp1 * Adyn;
//         Pinf = Q1 + Adyn.transpose() * Ptp1 * (Adyn - Bdyn * Kinf);
//         // if Kinf converges, break
//         if ((Kinf - Ktp1).cwiseAbs().maxCoeff() < 1e-5)
//         {
//             std::cout << "Kinf converged after " << i + 1 << " iterations" << std::endl;
//             break;
//         }
//         Ktp1 = Kinf;
//         Ptp1 = Pinf;
//     }

//     std::cout << "Precomputing finished" << std::endl;

//     // Compute cached matrices
//     MatrixXd Quu_inv = (R1 + Bdyn.transpose() * Pinf * Bdyn).inverse();
//     MatrixXd AmBKt = (Adyn - Bdyn * Kinf).transpose();

//     std::cout << "Kinf = " << Kinf.format(TinyFmt) << std::endl;
//     std::cout << "Pinf = " << Pinf.format(TinyFmt) << std::endl;
//     std::cout << "Quu_inv = " << Quu_inv.format(TinyFmt) << std::endl;
//     std::cout << "AmBKt = " << AmBKt.format(TinyFmt) << std::endl;

//     // Make code gen output directory structure
//     char workspace_dname[PATH_LENGTH];
//     char workspace_src_dname[PATH_LENGTH + DIR_NAME_LENGTH];
//     char workspace_tinympc_dname[PATH_LENGTH + DIR_NAME_LENGTH];
//     char workspace_include_dname[PATH_LENGTH + DIR_NAME_LENGTH];

//     sprintf(workspace_dname, "%s", output_dir);
//     sprintf(workspace_src_dname, "%s/src", workspace_dname);
//     sprintf(workspace_tinympc_dname, "%s/tinympc", workspace_dname);
//     sprintf(workspace_include_dname, "%s/include", workspace_dname);

//     struct stat st = {0};

//     if (stat(workspace_dname, &st) == -1)
//     {
//         printf("Creating generated code directory at %s\n", workspace_dname);
//         mkdir(workspace_dname, 0700);         // workspace/
//         mkdir(workspace_src_dname, 0700);     // workspace/src
//         mkdir(workspace_tinympc_dname, 0700); // workspace/tinympc
//         mkdir(workspace_include_dname, 0700); // workspace/include
//     }


//     // // Codegen global options file in codegen_dir/tinympc
//     // codegen_glob_opts(start_time, workspace_tinympc_dname, nx, nu, N);

//     // Codegen example main file in codegen_dir/src
//     codegen_example_main(start_time, workspace_src_dname);







//     // Create CMakeLists.txt files

//     // CMakeLists.txt for codegen_dir/src/
//     char src_cmake_fname[PATH_LENGTH + DIR_NAME_LENGTH + FILE_NAME_LENGTH];
//     FILE *src_cmake_f;

//     sprintf(src_cmake_fname, "%s/CMakeLists.txt", workspace_src_dname);

//     // Open source file
//     src_cmake_f = fopen(src_cmake_fname, "w+");
//     if (src_cmake_f == NULL)
//         printf("ERROR OPENING DATA WORKSPACE FILE\n");
//     // return tiny_error(TINY_FOPEN_ERROR);

//     // Preamble
//     time(&start_time);
//     fprintf(src_cmake_f, "#\n");
//     fprintf(src_cmake_f, "# This file was autogenerated by TinyMPC on %s", ctime(&start_time));
//     fprintf(src_cmake_f, "#\n\n");

//     fprintf(src_cmake_f, "add_executable(tiny_main tiny_main.cpp tiny_data_workspace.cpp)\n");
//     fprintf(src_cmake_f, "target_link_libraries(tiny_main LINK_PUBLIC tinympc)");

//     fclose(src_cmake_f);

//     // CMakeLists.txt for codegen_dir/tinympc/
//     char tinympc_cmake_fname[PATH_LENGTH + DIR_NAME_LENGTH + FILE_NAME_LENGTH];
//     FILE *tinympc_cmake_f;

//     sprintf(tinympc_cmake_fname, "%s/CMakeLists.txt", workspace_tinympc_dname);

//     // Open source file
//     tinympc_cmake_f = fopen(tinympc_cmake_fname, "w+");
//     if (tinympc_cmake_f == NULL)
//         printf("ERROR OPENING DATA WORKSPACE FILE\n");
//     // return tiny_error(TINY_FOPEN_ERROR);

//     // Preamble
//     time(&start_time);
//     fprintf(tinympc_cmake_f, "#\n");
//     fprintf(tinympc_cmake_f, "# This file was autogenerated by TinyMPC on %s", ctime(&start_time));
//     fprintf(tinympc_cmake_f, "#\n\n");

//     fprintf(tinympc_cmake_f, "add_library(tinympc SHARED\n");
//     fprintf(tinympc_cmake_f, "admm.cpp\n");
//     fprintf(tinympc_cmake_f, ")\n\n");

//     fprintf(tinympc_cmake_f, "target_include_directories(tinympc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)\n\n");

//     // Generate wrapper for high-level interfaces
//     if (gen_wrapper != 0)
//     {
//         fprintf(tinympc_cmake_f, "add_library(tinympcShared SHARED\n");
//         fprintf(tinympc_cmake_f, "admm.cpp\n");
//         fprintf(tinympc_cmake_f, "tiny_wrapper.cpp\n");
//         fprintf(tinympc_cmake_f, "../src/tiny_data_workspace.cpp\n");
//         fprintf(tinympc_cmake_f, ")\n\n");
//         fprintf(tinympc_cmake_f, "target_include_directories(tinympcShared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)");
//     }

//     fclose(tinympc_cmake_f);

//     // CMakeLists.txt for codegen_dir/
//     char codegen_cmake_fname[PATH_LENGTH + DIR_NAME_LENGTH + FILE_NAME_LENGTH];
//     FILE *codegen_cmake_f;

//     sprintf(codegen_cmake_fname, "%s/CMakeLists.txt", workspace_dname);

//     // Open source file
//     codegen_cmake_f = fopen(codegen_cmake_fname, "w+");
//     if (codegen_cmake_f == NULL)
//         printf("ERROR OPENING DATA WORKSPACE FILE\n");
//     // return tiny_error(TINY_FOPEN_ERROR);

//     // Preamble
//     time(&start_time);
//     fprintf(codegen_cmake_f, "#\n");
//     fprintf(codegen_cmake_f, "# This file was autogenerated by TinyMPC on %s", ctime(&start_time));
//     fprintf(codegen_cmake_f, "#\n\n");

//     fprintf(codegen_cmake_f, "cmake_minimum_required(VERSION 3.0.0)\n");
//     fprintf(codegen_cmake_f, "project(TinyMPC VERSION 0.2.0 LANGUAGES CXX)\n\n");

//     fprintf(codegen_cmake_f, "set(CMAKE_CXX_STANDARD 17)\n");
//     fprintf(codegen_cmake_f, "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n\n");

//     fprintf(codegen_cmake_f, "include_directories(include/Eigen)\n");
//     fprintf(codegen_cmake_f, "add_subdirectory(tinympc)\n");
//     fprintf(codegen_cmake_f, "add_subdirectory(src)\n");

//     fclose(codegen_cmake_f);

//     // Copy remaining files into corresponding code gen directory folders

//     // Copy all files in include directory to code gen include directory verbatim
//     // TODO: remove dependence on Eigen files, or at least make it so we don't
//     //          need to copy over all the Eigen files
//     char source_include_dname[PATH_LENGTH + DIR_NAME_LENGTH + FILE_NAME_LENGTH];
//     sprintf(source_include_dname, "%s/include", tinympc_dir);
//     copy_dir(source_include_dname, workspace_include_dname);
//     printf("Content of include folder copied from %s to %s\n", source_include_dname, workspace_include_dname);

//     // Copy only necessary tinympc solver files into code gen directory
//     char src_fname[PATH_LENGTH + DIR_NAME_LENGTH + FILE_NAME_LENGTH];
//     char dst_fname[PATH_LENGTH + DIR_NAME_LENGTH + FILE_NAME_LENGTH];
//     sprintf(src_fname, "%s/src/tinympc/admm.hpp", tinympc_dir);
//     sprintf(dst_fname, "%s/tinympc/admm.hpp", output_dir);
//     copy_file(src_fname, dst_fname);
//     printf("Content of %s copied to %s\n", src_fname, dst_fname);

//     sprintf(src_fname, "%s/src/tinympc/admm.cpp", tinympc_dir);
//     sprintf(dst_fname, "%s/tinympc/admm.cpp", output_dir);
//     copy_file(src_fname, dst_fname);
//     printf("Content of %s copied to %s\n", src_fname, dst_fname);

//     sprintf(src_fname, "%s/src/tinympc/types.hpp", tinympc_dir);
//     sprintf(dst_fname, "%s/tinympc/types.hpp", output_dir);
//     copy_file(src_fname, dst_fname);
//     printf("Content of %s copied to %s\n", src_fname, dst_fname);

//     if (gen_wrapper != 0)
//     {
//         sprintf(src_fname, "%s/src/tinympc/tiny_wrapper.hpp", tinympc_dir);
//         sprintf(dst_fname, "%s/tinympc/tiny_wrapper.hpp", output_dir);
//         copy_file(src_fname, dst_fname);
//         printf("Content of %s copied to %s\n", src_fname, dst_fname);

//         sprintf(src_fname, "%s/src/tinympc/tiny_wrapper.cpp", tinympc_dir);
//         sprintf(dst_fname, "%s/tinympc/tiny_wrapper.cpp", output_dir);
//         copy_file(src_fname, dst_fname);
//         printf("Content of %s copied to %s\n", src_fname, dst_fname);
//     }

//     // Create README.md
//     char readme_fname[PATH_LENGTH + FILE_NAME_LENGTH];
//     FILE *readme_f;

//     sprintf(readme_fname, "%s/README.md", workspace_dname);

//     // Open source file
//     readme_f = fopen(readme_fname, "w+");
//     if (readme_f == NULL)
//         printf("ERROR OPENING README FILE\n");
//     // return tiny_error(TINY_FOPEN_ERROR);

//     time(&start_time);
//     fprintf(readme_f, "# Auto-generated TinyMPC code\n\n");
//     fprintf(readme_f, "### This file was autogenerated on %s", ctime(&start_time));

//     fprintf(readme_f, "This directory includes the tinympc algorithm under tinympc/, \
//             problem data, settings, and parameters under src/, and Eigen under include/.\n\n");

//     fprintf(readme_f, "An example main function is included in src/tiny_main.cpp\n\n");

//     fprintf(readme_f, "Enable wrapper for high-level interfaces.\n\n");

//     fprintf(readme_f, "CMakeLists.txt files are included for easy integration into \
//             existing CMake projects.");        

//     fprintf(readme_f, "To run the example with cmake:\n");
//     fprintf(readme_f, "```\n \
//             cd generated_code\n \
//             mkdir build\n \
//             cd build\n \
//             cmake ..\n \
//             make\n \
//             ./src/tiny_main\n \
//             ```");

//     fclose(readme_f);

//     // TODO: add error codes
//     return 1;
// }


static void print_matrix(FILE *f, MatrixXd mat, int num_elements)
{
    for (int i = 0; i < num_elements; i++)
    {
        fprintf(f, "(tinytype)%.16f", mat.reshaped<RowMajor>()[i]);
        if (i < num_elements - 1)
            fprintf(f, ",");
    }
}


// Create inc/tiny_data.hpp file
int codegen_data_header(const char* output_dir, int verbose) {
    char data_hpp_fname[PATH_LENGTH + DIR_NAME_LENGTH + FILE_NAME_LENGTH];
    FILE *data_hpp_f;

    sprintf(data_hpp_fname, "%s/inc/tiny_data.hpp", output_dir);

    // Open source file
    data_hpp_f = fopen(data_hpp_fname, "w+");
    if (data_hpp_f == NULL)
        printf("ERROR OPENING TINY DATA HEADER FILE\n");

    // Preamble
    time_t start_time;
    time(&start_time);
    fprintf(data_hpp_f, "/*\n");
    fprintf(data_hpp_f, " * This file was autogenerated by TinyMPC on %s", ctime(&start_time));
    fprintf(data_hpp_f, " */\n\n");

    fprintf(data_hpp_f, "#pragma once\n\n");

    fprintf(data_hpp_f, "#include \"types.hpp\"\n\n");

    fprintf(data_hpp_f, "#ifdef __cplusplus\n");
    fprintf(data_hpp_f, "extern \"C\" {\n");
    fprintf(data_hpp_f, "#endif\n\n");

    fprintf(data_hpp_f, "extern TinySolver tiny_data_solver;\n\n");

    fprintf(data_hpp_f, "#ifdef __cplusplus\n");
    fprintf(data_hpp_f, "}\n");
    fprintf(data_hpp_f, "#endif\n");

    // Close codegen data header file
    fclose(data_hpp_f);

    if (verbose) {
        printf("Data header generated in %s\n", workspace_hpp_fname);
    }
    return 0;
}

// Create src/tiny_data.cpp file
int codegen_data_source(TinySolver* solver, const char* output_dir, int verbose) {
    char data_cpp_fname[PATH_LENGTH + DIR_NAME_LENGTH + FILE_NAME_LENGTH];
    FILE *data_cpp_f;

    int nx = solver->work->nx;
    int nu = solver->work->nu;
    int N = solver->work->N;

    sprintf(data_cpp_fname, "%s/src/tiny_data.cpp", output_dir);

    // Open source file
    data_cpp_f = fopen(data_cpp_fname, "w+");
    if (data_cpp_f == NULL)
        printf("ERROR OPENING TINY DATA SOURCE FILE\n");

    // Preamble
    time_t start_time;
    time(&start_time);
    fprintf(data_cpp_f, "/*\n");
    fprintf(data_cpp_f, " * This file was autogenerated by TinyMPC on %s", ctime(&start_time));
    fprintf(data_cpp_f, " */\n\n");

    // Open extern C
    fprintf(data_cpp_f, "#include \"tinympc/tiny_data.hpp\"\n\n");
    fprintf(data_cpp_f, "#ifdef __cplusplus\n");
    fprintf(data_cpp_f, "extern \"C\" {\n");
    fprintf(data_cpp_f, "#endif\n\n");

    // Solution
    fprintf(data_cpp_f, "/* Solution */\n");
    fprintf(data_cpp_f, "TinySettings settings = {\n");

    fprintf(data_cpp_f, "\t%d,\t\t// iter\n", solver->solution->iter);
    fprintf(data_cpp_f, "\t%d,\t\t// solved\n", solver->solution->solved);
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// x\n"); // x solution
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N-1), nu * N-1);
    fprintf(data_cpp_f, ").finished(),\t// x\n"); // u solution

    fprintf(data_cpp_f, "};\n\n");

    // Cache
    fprintf(data_cpp_f, "/* Matrices that must be recomputed with changes in time step, rho */\n");
    fprintf(data_cpp_f, "TinyCache cache = {\n");

    fprintf(data_cpp_f, "\t(tinytype)%.16f,\t// rho (step size/penalty)\n", solver->cache->rho);
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, nx);
    print_matrix(data_cpp_f, solver->cache->Kinf, nu * nx);
    fprintf(data_cpp_f, ").finished(),\t// Kinf\n"); // Kinf
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, nx);
    print_matrix(data_cpp_f, solver->cache->Pinf, nx * nx);
    fprintf(data_cpp_f, ").finished(),\t// Pinf\n"); // Pinf
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, nu);
    print_matrix(data_cpp_f, solver->cache->Quu_inv, nu * nu);
    fprintf(data_cpp_f, ").finished(),\t// Quu_inv\n"); // Quu_inv
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, nx);
    print_matrix(data_cpp_f, solver->cache->AmBKt, nx * nx);
    fprintf(data_cpp_f, ").finished(),\t// AmBKt\n"); // AmBKt

    fprintf(data_cpp_f, "};\n\n");

    // Settings
    fprintf(data_cpp_f, "/* User settings */\n");
    fprintf(data_cpp_f, "TinySettings settings = {\n");

    fprintf(data_cpp_f, "\t(tinytype)%.16f,\t// primal tolerance\n", solver->settings->abs_pri_tol);
    fprintf(data_cpp_f, "\t(tinytype)%.16f,\t// dual tolerance\n", solver->settings->abs_dua_tol);
    fprintf(data_cpp_f, "\t%d,\t\t// max iterations\n", solver->settings->max_iters);
    fprintf(data_cpp_f, "\t%d,\t\t// iterations per termination check\n", solver->settings->check_termination);
    fprintf(data_cpp_f, "\t%d,\t\t// enable state constraints\n", solver->settings->en_state_bound);
    fprintf(data_cpp_f, "\t%d\t\t// enable input constraints\n", solver->settings->en_input_bound);

    fprintf(data_cpp_f, "};\n\n");

    // Workspace
    fprintf(data_cpp_f, "/* Problem variables */\n");
    fprintf(data_cpp_f, "TinyWorkspace work = {\n");

    fprintf(data_cpp_f, "\t%d,\t// Number of states\n", nx);
    fprintf(data_cpp_f, "\t%d,\t// Number of control inputs\n", nu);
    fprintf(data_cpp_f, "\t%d,\t// Number of knotpoints in the horizon\n", N);

    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// x\n"); // x
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N - 1));
    fprintf(data_cpp_f, ").finished(),\t// u\n"); // u

    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// q\n"); // q
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N - 1));
    fprintf(data_cpp_f, ").finished(),\t// r\n"); // r

    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// p\n"); // p
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N - 1));
    fprintf(data_cpp_f, ").finished(),\t// d\n"); // d

    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// v\n"); // v
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// vnew\n"); // vnew
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N - 1));
    fprintf(data_cpp_f, ").finished(),\t// z\n"); // z
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N - 1));
    fprintf(data_cpp_f, ").finished(),\t// znew\n"); // znew

    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// g\n"); // g
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N - 1));
    fprintf(data_cpp_f, ").finished(),\t// y\n"); // y

    fprintf(data_cpp_f, "\t(tinyVector(%d) << ", nx);
    print_matrix(data_cpp_f, Q, nx);
    fprintf(data_cpp_f, ").finished(),\t// Q\n"); // Q
    fprintf(data_cpp_f, "\t(tinyVector(%d) << ", nu);
    print_matrix(data_cpp_f, R, nu);
    fprintf(data_cpp_f, ").finished(),\t// R\n"); // R
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, nx);
    print_matrix(data_cpp_f, Adyn, nx * nx);
    fprintf(data_cpp_f, ").finished(),\t// Adyn\n"); // Adyn
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, nu);
    print_matrix(data_cpp_f, Bdyn, nx * nu);
    fprintf(data_cpp_f, ").finished(),\t// Bdyn\n"); // Bdyn

    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, x_min, nx * N);
    fprintf(data_cpp_f, ").finished(),\t// x_min\n"); // x_min
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, x_max, nx * N);
    fprintf(data_cpp_f, ").finished(),\t// x_max\n"); // x_max
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, u_min, nu * (N - 1));
    fprintf(data_cpp_f, ").finished(),\t// u_min\n"); // u_min
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, u_max, nu * (N - 1));
    fprintf(data_cpp_f, ").finished(),\t// u_max\n"); // u_max
    
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// Xref\n"); // Xref
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N - 1));
    fprintf(data_cpp_f, ").finished(),\t// Uref\n"); // Uref

    fprintf(data_cpp_f, "\t(tinyVector(%d) << ", nu);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, 1), nu);
    fprintf(data_cpp_f, ").finished(),\t// Qu\n"); // Qu

    fprintf(data_cpp_f, "\t(tinytype)%.16f,\t// state primal residual\n", 0.0);
    fprintf(data_cpp_f, "\t(tinytype)%.16f,\t// input primal residual\n", 0.0);
    fprintf(data_cpp_f, "\t(tinytype)%.16f,\t// state dual residual\n", 0.0);
    fprintf(data_cpp_f, "\t(tinytype)%.16f,\t// input dual residual\n", 0.0);
    fprintf(data_cpp_f, "\t%d,\t// solve status\n", 0);
    fprintf(data_cpp_f, "\t%d,\t// solve iteration\n", 0);
    fprintf(data_cpp_f, "};\n\n");

    // Write solver struct definition to workspace file
    fprintf(data_cpp_f, "TinySolver tiny_data_solver = {&solution, &settings, &cache, &work};\n\n");

    // Close extern C
    fprintf(data_cpp_f, "#ifdef __cplusplus\n");
    fprintf(data_cpp_f, "}\n");
    fprintf(data_cpp_f, "#endif\n\n");

    // Close codegen data file
    fclose(data_cpp_f);
    if (verbose) {
        printf("Data generated in %s\n", data_cpp_fname);
    }
    return 0;
}

int codegen_example(const char* output_dir, int verbose) {
    char example_cpp_fname[PATH_LENGTH + DIR_NAME_LENGTH + FILE_NAME_LENGTH];
    FILE *example_cpp_f;

    sprintf(example_cpp_fname, "%s/src/tiny_main.cpp", output_dir);

    // Open global options file
    example_cpp_f = fopen(example_cpp_fname, "w+");
    if (example_cpp_f == NULL)
        printf("ERROR OPENING EXAMPLE MAIN FILE\n");

    // Preamble
    time(&start_time);
    fprintf(example_cpp_f, "/*\n");
    fprintf(example_cpp_f, " * This file was autogenerated by TinyMPC on %s", ctime(&start_time));
    fprintf(example_cpp_f, " */\n\n");

    fprintf(example_cpp_f, "#include <iostream>\n\n");

    fprintf(example_cpp_f, "#include <tinympc/tiny_api.hpp>\n");
    fprintf(example_cpp_f, "#include <tinympc/tiny_data.hpp>\n\n");

    fprintf(example_cpp_f, "using namespace Eigen;\n");
    fprintf(example_cpp_f, "IOFormat TinyFmt(4, 0, \", \", \"\\n\", \"[\", \"]\");\n\n");

    fprintf(example_cpp_f, "#ifdef __cplusplus\n");
    fprintf(example_cpp_f, "extern \"C\" {\n");
    fprintf(example_cpp_f, "#endif\n\n");

    fprintf(example_cpp_f, "int main()\n");
    fprintf(example_cpp_f, "{\n");
    fprintf(example_cpp_f, "\tint exitflag = 1;\n");
    fprintf(example_cpp_f, "\t// Double check some data\n");
    fprintf(example_cpp_f, "\tstd::cout << tiny_data_solver.settings->max_iter << std::endl;\n");
    fprintf(example_cpp_f, "\tstd::cout << tiny_data_solver.cache->AmBKt.format(TinyFmt) << std::endl;\n");
    fprintf(example_cpp_f, "\tstd::cout << tiny_data_solver.work->Adyn.format(TinyFmt) << std::endl;\n\n");

    fprintf(example_cpp_f, "\t// Visit https://tinympc.org/ to see how to set the initial condition and update the reference trajectory.\n\n");

    fprintf(example_cpp_f, "\texitflag = tiny_solve(&tiny_data_solver);\n\n");
    fprintf(example_cpp_f, "\tif (exitflag == 0) printf(\"Hooray! Solved with no error!\\n\");\n");
    fprintf(example_cpp_f, "\telse printf(\"Oops! Something went wrong!\\n\");\n");

    fprintf(example_cpp_f, "\treturn 0;\n");
    fprintf(example_cpp_f, "}\n\n");

    fprintf(example_cpp_f, "#ifdef __cplusplus\n");
    fprintf(example_cpp_f, "} /* extern \"C\" */\n");
    fprintf(example_cpp_f, "#endif\n");

    // Close codegen example main file
    fclose(example_cpp_f);
    if (verbose) {
        printf("Example tinympc main generated in %s\n", example_cpp_fname);
    }
    return 0;
}

#ifdef __cplusplus
}
#endif