// Debug configuration for embedded environments
// Define DEBUG_MODE=0 to disable iostream usage in embedded builds
#ifndef DEBUG_MODE
#define DEBUG_MODE 1
#include <iostream>
#endif

#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
//#include <error.h>
#include "error.hpp"
#include <Eigen/Dense>

// #include "types.hpp"
#include "codegen.hpp"

#ifdef __MINGW32__
#include <direct.h>
inline int mkdir(const char *pathname, int flags) {
    return _mkdir(pathname);
}
#endif

/* Define the maximum allowed length of the path (directory + filename + extension) */
#define PATH_LENGTH 2048

using namespace Eigen;

template <typename Derived>
static void print_matrix(FILE *f, const Eigen::MatrixBase<Derived>& mat, int num_elements)
{
    // Check if matrix is uninitialized or too small
    if (mat.size() == 0 || mat.size() < num_elements) {
        // Print zeros for all elements
        for (int i = 0; i < num_elements; i++) {
            fprintf(f, "(tinytype)0.0000000000000000");
            if (i < num_elements - 1)
                fprintf(f, ",");
        }
        return;
    }
    
    // Matrix is properly initialized and has enough elements
    for (int i = 0; i < num_elements; i++) {
        fprintf(f, "(tinytype)%.16f", mat.template reshaped<RowMajor>()[i]);
        if (i < num_elements - 1)
            fprintf(f, ",");
    }
}

template <typename Derived>
static void print_problem_data_array(FILE *f, const char* name,
                                     const char* size_expr,
                                     const Eigen::MatrixBase<Derived>& mat,
                                     int num_elements)
{
    fprintf(f, "tinytype %s[%s] = {\n    ", name, size_expr);

    if (mat.size() == 0 || mat.size() < num_elements) {
        for (int i = 0; i < num_elements; ++i) {
            fprintf(f, "0.000000f");
            if (i < num_elements - 1) {
                fprintf(f, ", ");
                if ((i + 1) % 6 == 0) {
                    fprintf(f, "\n    ");
                }
            }
        }
    } else {
        for (int i = 0; i < num_elements; ++i) {
            fprintf(f, "%.6ff", mat.template reshaped<RowMajor>()[i]);
            if (i < num_elements - 1) {
                fprintf(f, ", ");
                if ((i + 1) % 6 == 0) {
                    fprintf(f, "\n    ");
                }
            }
        }
    }

    fprintf(f, "\n};\n\n");
}

template <typename Derived>
static void print_legacy_assignment(FILE *f, const char* name,
                                    const Eigen::MatrixBase<Derived>& mat,
                                    int rows, int cols)
{
    fprintf(f, "%s << \n", name);
    if (mat.size() == 0 || mat.rows() != rows || mat.cols() != cols) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                fprintf(f, "0.000000f");
                if (!(r == rows - 1 && c == cols - 1)) {
                    fprintf(f, ",");
                }
            }
            fprintf(f, "\n");
        }
        fprintf(f, ";\n\n");
        return;
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            fprintf(f, "%.6ff", mat(r, c));
            if (!(r == rows - 1 && c == cols - 1)) {
                fprintf(f, ",");
            }
        }
        fprintf(f, "\n");
    }
    fprintf(f, ";\n\n");
}


static void create_directory(const char* dir, int verbose) {
    // Attempt to create directory
    if (mkdir(dir, S_IRWXU|S_IRWXG|S_IROTH)) {
        if (errno == EEXIST) { // Skip if directory already exists
#if DEBUG_MODE
            if (verbose) {
                std::cout << dir << " already exists, skipping." << std::endl;
            }
#else
            (void)verbose;
#endif
        } else {
            ERROR_MSG(EXIT_FAILURE, "Failed to create directory %s", dir);
        }
    }
}

// TODO: Make this fail if tiny_setup has not already been called
#ifdef __cplusplus
extern "C" {
#endif

int tiny_codegen(TinySolver* solver, const char* output_dir, int verbose) {
    if (!solver) {
#if DEBUG_MODE
        std::cout << "Error in tiny_codegen: solver is nullptr" << std::endl;
#endif
        return 1;
    }
    int status = 0;
    status |= codegen_create_directories(output_dir, verbose);
    status |= codegen_data_header(output_dir, verbose);
    status |= codegen_data_source(solver, output_dir, verbose);
    status |= codegen_example(output_dir, verbose);

    return status;
}

int tiny_codegen_problem_data(TinySolver* solver, const char* output_dir,
                              const char* basename, int verbose) {
    if (!solver) {
#if DEBUG_MODE
        std::cout << "Error in tiny_codegen_problem_data: solver is nullptr" << std::endl;
#endif
        return 1;
    }

    int status = 0;
    status |= codegen_create_directories(output_dir, verbose);
    status |= codegen_problem_data_header(
        solver, output_dir,
        basename ? basename : "generated_problem_data", verbose);
    return status;
}

int tiny_codegen_crazyflie_params(TinySolver* solver, const char* output_dir,
                                  const char* basename, int verbose) {
    if (!solver) {
#if DEBUG_MODE
        std::cout << "Error in tiny_codegen_crazyflie_params: solver is nullptr" << std::endl;
#endif
        return 1;
    }

    int status = 0;
    status |= codegen_create_directories(output_dir, verbose);
    status |= codegen_crazyflie_params_header(
        solver, output_dir,
        basename ? basename : "generated_params", verbose);
    return status;
}

int tiny_codegen_with_sensitivity(TinySolver* solver, const char* output_dir,
                                tinyMatrix* dK, tinyMatrix* dP,
                                tinyMatrix* dC1, tinyMatrix* dC2, int verbose) {
    if (!solver) {
#if DEBUG_MODE
        std::cout << "Error in tiny_codegen_with_sensitivity: solver is nullptr" << std::endl;
#endif
        return 1;
    }

    // Only store sensitivity matrices if adaptive rho is enabled
    if (solver->settings->adaptive_rho) {
        // Store the sensitivity matrices in the solver's cache
        solver->cache->dKinf_drho = *dK;
        solver->cache->dPinf_drho = *dP;
        solver->cache->dC1_drho = *dC1;
        solver->cache->dC2_drho = *dC2;
    }

    // Call the regular codegen function which will now include the sensitivity matrices if adaptive_rho is enabled
    return tiny_codegen(solver, output_dir, verbose);
}

// Create code generation folder structure in whichever directory the executable calling tiny_codegen was called
int codegen_create_directories(const char* output_dir, int verbose) {

    // Create output folder (root folder for code generation)
    create_directory(output_dir, verbose);

    // Create src folder
    char src_dir[PATH_LENGTH];
    snprintf(src_dir, PATH_LENGTH, "%s/src/", output_dir);
    create_directory(src_dir, verbose);

    // Create tinympc folder
    char tinympc_dir[PATH_LENGTH];
    snprintf(tinympc_dir, PATH_LENGTH, "%s/tinympc/", output_dir);
    create_directory(tinympc_dir, verbose);

    char problem_data_dir[PATH_LENGTH];
    snprintf(problem_data_dir, PATH_LENGTH, "%s/problem_data/", output_dir);
    create_directory(problem_data_dir, verbose);

    char crazyflie_dir[PATH_LENGTH];
    snprintf(crazyflie_dir, PATH_LENGTH, "%s/crazyflie/", output_dir);
    create_directory(crazyflie_dir, verbose);

    // // Create include folder
    // char inc_dir[PATH_LENGTH];
    // sprintf(inc_dir, "%s/include/", output_dir);
    // create_directory(inc_dir, verbose);

    return EXIT_SUCCESS;
}

int codegen_problem_data_header(TinySolver* solver, const char* output_dir,
                                const char* basename, int verbose) {
    char header_fname[PATH_LENGTH];
    FILE *header_f;

    int nx = solver->work->nx;
    int nu = solver->work->nu;
    int N = solver->work->N;

    tinyVector Q_nominal;
    tinyVector R_nominal;
    tinyMatrix R_augmented;
#ifdef TINYMPC_EMBEDDED
    Q_nominal = solver->work->Q;
    R_nominal = solver->work->R;
    R_augmented = solver->work->R.asDiagonal();
    R_augmented += solver->cache->rho * tinyMatrix::Identity(nu, nu);
#else
    Q_nominal = solver->work->Q.array() - solver->cache->rho;
    R_nominal = solver->work->R.array() - solver->cache->rho;
    R_augmented = solver->work->R.asDiagonal();
#endif
    tinyMatrix coeff_d2p = solver->cache->Kinf.transpose() * R_augmented
                         - solver->cache->AmBKt * solver->cache->Pinf * solver->work->Bdyn;

    snprintf(header_fname, PATH_LENGTH, "%s/problem_data/%s.hpp", output_dir, basename);

    header_f = fopen(header_fname, "w+");
    if (header_f == NULL) {
        ERROR_MSG(EXIT_FAILURE, "Failed to open file %s", header_fname);
    }

    time_t start_time;
    time(&start_time);
    fprintf(header_f, "/*\n");
    fprintf(header_f, " * This file was autogenerated by TinyMPC on %s", ctime(&start_time));
    fprintf(header_f, " *\n");
    fprintf(header_f, " * Embedded-oriented problem data export.\n");
    fprintf(header_f, " * Define NSTATES, NINPUTS, and NHORIZON before including this header.\n");
    fprintf(header_f, " */\n\n");
    fprintf(header_f, "#pragma once\n\n");
    fprintf(header_f, "#include <tinympc/types.hpp>\n\n");

    fprintf(header_f, "tinytype rho_value = %.6ff;\n\n", solver->cache->rho);

    print_problem_data_array(header_f, "Adyn_data", "NSTATES * NSTATES",
                             solver->work->Adyn, nx * nx);
    print_problem_data_array(header_f, "Bdyn_data", "NSTATES * NINPUTS",
                             solver->work->Bdyn, nx * nu);
    print_problem_data_array(header_f, "fdyn_data", "NSTATES",
                             solver->work->fdyn, nx);
    print_problem_data_array(header_f, "Q_data", "NSTATES",
                             Q_nominal, nx);
    print_problem_data_array(header_f, "R_data", "NINPUTS",
                             R_nominal, nu);

    print_problem_data_array(header_f, "Kinf_data", "NINPUTS * NSTATES",
                             solver->cache->Kinf, nu * nx);
    print_problem_data_array(header_f, "Pinf_data", "NSTATES * NSTATES",
                             solver->cache->Pinf, nx * nx);
    print_problem_data_array(header_f, "Quu_inv_data", "NINPUTS * NINPUTS",
                             solver->cache->Quu_inv, nu * nu);
    print_problem_data_array(header_f, "AmBKt_data", "NSTATES * NSTATES",
                             solver->cache->AmBKt, nx * nx);
    print_problem_data_array(header_f, "coeff_d2p_data", "NSTATES * NINPUTS",
                             coeff_d2p, nx * nu);
    print_problem_data_array(header_f, "APf_data", "NSTATES",
                             solver->cache->APf, nx);
    print_problem_data_array(header_f, "BPf_data", "NINPUTS",
                             solver->cache->BPf, nu);

    if (solver->work->x_min.size() == nx * N && solver->work->x_max.size() == nx * N &&
        solver->work->u_min.size() == nu * (N - 1) && solver->work->u_max.size() == nu * (N - 1)) {
        print_problem_data_array(header_f, "x_min_data", "NSTATES * NHORIZON",
                                 solver->work->x_min, nx * N);
        print_problem_data_array(header_f, "x_max_data", "NSTATES * NHORIZON",
                                 solver->work->x_max, nx * N);
        print_problem_data_array(header_f, "u_min_data", "NINPUTS * (NHORIZON - 1)",
                                 solver->work->u_min, nu * (N - 1));
        print_problem_data_array(header_f, "u_max_data", "NINPUTS * (NHORIZON - 1)",
                                 solver->work->u_max, nu * (N - 1));
    }

    fclose(header_f);

    if (verbose) {
        printf("Problem data generated in %s\n", header_fname);
    }
    return 0;
}

int codegen_crazyflie_params_header(TinySolver* solver, const char* output_dir,
                                    const char* basename, int verbose) {
    char header_fname[PATH_LENGTH];
    FILE *header_f;

    int nx = solver->work->nx;
    int nu = solver->work->nu;

    tinyMatrix Q_nominal;
    tinyMatrix R_nominal;
    tinyMatrix R_augmented;
#ifdef TINYMPC_EMBEDDED
    Q_nominal = solver->work->Q.asDiagonal();
    R_nominal = solver->work->R.asDiagonal();
    R_augmented = solver->work->R.asDiagonal();
    R_augmented += solver->cache->rho * tinyMatrix::Identity(nu, nu);
#else
    Q_nominal =
        (solver->work->Q.array() - solver->cache->rho).matrix().asDiagonal();
    R_nominal =
        (solver->work->R.array() - solver->cache->rho).matrix().asDiagonal();
    R_augmented = solver->work->R.asDiagonal();
#endif
    tinyMatrix coeff_d2p = solver->cache->Kinf.transpose() * R_augmented
                         - solver->cache->AmBKt * solver->cache->Pinf * solver->work->Bdyn;

    snprintf(header_fname, PATH_LENGTH, "%s/crazyflie/%s.h", output_dir, basename);

    header_f = fopen(header_fname, "w+");
    if (header_f == NULL) {
        ERROR_MSG(EXIT_FAILURE, "Failed to open file %s", header_fname);
    }

    time_t start_time;
    time(&start_time);
    fprintf(header_f, "/*\n");
    fprintf(header_f, " * This file was autogenerated by TinyMPC on %s", ctime(&start_time));
    fprintf(header_f, " *\n");
    fprintf(header_f, " * Deployable Crazyflie cache include for controller_tinympc_eigen.\n");
    fprintf(header_f, " * It matches the params_*.h assignment style used by the firmware.\n");
    fprintf(header_f, " */\n\n");

    print_legacy_assignment(header_f, "Kinf", solver->cache->Kinf, nu, nx);
    print_legacy_assignment(header_f, "Pinf", solver->cache->Pinf, nx, nx);
    print_legacy_assignment(header_f, "A", solver->work->Adyn, nx, nx);
    print_legacy_assignment(header_f, "B", solver->work->Bdyn, nx, nu);
    print_legacy_assignment(header_f, "Quu_inv", solver->cache->Quu_inv, nu, nu);
    print_legacy_assignment(header_f, "AmBKt", solver->cache->AmBKt, nx, nx);
    print_legacy_assignment(header_f, "coeff_d2p", coeff_d2p, nx, nu);
    print_legacy_assignment(header_f, "Q", Q_nominal, nx, nx);
    print_legacy_assignment(header_f, "R", R_nominal, nu, nu);

    fclose(header_f);

    if (verbose) {
        printf("Crazyflie params generated in %s\n", header_fname);
    }
    return 0;
}

// Create inc/tiny_data.hpp file
int codegen_data_header(const char* output_dir, int verbose) {
    char data_hpp_fname[PATH_LENGTH];
    FILE *data_hpp_f;

    snprintf(data_hpp_fname, PATH_LENGTH, "%s/tinympc/tiny_data.hpp", output_dir);

    // Open data header file
    data_hpp_f = fopen(data_hpp_fname, "w+");
    if (data_hpp_f == NULL)
        ERROR_MSG(EXIT_FAILURE, "Failed to open file %s", data_hpp_fname);
    
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

    fprintf(data_hpp_f, "extern TinySolver tiny_solver;\n\n");

    fprintf(data_hpp_f, "#ifdef __cplusplus\n");
    fprintf(data_hpp_f, "}\n");
    fprintf(data_hpp_f, "#endif\n");

    // Close codegen data header file
    fclose(data_hpp_f);

    if (verbose) {
        printf("Data header generated in %s\n", data_hpp_fname);
    }
    return 0;
}

// Create src/tiny_data.cpp file
int codegen_data_source(TinySolver* solver, const char* output_dir, int verbose) {
    char data_cpp_fname[PATH_LENGTH];
    FILE *data_cpp_f;

    int nx = solver->work->nx;
    int nu = solver->work->nu;
    int N = solver->work->N;

    snprintf(data_cpp_fname, PATH_LENGTH, "%s/src/tiny_data.cpp", output_dir);

    // Open data source file
    data_cpp_f = fopen(data_cpp_fname, "w+");
    if (data_cpp_f == NULL)
        ERROR_MSG(EXIT_FAILURE, "Failed to open file %s", data_cpp_fname);

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
    fprintf(data_cpp_f, "TinySolution solution = {\n");

    fprintf(data_cpp_f, "\t%d,\t\t// iter\n", solver->solution->iter);
    fprintf(data_cpp_f, "\t%d,\t\t// solved\n", solver->solution->solved);
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// x\n"); // x solution
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N-1), nu * (N-1));
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
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, nx);
    print_matrix(data_cpp_f, solver->cache->C1, nx * nx);
    fprintf(data_cpp_f, ").finished(),\t// C1\n"); // C1
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, nx);
    print_matrix(data_cpp_f, solver->cache->C2, nx * nx);
    fprintf(data_cpp_f, ").finished()"); // C2, no comma if no sensitivity matrices

    // Only print sensitivity matrices if adaptive rho is enabled
    if (solver->settings->adaptive_rho) {
        fprintf(data_cpp_f, ",\t// C2\n"); // Add comma and comment for C2 if we have more matrices
        
        // Add sensitivity matrices within the struct initialization
        fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, nx);
        print_matrix(data_cpp_f, solver->cache->dKinf_drho, nu * nx);
        fprintf(data_cpp_f, ").finished(),\t// dKinf_drho\n"); // dKinf_drho
        fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, nx);
        print_matrix(data_cpp_f, solver->cache->dPinf_drho, nx * nx);
        fprintf(data_cpp_f, ").finished(),\t// dPinf_drho\n"); // dPinf_drho
        fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, nx);
        print_matrix(data_cpp_f, solver->cache->dC1_drho, nx * nx);
        fprintf(data_cpp_f, ").finished(),\t// dC1_drho\n"); // dC1_drho
        fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, nx);
        print_matrix(data_cpp_f, solver->cache->dC2_drho, nx * nx);
        fprintf(data_cpp_f, ").finished()\t// dC2_drho\n"); // dC2_drho
    } else {
        fprintf(data_cpp_f, "\t// C2\n"); // Just add comment for C2
    }

    fprintf(data_cpp_f, "};\n\n");

    // Settings
    fprintf(data_cpp_f, "/* User settings */\n");
    fprintf(data_cpp_f, "TinySettings settings = {\n");

    fprintf(data_cpp_f, "\t(tinytype)%.16f,\t// primal tolerance\n", solver->settings->abs_pri_tol);
    fprintf(data_cpp_f, "\t(tinytype)%.16f,\t// dual tolerance\n", solver->settings->abs_dua_tol);
    fprintf(data_cpp_f, "\t%d,\t\t// max iterations\n", solver->settings->max_iter);
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
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N-1));
    fprintf(data_cpp_f, ").finished(),\t// u\n"); // u

    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// q\n"); // q
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N-1));
    fprintf(data_cpp_f, ").finished(),\t// r\n"); // r

    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// p\n"); // p
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N-1));
    fprintf(data_cpp_f, ").finished(),\t// d\n"); // d

    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// v\n"); // v
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// vnew\n"); // vnew
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N-1));
    fprintf(data_cpp_f, ").finished(),\t// z\n"); // z
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N-1));
    fprintf(data_cpp_f, ").finished(),\t// znew\n"); // znew

    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// g\n"); // g
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N-1));
    fprintf(data_cpp_f, ").finished(),\t// y\n"); // y

    fprintf(data_cpp_f, "\t(tinyVector(%d) << ", nx);
    print_matrix(data_cpp_f, solver->work->Q, nx);
    fprintf(data_cpp_f, ").finished(),\t// Q\n"); // Q
    fprintf(data_cpp_f, "\t(tinyVector(%d) << ", nu);
    print_matrix(data_cpp_f, solver->work->R, nu);
    fprintf(data_cpp_f, ").finished(),\t// R\n"); // R
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, nx);
    print_matrix(data_cpp_f, solver->work->Adyn, nx * nx);
    fprintf(data_cpp_f, ").finished(),\t// Adyn\n"); // Adyn
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, nu);
    print_matrix(data_cpp_f, solver->work->Bdyn, nx * nu);
    fprintf(data_cpp_f, ").finished(),\t// Bdyn\n"); // Bdyn

    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, solver->work->x_min, nx * N);
    fprintf(data_cpp_f, ").finished(),\t// x_min\n"); // x_min
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, solver->work->x_max, nx * N);
    fprintf(data_cpp_f, ").finished(),\t// x_max\n"); // x_max
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, solver->work->u_min, nu * (N-1));
    fprintf(data_cpp_f, ").finished(),\t// u_min\n"); // u_min
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, solver->work->u_max, nu * (N-1));
    fprintf(data_cpp_f, ").finished(),\t// u_max\n"); // u_max
    
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nx, N);
    print_matrix(data_cpp_f, MatrixXd::Zero(nx, N), nx * N);
    fprintf(data_cpp_f, ").finished(),\t// Xref\n"); // Xref
    fprintf(data_cpp_f, "\t(tinyMatrix(%d, %d) << ", nu, N-1);
    print_matrix(data_cpp_f, MatrixXd::Zero(nu, N - 1), nu * (N-1));
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
    fprintf(data_cpp_f, "TinySolver tiny_solver = {&solution, &settings, &cache, &work};\n\n");

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
    char example_cpp_fname[PATH_LENGTH];
    FILE *example_cpp_f;

    snprintf(example_cpp_fname, PATH_LENGTH, "%s/src/tiny_main.cpp", output_dir);

    // Open example file
    example_cpp_f = fopen(example_cpp_fname, "w+");
    if (example_cpp_f == NULL)
        ERROR_MSG(EXIT_FAILURE, "Failed to open file %s", example_cpp_fname);

    // Preamble
    time_t start_time;
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
    fprintf(example_cpp_f, "\tstd::cout << \"rho: \" << tiny_solver.cache->rho << std::endl;\n");
    fprintf(example_cpp_f, "\tstd::cout << \"\\nmax iters: \" << tiny_solver.settings->max_iter << std::endl;\n");
    fprintf(example_cpp_f, "\tstd::cout << \"\\nState transition matrix:\\n\" << tiny_solver.work->Adyn.format(TinyFmt) << std::endl;\n");
    fprintf(example_cpp_f, "\tstd::cout << \"\\nInput/control matrix:\\n\" << tiny_solver.work->Bdyn.format(TinyFmt) << std::endl;\n\n");

    fprintf(example_cpp_f, "\t// Visit https://tinympc.org/ to see how to set the initial condition and update the reference trajectory.\n\n");

    fprintf(example_cpp_f, "\tstd::cout << \"\\nSolving...\\n\" << std::endl;\n\n");
    fprintf(example_cpp_f, "\texitflag = tiny_solve(&tiny_solver);\n\n");
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
