#pragma once

// #include <iostream>

#include "admm.hpp"
#include "codegen.hpp"
// #include <tinympc/tiny_data_workspace.hpp>


#ifdef __cplusplus
extern "C" {
#endif

int tiny_setup(TinyCache* cache, TinyWorkspace* work, TinySolution* solution,
                tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix Q, tinyMatrix R, 
                tinytype rho, int nx, int nu, int N,
                tinyMatrix x_min, tinyMatrix x_max, tinyMatrix u_min, tinyMatrix u_max,
                TinySettings* settings, int verbose);
int tiny_precompute_and_set_cache(TinyCache *cache, 
                                    tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix Q, tinyMatrix R,
                                    int nx, int nu, tinytype rho, int verbose);
int tiny_solve(TinySolver *solver);

int tiny_update_settings(TinySettings* settings,
                            tinytype abs_pri_tol, tinytype abs_dua_tol, 
                            int max_iter, int check_termination, 
                            int en_state_bound, int en_input_bound);
int tiny_set_default_settings(TinySettings* settings);

int tiny_set_x0(TinySolver* solver, tinyVector x0);
int tiny_set_x_ref(TinySolver* solver, tinyMatrix x_ref);
int tiny_set_u_ref(TinySolver* solver, tinyMatrix u_ref);


int tiny_codegen(TinySolver* solver, const char* output_dir, int verbose);
    // int nx, int nu, int N,
    //              double *Adyn, double *Bdyn, double *Q, double *R,
    //              double *x_min, double *x_max, double *u_min, double *u_max,
    //              double rho, double abs_pri_tol, double abs_dua_tol, 
    //              int max_iters, int check_termination, int gen_wrapper,
    //              const char *tinympc_dir, const char *output_dir);

#ifdef __cplusplus
}
#endif