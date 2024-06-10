#pragma once

#include <iostream>
#include "admm.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int tiny_setup(TinySolver** solverp,
                tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix fdyn, tinyMatrix Q, tinyMatrix R, 
                tinytype rho, int nx, int nu, int N, int verbose);
int tiny_set_bound_constraints(TinySolver* solver,
                    tinyMatrix x_min, tinyMatrix x_max,
                    tinyMatrix u_min, tinyMatrix u_max);
int tiny_set_cone_constraints(TinySolver* solver,
                              VectorXi Acu, VectorXi qcu, tinyVector cu,
                              VectorXi Acx, VectorXi qcx, tinyVector cx);
int tiny_precompute_and_set_cache(TinyCache *cache, 
                                    tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix fdyn, tinyMatrix Q, tinyMatrix R,
                                    int nx, int nu, tinytype rho, int verbose);
int tiny_solve(TinySolver *solver);

int tiny_update_settings(TinySettings* settings,
                            tinytype abs_pri_tol, tinytype abs_dua_tol, 
                            int max_iter, int check_termination, 
                            int en_state_bound, int en_input_bound,
                            int en_state_soc, int en_input_soc);
int tiny_set_default_settings(TinySettings* settings);

int tiny_set_x0(TinySolver* solver, tinyVector x0);
int tiny_set_x_ref(TinySolver* solver, tinyMatrix x_ref);
int tiny_set_u_ref(TinySolver* solver, tinyMatrix u_ref);


#ifdef __cplusplus
}
#endif