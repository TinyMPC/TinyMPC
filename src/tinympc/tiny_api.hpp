#pragma once

// #include <iostream>

#include "admm.hpp"
// #include <tinympc/tiny_data_workspace.hpp>


#ifdef __cplusplus
extern "C"
{
#endif

    int tiny_setup(TinyCache* cache, TinyWorkspace* work, TinySolution* solution,
                   tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix Q, tinyMatrix R, 
                   tinytype rho, int nx, int nu, int N,
                   tinyVector x_min, tinyVector x_max, tinyVector u_min, tinyVector u_max,
                   TinySettings* settings);
    int tiny_precompute_and_set_cache(TinyCache *cache, 
                                      tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix Q, tinyMatrix R,
                                      int nx, int nu, tinytype rho);

    void tiny_update_settings(TinySettings* settings,
                              tinytype abs_pri_tol, tinytype abs_dua_tol, 
                              int max_iter, int check_termination, 
                              int en_state_bound, int en_input_bound);
    void tiny_set_default_settings(TinySettings* settings);


#ifdef __cplusplus
}
#endif