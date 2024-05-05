#pragma once

// #include <iostream>

#include "admm.hpp"
// #include <tinympc/tiny_data_workspace.hpp>


#ifdef __cplusplus
extern "C"
{
#endif

    void tiny_precompute_and_set_cache(TinyCache *cache, tinyMatrix Adyn, tinyMatrix Bdyn, tinyMatrix Q, tinyMatrix R, int nx, int nu, double rho);

    void tiny_update_settings(tinytype abs_pri_tol, tinytype abs_dua_tol, int max_iter, int check_termination, int en_state_bound, int en_input_bound);
    void tiny_set_default_settings(TinySettings* settings);


    // void set_x0(float *x0, int verbose);
    // void set_xref(float *xref, int verbose);
    // void set_uref(float *uref, int verbose);
    // void set_umin(float *umin, int verbose);
    // void set_umax(float *umax, int verbose);
    // void set_xmin(float *xmin, int verbose);
    // void set_xmax(float *xmax, int verbose);
    // void reset_dual_variables(int verbose);
    // void call_tiny_solve(int verbose);
    // void get_x(float *x_soln, int verbose);
    // void get_u(float *u_soln, int verbose);

#ifdef __cplusplus
}
#endif