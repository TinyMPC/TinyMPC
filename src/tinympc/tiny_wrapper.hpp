#pragma once

#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/tiny_data_workspace.hpp>


#ifdef __cplusplus
extern "C"
{
#endif

    void set_x0(float *x0, int verbose);
    void set_xref(float *xref, int verbose);
    void reset_dual_variables(int verbose);
    void call_tiny_solve(int verbose);
    void get_x(float *x_soln, int verbose);
    void get_u(float *u_soln, int verbose);
    void edit_x(float *x, int verbose);

#ifdef __cplusplus
}
#endif