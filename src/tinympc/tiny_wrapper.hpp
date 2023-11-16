#pragma once

#include <iostream>

#include <tinympc/admm.hpp>
#include <tinympc/tiny_data_workspace.hpp>


#ifdef __cplusplus
extern "C"
{
#endif

    void set_x(float *x0);
    void set_xref(float *xref);
    void reset_dual_variables();
    void call_tiny_solve();
    void get_x(float *x_soln);
    void get_u(float *u_soln);
    void edit_x(float *x);

#ifdef __cplusplus
}
#endif