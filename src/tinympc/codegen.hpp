#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

    int tiny_codegen(int nx, int nu, int N,
                     double *Adyn, double *Bdyn, double *Q, double *R,
                     double *x_min, double *x_max, double *u_min, double *u_max,
                     double rho, double abs_pri_tol, double abs_dua_tol, 
                     int max_iters, int check_termination, int gen_wrapper,
                     const char *tinympc_dir, const char *output_dir);

#ifdef __cplusplus
}
#endif