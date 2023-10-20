#pragma once

#include "types.hpp"


#ifdef __cplusplus
extern "C" {
#endif

void solve_lqr(TinySolver *problem);
void solve_admm(TinySolver *problem);

void update_primal(TinySolver *problem);
void backward_pass_grad(TinySolver *problem);
void forward_pass(TinySolver *problem);
void update_slack(TinySolver *problem);
void update_dual(TinySolver *problem);
void update_linear_cost(TinySolver *problem);

#ifdef __cplusplus
}
#endif