#pragma once

#include "types.hpp"

#ifdef __cplusplus
extern "C"
{
#endif

int solve(TinySolver *solver);

void update_primal(TinySolver *solver);
void backward_pass_grad(TinySolver *solver);
void forward_pass(TinySolver *solver);
void update_slack(TinySolver *solver);
void update_dual(TinySolver *solver);
void update_linear_cost(TinySolver *solver);
bool termination_condition(TinySolver *solver);

#ifdef __cplusplus
}
#endif