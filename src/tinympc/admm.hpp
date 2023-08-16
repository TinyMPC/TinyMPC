#pragma once

#include "types.hpp"


#ifdef __cplusplus
extern "C" {
#endif

void solve_admm(struct tiny_problem *problem, struct tiny_params *params);

void update_primal(struct tiny_problem *problem, struct tiny_params *params);
void backward_pass_grad(struct tiny_problem *problem, struct tiny_params *params);
void forward_pass(struct tiny_problem *problem, struct tiny_params *params);
void update_slack(struct tiny_problem *problem, struct tiny_params *params);
void update_dual(struct tiny_problem *problem, struct tiny_params *params);
void update_linear_cost(struct tiny_problem *problem, struct tiny_params *params);

#ifdef __cplusplus
}
#endif