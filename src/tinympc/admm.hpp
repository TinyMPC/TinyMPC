#pragma once

#include "types.hpp"


#ifdef __cplusplus
extern "C" {
#endif

void solve_admm(struct tiny_problem *problem, struct tiny_params *params);

void backward_pass_grad(tiny_VectorNx q[], tiny_VectorNu r[], tiny_VectorNx p[], tiny_VectorNu d[], struct tiny_params *params);

#ifdef __cplusplus
}
#endif