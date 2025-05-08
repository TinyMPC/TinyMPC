#pragma once

#include "types.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int solve(TinySolver *solver);

void update_primal(TinySolver *solver);
void backward_pass_grad(TinySolver *solver);
void forward_pass(TinySolver *solver);
void update_slack(TinySolver *solver);
void update_dual(TinySolver *solver);
void update_linear_cost(TinySolver *solver);
bool termination_condition(TinySolver *solver);

/**
 * Project a vector s onto the second order cone defined by mu
 * @param s, mu
 * @return projection onto cone if s is outside cone. Return s if s is inside cone.
*/
tinyVector project_soc(tinyVector s, float mu);

#ifdef __cplusplus
}
#endif