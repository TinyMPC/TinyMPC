#include <iostream>

#include "admm.hpp"


extern "C" {


void solve_admm(struct tiny_problem *problem, struct tiny_params *params) {


    forward_pass(problem, params);
    update_slack(problem, params);
    update_dual(problem, params);
    update_linear_cost(problem, params);

    for (int i=0; i<problem->max_iter; i++) {
        update_primal(problem, params);
        update_slack(problem, params);
        update_dual(problem, params);
        update_linear_cost(problem, params);

        // problem->primal_residual_state = # TODO: get maximum of abs.(problem->x - problem->vnew)
        // problem->primal_residual_input = # TODO: get maximum of abs.(problem->u - problem->znew)
        // problem->dual_residual_state = # TODO: get maximum of abs.(problem->v - problem->vnew)
        // problem->dual_residual_input = # TODO: get maximum of abs.(problem->z - problem->znew)


        // TODO: convert arrays of Eigen vectors into one Eigen matrix
        // Save previous slack variables
        for (int j=0; j<NHORIZON; j++) {
            problem->v[j] = problem->vnew[j];
        }
        for (int j=0; j<NHORIZON-1; j++) {
            problem->z[j] = problem->znew[j];
        }

        // TODO: remove convergence check and just return when allotted runtime is up
        // Check for convergence
        if (problem->primal_residual_state < problem->abs_tol &&
            problem->primal_residual_input < problem->abs_tol &&
            problem->dual_residual_state < problem->abs_tol &&
            problem->dual_residual_input < problem->abs_tol)
        {
            problem->status = 1;
            break;
        }

        // TODO: add rho scaling

        problem->iter += 1;
    }
}

/**
 * Update linear terms from Riccati backward pass
*/
void backward_pass_grad(tiny_VectorNx q[], tiny_VectorNu r[], tiny_VectorNx p[], tiny_VectorNu d[], struct tiny_params *params) {
    for (int i=NHORIZON-1; i>0; i--) {
        std::cout << i << std::endl;
    }
}

}