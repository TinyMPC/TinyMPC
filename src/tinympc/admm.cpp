#include <iostream>

#include "admm.hpp"


extern "C" {


void solve_admm(struct tiny_problem *problem, struct tiny_params *params) {


    forward_pass(problem, params);
    update_slack(problem, params);
    update_dual(problem, params);
    update_linear_cost(problem, params);

    for (int i=0; i<problem->max_iter; i++) {
        // Solve linear system with Riccati and roll out to get new trajectory
        update_primal(problem, params);

        // Project slack variables into feasible domain
        update_slack(problem, params);

        // Compute next iteration of dual variables
        update_dual(problem, params);

        // Update linear control cost terms using reference trajectory, duals, and slack variables
        update_linear_cost(problem, params);

        problem->primal_residual_state = (problem->x - problem->vnew).cwiseAbs().maxCoeff();
        problem->dual_residual_state = (problem->v - problem->vnew).cwiseAbs().maxCoeff();
        problem->primal_residual_input = (problem->u - problem->znew).cwiseAbs().maxCoeff();
        problem->dual_residual_input = (problem->z - problem->znew).cwiseAbs().maxCoeff();


        // TODO: convert arrays of Eigen vectors into one Eigen matrix
        // Save previous slack variables
        problem->v = problem->vnew;
        problem->z = problem->znew;

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
 * Do backward Riccati pass then forward roll out
*/
void update_primal(struct tiny_problem *problem, struct tiny_params *params) {
    backward_pass_grad(problem, params);
    forward_pass(problem, params);
}

/**
 * Update linear terms from Riccati backward pass
*/
void backward_pass_grad(struct tiny_problem *problem, struct tiny_params *params) {
    for (int i=NHORIZON-1; i>0; i--) {
        std::cout << i << std::endl;
    }
}

/**
 * Use LQR feedback policy to roll out trajectory
*/
void forward_pass(struct tiny_problem *problem, struct tiny_params *params) {

}

/**
 * Project slack (auxiliary) variables into their feasible domain, defined by
 * projection functions related to each constraint
 * TODO: pass in meta information with each constraint assigning it to a
 * projection function
*/
void update_slack(struct tiny_problem *problem, struct tiny_params *params) {

}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
*/
void update_dual(struct tiny_problem *problem, struct tiny_params *params) {

}

/**
 * Update linear control cost terms in the Riccati feedback using the changing
 * slack and dual variables from ADMM
*/
void update_linear_cost(struct tiny_problem *problem, struct tiny_params *params) {

}

}