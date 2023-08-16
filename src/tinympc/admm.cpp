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
    for (int i=NHORIZON-2; i>=0; i--) {
        problem->d.col(i) = params->cache.Quu_inv * (params->cache.Bdyn.transpose() * problem->p.col(i+1) + problem->r.col(i));
        problem->p.col(i) = problem->q.col(i) + params->cache.AmBKt * problem->p.col(i+1) - params->cache.Kinf.transpose() * problem->r.col(i) + params->cache.coeff_d2p * problem->d.col(i);
    }
}

/**
 * Use LQR feedback policy to roll out trajectory
*/
void forward_pass(struct tiny_problem *problem, struct tiny_params *params) {
    std::cout << problem->x.col(0) << "\n" << std::endl;
    for (int i=0; i<NHORIZON-1; i++) {
        problem->u.col(i) = -params->cache.Kinf * problem->x.col(i) - problem->d.col(i);
        std::cout << problem->u.col(i) << "\n" << std::endl;
        problem->x.col(i+1) = params->cache.Adyn * problem->x.col(i) + params->cache.Bdyn * problem->u.col(i);
        std::cout << problem->x.col(i+1) << "\n" << std::endl;
    }
}

/**
 * Project slack (auxiliary) variables into their feasible domain, defined by
 * projection functions related to each constraint
 * TODO: pass in meta information with each constraint assigning it to a
 * projection function
*/
void update_slack(struct tiny_problem *problem, struct tiny_params *params) {
    // Box constraints on input
    problem->z = params->u_max.cwiseMin(params->u_min.cwiseMax(problem->u));

    // Half space constraints on state
    // TODO: support multiple half plane constraints per knot point
    //      currently this only works for one constraint per knot point
    // TODO: can potentially take advantage of the fact that A_constraints[3:end] is zero and just do
    //      v.col(i) = x.col(i) - dist*A_constraints[i] since we have to copy x[3:end] into v anyway
    //      downside is it's not clear this is happening externally and so values of A_constraints
    //      not set to zero (other than the first three) can cause the algorithm to fail
    // TODO: the only state values changing here are the first three (x, y, z) so it doesn't make sense
    //      to do operations on the remaining 9 when projecting (or doing anything related to the dual
    //      or auxiliary variables). v and g could be of size (3) and everything would work the same.
    //      The only reason this doesn't break is because in the update_linear_cost function subtracts
    //      g from v and so the last nine entries are always zero.
    for (int i=0; i<NHORIZON; i++) {
        tiny_VectorNx xg = problem->x.col(i) + problem->g.col(i);
        tinytype dist = params->A_constraints[i].head(3) * xg.head(3) - params->x_max[i](0);
        if (dist <= 0) {
            problem->v.col(i) = xg;
        }
        else {
            Matrix<tinytype, 3, 1> xyz_new = xg.head(3) - dist*params->A_constraints[i].head(3).transpose();
            problem->v.col(i) << xyz_new, xg.tail(NSTATES-3);
        }
    }
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
*/
void update_dual(struct tiny_problem *problem, struct tiny_params *params) {
    problem->y = problem->y + problem->u - problem->z;
    problem->g = problem->g + problem->x - problem->v;
}

/**
 * Update linear control cost terms in the Riccati feedback using the changing
 * slack and dual variables from ADMM
*/
void update_linear_cost(struct tiny_problem *problem, struct tiny_params *params) {
    for (int i=0; i<NHORIZON-1; i++) {
        problem->r.col(i) = -params->cache.rho * (problem->z.col(i) - problem->y.col(i)) - params->R * params->Uref.col(i);
        std::cout << problem->r.col(i) << "\n" << std::endl;
        problem->q.col(i) = -params->cache.rho * (problem->v.col(i) - problem->g.col(i)) - params->Q * params->Xref.col(i);
        std::cout << problem->q.col(i) << "\n" << std::endl;
    }
    problem->p.col(NHORIZON-1) = -params->cache.rho * (problem->v.col(NHORIZON-1) - problem->g.col(NHORIZON-1)) - params->Qf * params->Xref.col(NHORIZON-1);
    std::cout << problem->p.col(NHORIZON-1) << std::endl;
}

} /* extern "C" */