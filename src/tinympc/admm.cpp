#include <iostream>

#include "admm.hpp"

#define DEBUG_MODULE "TINYALG"


extern "C" {

#include "debug.h"

static uint64_t startTimestamp;
static uint64_t timeTaken;

void multAdyn(tiny_VectorNx &Ax, const tiny_MatrixNxNx &A, const tiny_VectorNx &x) {
    Ax(0) += (x(0) + A(0,4)*x(4) + A(0,6)*x(6) + A(0,10)*x(10));
    Ax(1) += (x(1) + A(1,3)*x(3) + A(1,7)*x(7) + A(1,9)*x(9));
    Ax(2) += x(2) + A(2,8)*x(8);
    Ax(3) += x(3) + A(3,9)*x(9);
    Ax(4) += x(4) + A(4,10)*x(10);
    Ax(5) += x(5) + A(5,11)*x(11);
    Ax(6) += (x(6) + A(6,4)*x(4) + A(6,10)*x(10));
    Ax(7) += (x(7) + A(7,3)*x(3) + A(7,9)*x(9));
    Ax(8) += x(8);
    Ax(9) += x(9);
    Ax(10) += x(10);
    Ax(11) += x(11);
}

void solve_lqr(struct tiny_problem *problem, const struct tiny_params *params) {
    problem->u.col(0) = -params->cache.Kinf * (problem->x.col(0) - params->Xref.col(0));
}


void solve_admm(struct tiny_problem *problem, const struct tiny_params *params) {


    problem->status = 0;
    problem->iter = 1;

    // backward_pass_grad(problem, params);
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

        // problem->primal_residual_state = (problem->x - problem->vnew).cwiseAbs().maxCoeff();
        // problem->dual_residual_state = ((problem->v - problem->vnew).cwiseAbs().maxCoeff()) * params->cache.rho;
        // problem->primal_residual_input = (problem->u - problem->znew).cwiseAbs().maxCoeff();
        // problem->dual_residual_input = ((problem->z - problem->znew).cwiseAbs().maxCoeff()) * params->cache.rho;

        // TODO: convert arrays of Eigen vectors into one Eigen matrix
        // Save previous slack variables
        problem->v = problem->vnew;
        problem->z = problem->znew;

        // // TODO: remove convergence check and just return when allotted runtime is up
        // // Check for convergence
        // if (problem->primal_residual_state < problem->abs_tol &&
        //     problem->primal_residual_input < problem->abs_tol &&
        //     problem->dual_residual_state < problem->abs_tol &&
        //     problem->dual_residual_input < problem->abs_tol)
        // {
        //     problem->status = 1;
        //     break;
        // }

        // TODO: add rho scaling

        problem->iter += 1;

        // std::cout << problem->primal_residual_state << std::endl;
        // std::cout << problem->dual_residual_state << std::endl;
        // std::cout << problem->primal_residual_input << std::endl;
        // std::cout << problem->dual_residual_input << "\n" << std::endl;
    }

}

/**
 * Do backward Riccati pass then forward roll out
*/
void update_primal(struct tiny_problem *problem, const struct tiny_params *params) {
    backward_pass_grad(problem, params);
    forward_pass(problem, params);
}

/**
 * Update linear terms from Riccati backward pass
*/
void backward_pass_grad(struct tiny_problem *problem, const struct tiny_params *params) {
    for (int i=NHORIZON-2; i>=0; i--) {
        // problem->Qu.noalias() = params->cache.Bdyn.transpose().lazyProduct(problem->p.col(i+1));
        // problem->Qu += problem->r.col(i);
        // (problem->d.col(i)).noalias() = params->cache.Quu_inv.lazyProduct(problem->Qu);
        (problem->d.col(i)).noalias() = params->cache.Quu_inv * (params->cache.Bdyn.transpose() * problem->p.col(i+1) + problem->r.col(i));
        (problem->p.col(i)).noalias() = problem->q.col(i) + params->cache.AmBKt.lazyProduct(problem->p.col(i+1)) - (params->cache.Kinf.transpose()).lazyProduct(problem->r.col(i)); // + params->cache.coeff_d2p * problem->d.col(i); // coeff_d2p always appears to be zeros
    }
}

/**
 * Use LQR feedback policy to roll out trajectory
*/
void forward_pass(struct tiny_problem *problem, const struct tiny_params *params) {
    for (int i=0; i<NHORIZON-1; i++) {
        (problem->u.col(i)).noalias() = -params->cache.Kinf.lazyProduct(problem->x.col(i)) - problem->d.col(i);
        // problem->u.col(i) << .001, .02, .3, 4;
        // DEBUG_PRINT("u(0): %f\n", problem->u.col(0)(0));
        // (problem->x.col(i+1)).noalias() = params->cache.Adyn.lazyProduct(problem->x.col(i)) + params->cache.Bdyn.lazyProduct(problem->u.col(i));
        multAdyn(problem->Ax, params->cache.Adyn, problem->x.col(i));
        (problem->x.col(i+1)).noalias() = problem->Ax + params->cache.Bdyn.lazyProduct(problem->u.col(i));
    }
}

/**
 * Project slack (auxiliary) variables into their feasible domain, defined by
 * projection functions related to each constraint
 * TODO: pass in meta information with each constraint assigning it to a
 * projection function
*/
void update_slack(struct tiny_problem *problem, const struct tiny_params *params) {
    // Box constraints on input
    // Get current time

    problem->znew = params->u_max.cwiseMin(params->u_min.cwiseMax(problem->u));

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
    // startTimestamp = usecTimestamp();
    problem->xg = problem->x + problem->g;
    // problem->dists = (params->A_constraints.transpose().cwiseProduct(problem->xg)).colwise().sum();
    // problem->dists -= params->x_max;
    for (int i=0; i<NHORIZON; i++) {
        problem->dist = (params->A_constraints[i].head(3)).lazyProduct(problem->xg.col(i).head(3)); // Distances can be computed in one step outside the for loop
        problem->dist -= params->x_max[i](0);
        // DEBUG_PRINT("dist: %f\n", dist);
        if (problem->dist <= 0) {
            problem->vnew.col(i) = problem->xg.col(i);
        }
        else {
            problem->xyz_new = problem->xg.col(i).head(3) - problem->dist*params->A_constraints[i].head(3).transpose();
            problem->vnew.col(i) << problem->xyz_new, problem->xg.col(i).tail(NSTATES-3);
        }
    }
    // timeTaken = usecTimestamp() - startTimestamp;
    // DEBUG_PRINT("slack: %d\n", timeTaken);
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
*/
void update_dual(struct tiny_problem *problem, const struct tiny_params *params) {
    problem->y = problem->y + problem->u - problem->znew;
    problem->g = problem->g + problem->x - problem->vnew;
}

/**
 * Update linear control cost terms in the Riccati feedback using the changing
 * slack and dual variables from ADMM
*/
void update_linear_cost(struct tiny_problem *problem, const struct tiny_params *params) {
    // problem->r = -(params->Uref.array().colwise() * params->R.array()); // Uref = 0 so commented out for speed up. Need to uncomment if using Uref
    problem->r = -params->cache.rho * (problem->znew - problem->y);
    problem->q = -(params->Xref.array().colwise() * params->Q.array());
    problem->q -= params->cache.rho * (problem->vnew - problem->g);
    problem->p.col(NHORIZON-1) = -(params->Xref.col(NHORIZON-1).array().colwise() * params->Qf.array());
    problem->p.col(NHORIZON-1) -= params->cache.rho * (problem->vnew.col(NHORIZON-1) - problem->g.col(NHORIZON-1));

    // for (int i=0; i<NHORIZON-1; i++) {
    //     problem->r.col(i) = -params->cache.rho * (problem->znew.col(i) - problem->y.col(i)) - params->R * params->Uref.col(i);
    //     problem->q.col(i) = -params->cache.rho * (problem->vnew.col(i) - problem->g.col(i)) - params->Q * params->Xref.col(i);
    // }
    // problem->p.col(NHORIZON-1) = -params->cache.rho * (problem->vnew.col(NHORIZON-1) - problem->g.col(NHORIZON-1)) - params->Qf * params->Xref.col(NHORIZON-1);
}

} /* extern "C" */