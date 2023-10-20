#include <iostream>

#include "admm.hpp"

#define DEBUG_MODULE "TINYALG"

extern "C"
{

    // #include "debug.h"

    static uint64_t startTimestamp;

    void solve_admm(TinySolver *problem)
    {
        // Initialize variables
        problem->work->status = 0;
        problem->work->iter = 1;

        forward_pass(problem);
        update_slack(problem);
        update_dual(problem);
        update_linear_cost(problem);
        for (int i = 0; i < problem->settings->max_iter; i++)
        {

            // Solve linear system with Riccati and roll out to get new trajectory
            update_primal(problem);

            // Project slack variables into feasible domain
            update_slack(problem);

            // Compute next iteration of dual variables
            update_dual(problem);

            // Update linear control cost terms using reference trajectory, duals, and slack variables
            update_linear_cost(problem);

            if (problem->work->iter % problem->settings->check_termination == 0)
            {
                problem->work->primal_residual_state = (problem->work->x - problem->work->vnew).cwiseAbs().maxCoeff();
                problem->work->dual_residual_state = ((problem->work->v - problem->work->vnew).cwiseAbs().maxCoeff()) * problem->cache->rho;
                problem->work->primal_residual_input = (problem->work->u - problem->work->znew).cwiseAbs().maxCoeff();
                problem->work->dual_residual_input = ((problem->work->z - problem->work->znew).cwiseAbs().maxCoeff()) * problem->cache->rho;

                if (problem->work->primal_residual_state < problem->settings->abs_pri_tol &&
                    problem->work->primal_residual_input < problem->settings->abs_pri_tol &&
                    problem->work->dual_residual_state < problem->settings->abs_dua_tol &&
                    problem->work->dual_residual_input < problem->settings->abs_dua_tol)
                {
                    problem->work->status = 1;
                    break;
                }
            }

            // Save previous slack variables
            problem->work->v = problem->work->vnew;
            problem->work->z = problem->work->znew;

            problem->work->iter += 1;

            // std::cout << problem->work->primal_residual_state << std::endl;
            // std::cout << problem->work->dual_residual_state << std::endl;
            // std::cout << problem->work->primal_residual_input << std::endl;
            // std::cout << problem->work->dual_residual_input << "\n" << std::endl;
        }
    }

    /**
     * Do backward Riccati pass then forward roll out
     */
    void update_primal(TinySolver *problem)
    {
        backward_pass_grad(problem);
        forward_pass(problem);
    }

    /**
     * Update linear terms from Riccati backward pass
     */
    void backward_pass_grad(TinySolver *problem)
    {
        for (int i = NHORIZON - 2; i >= 0; i--)
        {
            (problem->work->d.col(i)).noalias() = problem->cache->Quu_inv * (problem->work->Bdyn.transpose() * problem->work->p.col(i + 1) + problem->work->r.col(i));
            (problem->work->p.col(i)).noalias() = problem->work->q.col(i) + problem->cache->AmBKt.lazyProduct(problem->work->p.col(i + 1)) - (problem->cache->Kinf.transpose()).lazyProduct(problem->work->r.col(i)); // + problem->cache->coeff_d2p * problem->work->d.col(i); // coeff_d2p always appears to be zeros
        }
    }

    /**
     * Use LQR feedback policy to roll out trajectory
     */
    void forward_pass(TinySolver *problem)
    {
        for (int i = 0; i < NHORIZON - 1; i++)
        {
            (problem->work->u.col(i)).noalias() = -problem->cache->Kinf.lazyProduct(problem->work->x.col(i)) - problem->work->d.col(i);
            // problem->work->u.col(i) << .001, .02, .3, 4;
            // DEBUG_PRINT("u(0): %f\n", problem->work->u.col(0)(0));
            // multAdyn(problem->Ax->cache.Adyn, problem->work->x.col(i));
            (problem->work->x.col(i + 1)).noalias() = problem->work->Adyn.lazyProduct(problem->work->x.col(i)) + problem->work->Bdyn.lazyProduct(problem->work->u.col(i));
        }
    }

    /**
     * Project slack (auxiliary) variables into their feasible domain, defined by
     * projection functions related to each constraint
     * TODO: pass in meta information with each constraint assigning it to a
     * projection function
     */
    void update_slack(TinySolver *problem)
    {
        problem->work->znew = problem->work->u + problem->work->y;
        problem->work->vnew = problem->work->x + problem->work->g;

        // Box constraints on input
        if (problem->settings->en_input_bound)
        {
            problem->work->znew = problem->work->u_max.cwiseMin(problem->work->u_min.cwiseMax(problem->work->znew));
        }

        // Box constraints on state
        if (problem->settings->en_input_bound)
        {
            problem->work->vnew = problem->work->x_max.cwiseMin(problem->work->x_min.cwiseMax(problem->work->vnew));
        }
    }

    /**
     * Update next iteration of dual variables by performing the augmented
     * lagrangian multiplier update
     */
    void update_dual(TinySolver *problem)
    {
        problem->work->y = problem->work->y + problem->work->u - problem->work->znew;
        problem->work->g = problem->work->g + problem->work->x - problem->work->vnew;
    }

    /**
     * Update linear control cost terms in the Riccati feedback using the changing
     * slack and dual variables from ADMM
     */
    void update_linear_cost(TinySolver *problem)
    {
        // problem->work->r = -(problem->Uref.array().colwise() * problem->work->r.array()); // Uref = 0 so commented out for speed up. Need to uncomment if using Uref
        problem->work->r = -problem->cache->rho * (problem->work->znew - problem->work->y);
        problem->work->q = -(problem->work->Xref.array().colwise() * problem->work->Q.array());
        problem->work->q -= problem->cache->rho * (problem->work->vnew - problem->work->g);
        problem->work->p.col(NHORIZON - 1) = -(problem->work->Xref.col(NHORIZON - 1).array().colwise() * problem->work->Qf.array());
        problem->work->p.col(NHORIZON - 1) -= problem->cache->rho * (problem->work->vnew.col(NHORIZON - 1) - problem->work->g.col(NHORIZON - 1));
    }

} /* extern "C" */