#include <iostream>

#include "admm.hpp"
#include "rho_benchmark.hpp"    

#define DEBUG_MODULE "TINYALG"

extern "C" {

/**
    * Update linear terms from Riccati backward pass
    */
void backward_pass_grad(TinySolver *solver)
{
    for (int i = solver->work->N - 2; i >= 0; i--)
    {
        (solver->work->d.col(i)).noalias() = solver->cache->Quu_inv * (solver->work->Bdyn.transpose() * solver->work->p.col(i + 1) + solver->work->r.col(i));
        (solver->work->p.col(i)).noalias() = solver->work->q.col(i) + solver->cache->AmBKt.lazyProduct(solver->work->p.col(i + 1)) - (solver->cache->Kinf.transpose()).lazyProduct(solver->work->r.col(i)); 
    }
}

/**
    * Use LQR feedback policy to roll out trajectory
    */
void forward_pass(TinySolver *solver)
{
    for (int i = 0; i < solver->work->N - 1; i++)
    {
        (solver->work->u.col(i)).noalias() = -solver->cache->Kinf.lazyProduct(solver->work->x.col(i)) - solver->work->d.col(i);
        // solver->work->u.col(i) << .001, .02, .3, 4;
        // DEBUG_PRINT("u(0): %f\n", solver->work->u.col(0)(0));
        // multAdyn(solver->Ax->cache.Adyn, solver->work->x.col(i));
        (solver->work->x.col(i + 1)).noalias() = solver->work->Adyn.lazyProduct(solver->work->x.col(i)) + solver->work->Bdyn.lazyProduct(solver->work->u.col(i));
    }
}

/**
    * Project slack (auxiliary) variables into their feasible domain, defined by
    * projection functions related to each constraint
    * TODO: pass in meta information with each constraint assigning it to a
    * projection function
    */
void update_slack(TinySolver *solver)
{
    solver->work->znew = solver->work->u + solver->work->y;
    solver->work->vnew = solver->work->x + solver->work->g;

    // Box constraints on input
    if (solver->settings->en_input_bound)
    {
        solver->work->znew = solver->work->u_max.cwiseMin(solver->work->u_min.cwiseMax(solver->work->znew));
    }

    // Box constraints on state
    if (solver->settings->en_state_bound)
    {
        solver->work->vnew = solver->work->x_max.cwiseMin(solver->work->x_min.cwiseMax(solver->work->vnew));
    }
}

/**
    * Update next iteration of dual variables by performing the augmented
    * lagrangian multiplier update
    */
void update_dual(TinySolver *solver)
{
    solver->work->y = solver->work->y + solver->work->u - solver->work->znew;
    solver->work->g = solver->work->g + solver->work->x - solver->work->vnew;
}

/**
    * Update linear control cost terms in the Riccati feedback using the changing
    * slack and dual variables from ADMM
    */
void update_linear_cost(TinySolver *solver)
{
    solver->work->r = -(solver->work->Uref.array().colwise() * solver->work->R.array()); // Uref = 0 so commented out for speed up. Need to uncomment if using Uref
    (solver->work->r).noalias() -= solver->cache->rho * (solver->work->znew - solver->work->y);
    solver->work->q = -(solver->work->Xref.array().colwise() * solver->work->Q.array());
    (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vnew - solver->work->g);
    solver->work->p.col(solver->work->N - 1) = -(solver->work->Xref.col(solver->work->N - 1).transpose().lazyProduct(solver->cache->Pinf));
    (solver->work->p.col(solver->work->N - 1)).noalias() -= solver->cache->rho * (solver->work->vnew.col(solver->work->N - 1) - solver->work->g.col(solver->work->N - 1));
}

/**
    * Check for termination condition by evaluating whether the largest absolute
    * primal and dual residuals for states and inputs are below threhold.
    */
bool termination_condition(TinySolver *solver)
{
    if (solver->work->iter % solver->settings->check_termination == 0)
    {
        solver->work->primal_residual_state = (solver->work->x - solver->work->vnew).cwiseAbs().maxCoeff();
        solver->work->dual_residual_state = ((solver->work->v - solver->work->vnew).cwiseAbs().maxCoeff()) * solver->cache->rho;
        solver->work->primal_residual_input = (solver->work->u - solver->work->znew).cwiseAbs().maxCoeff();
        solver->work->dual_residual_input = ((solver->work->z - solver->work->znew).cwiseAbs().maxCoeff()) * solver->cache->rho;

        if (solver->work->primal_residual_state < solver->settings->abs_pri_tol &&
            solver->work->primal_residual_input < solver->settings->abs_pri_tol &&
            solver->work->dual_residual_state < solver->settings->abs_dua_tol &&
            solver->work->dual_residual_input < solver->settings->abs_dua_tol)
        {
            return true;                 
        }
    }
    return false;
}

int solve(TinySolver *solver)
{
    // Initialize variables
    solver->solution->solved = 0;
    solver->solution->iter = 0;
    solver->work->status = 11; // TINY_UNSOLVED
    solver->work->iter = 0;

    // Setup for adaptive rho
    RhoAdapter adapter;
    adapter.rho_min = solver->settings->adaptive_rho_min;
    adapter.rho_max = solver->settings->adaptive_rho_max;
    adapter.clip = solver->settings->adaptive_rho_enable_clipping;
    
    RhoBenchmarkResult rho_result;

    // Store previous values for residuals
    tinyMatrix v_prev = solver->work->vnew;
    tinyMatrix z_prev = solver->work->znew;

    for (int i = 0; i < solver->settings->max_iter; i++)
    {
        // Solve linear system with Riccati and roll out to get new trajectory
        forward_pass(solver);

        // Project slack variables into feasible domain
        update_slack(solver);

        // Compute next iteration of dual variables
        update_dual(solver);

        // Update linear control cost terms using reference trajectory, duals, and slack variables
        update_linear_cost(solver);

        solver->work->iter += 1;

        // Calculate residuals for adaptive rho
        tinytype pri_res_input = (solver->work->u - solver->work->znew).cwiseAbs().maxCoeff();
        tinytype pri_res_state = (solver->work->x - solver->work->vnew).cwiseAbs().maxCoeff();
        tinytype dua_res_input = solver->cache->rho * (solver->work->znew - z_prev).cwiseAbs().maxCoeff();
        tinytype dua_res_state = solver->cache->rho * (solver->work->vnew - v_prev).cwiseAbs().maxCoeff();

        // Update rho every 5 iterations
        if (i> 0 && i % 5 == 0) {
            benchmark_rho_adaptation(
                &adapter,
                solver->work->x,
                solver->work->u,
                solver->work->vnew,
                solver->work->znew,
                solver->work->g,
                solver->work->y,
                solver->cache,
                solver->work,
                solver->work->N,
                &rho_result
            );
            
            // Update matrices using Taylor expansion
           update_matrices_with_derivatives(solver->cache, rho_result.final_rho);
        }
        
        // Store previous values for next iteration
        z_prev = solver->work->znew;
        v_prev = solver->work->vnew;

        // Check for whether cost is minimized by calculating residuals
        if (termination_condition(solver)) {
            solver->work->status = 1; // TINY_SOLVED

            // Save solution
            solver->solution->iter = solver->work->iter;
            solver->solution->solved = 1;
            solver->solution->x = solver->work->vnew;
            solver->solution->u = solver->work->znew;

            std::cout << "Solver converged in " << solver->work->iter << " iterations" << std::endl;

            return 0;
        }

        // Save previous slack variables
        solver->work->v = solver->work->vnew;
        solver->work->z = solver->work->znew;

        backward_pass_grad(solver);
    }
    
    solver->solution->iter = solver->work->iter;
    solver->solution->solved = 0;
    solver->solution->x = solver->work->vnew;
    solver->solution->u = solver->work->znew;
    return 1;
}

} /* extern "C" */