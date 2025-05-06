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
        (solver->work->d.col(i)).noalias() = solver->cache->Quu_inv * (solver->work->Bdyn.transpose() * solver->work->p.col(i + 1) + solver->work->r.col(i) + solver->cache->BPf);
        (solver->work->p.col(i)).noalias() = solver->work->q.col(i) + solver->cache->AmBKt.lazyProduct(solver->work->p.col(i + 1)) - (solver->cache->Kinf.transpose()).lazyProduct(solver->work->r.col(i)) + solver->cache->APf; 
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
        (solver->work->x.col(i + 1)).noalias() = solver->work->Adyn.lazyProduct(solver->work->x.col(i)) + solver->work->Bdyn.lazyProduct(solver->work->u.col(i)) + solver->work->fdyn;
    }
}

/**
 * Project a vector s onto the second order cone defined by mu
 * @param s, mu
 * @return projection onto cone if s is outside cone. Return s if s is inside cone.
*/
tinyVector project_soc(tinyVector s, float mu) {
    tinytype u0 = s(Eigen::placeholders::last) * mu;
    tinyVector u1 = s.head(s.rows()-1);
    float a = u1.norm();
    tinyVector cone_origin(s.rows());
    cone_origin.setZero();

    if (a <= -u0) { // below cone
        return cone_origin;
    }
    else if (a <= u0) { // in cone
        return s;
    }
    else if (a >= abs(u0)) { // outside cone
        Matrix<tinytype, 3, 1> u2(u1.size() + 1);
        u2 << u1, a/mu;
        return 0.5 * (1 + u0/a) * u2;
    }
    else {
        return cone_origin;
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

    // Update bound constraint slack variables for state
    solver->work->vnew = solver->work->x + solver->work->g;
    
    // Update bound constraint slack variables for input
    solver->work->znew = solver->work->u + solver->work->y;

    // Box constraints on state
    if (solver->settings->en_state_bound) {
        solver->work->vnew = solver->work->x_max.cwiseMin(solver->work->x_min.cwiseMax(solver->work->vnew));
    }

    // Box constraints on input
    if (solver->settings->en_input_bound) {
        solver->work->znew = solver->work->u_max.cwiseMin(solver->work->u_min.cwiseMax(solver->work->znew));
    }

    
    // Update second order cone slack variables for state
    if (solver->settings->en_state_soc && solver->work->numStateCones > 0) {
        solver->work->vcnew = solver->work->x + solver->work->gc;
    }

    // Update second order cone slack variables for input
    if (solver->settings->en_input_soc && solver->work->numInputCones > 0) {
        solver->work->zcnew = solver->work->u + solver->work->yc;
    }

    // Cone constraints on state
    if (solver->settings->en_state_soc) {
        for (int i=0; i<solver->work->N; i++) {
            for (int k=0; k<solver->work->numStateCones; k++) {
                int start = solver->work->Acx(k);
                int num_xs = solver->work->qcx(k);
                tinytype mu = solver->work->cx(k);
                tinyVector col = solver->work->vcnew.block(start, i, num_xs, 1);
                solver->work->vcnew.block(start, k, num_xs, 1) = project_soc(col, mu);
            }
        }
    }

    // Cone constraints on input
    if (solver->settings->en_input_soc) {
        for (int i=0; i<solver->work->N-1; i++) {
            for (int k=0; k<solver->work->numInputCones; k++) {
                int start = solver->work->Acu(k);
                int num_us = solver->work->qcu(k);
                tinytype mu = solver->work->cu(k);
                tinyVector col = solver->work->zcnew.block(start, i, num_us, 1);
                solver->work->zcnew.block(start, k, num_us, 1) = project_soc(col, mu);
            }
        }
    }
    
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
*/
void update_dual(TinySolver *solver)
{
    // Update bound constraint dual variables for state
    solver->work->g = solver->work->g + solver->work->x - solver->work->vnew;

    // Update bound constraint dual variables for input
    solver->work->y = solver->work->y + solver->work->u - solver->work->znew;
    
    // Update second order cone dual variables for state
    if (solver->settings->en_state_soc && solver->work->numStateCones > 0) {
        solver->work->gc = solver->work->gc + solver->work->x - solver->work->vcnew;
    }

    // Update second order cone dual variables for input
    if (solver->settings->en_input_soc && solver->work->numInputCones > 0) {
        solver->work->yc = solver->work->yc + solver->work->u - solver->work->zcnew;
    }
}

/**
 * Update linear control cost terms in the Riccati feedback using the changing
 * slack and dual variables from ADMM
*/
void update_linear_cost(TinySolver *solver)
{

    // Update state cost terms
    solver->work->q = -(solver->work->Xref.array().colwise() * solver->work->Q.array());
    (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vnew - solver->work->g);
    if (solver->settings->en_state_soc && solver->work->numStateCones > 0) {
        (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vcnew - solver->work->gc);
    }

    // Update input cost terms
    solver->work->r = -(solver->work->Uref.array().colwise() * solver->work->R.array());
    (solver->work->r).noalias() -= solver->cache->rho * (solver->work->znew - solver->work->y);
    if (solver->settings->en_input_soc && solver->work->numInputCones > 0) {
        (solver->work->r).noalias() -= solver->cache->rho * (solver->work->zcnew - solver->work->yc);
    }

    // Update terminal cost
    solver->work->p.col(solver->work->N - 1) = -(solver->work->Xref.col(solver->work->N - 1).transpose().lazyProduct(solver->cache->Pinf));
    (solver->work->p.col(solver->work->N - 1)).noalias() -= solver->cache->rho * (solver->work->vnew.col(solver->work->N - 1) - solver->work->g.col(solver->work->N - 1));

    if (solver->settings->en_state_soc && solver->work->numStateCones > 0) {
        solver->work->p.col(solver->work->N - 1) -= solver->cache->rho * (solver->work->vcnew.col(solver->work->N - 1) - solver->work->gc.col(solver->work->N - 1));
    }
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
    
    // Initialize SOC slack variables if needed
    if (solver->settings->en_state_soc && solver->work->numStateCones > 0) {
        solver->work->vcnew = solver->work->x;
    }
    
    if (solver->settings->en_input_soc && solver->work->numInputCones > 0) {
        solver->work->zcnew = solver->work->u;
    }

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

        // Handle adaptive rho if enabled
        if (solver->settings->adaptive_rho) {
            // Calculate residuals for adaptive rho
            tinytype pri_res_input = (solver->work->u - solver->work->znew).cwiseAbs().maxCoeff();
            tinytype pri_res_state = (solver->work->x - solver->work->vnew).cwiseAbs().maxCoeff();
            tinytype dua_res_input = solver->cache->rho * (solver->work->znew - z_prev).cwiseAbs().maxCoeff();
            tinytype dua_res_state = solver->cache->rho * (solver->work->vnew - v_prev).cwiseAbs().maxCoeff();

            // Update rho every 5 iterations
            if (i > 0 && i % 5 == 0) {
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