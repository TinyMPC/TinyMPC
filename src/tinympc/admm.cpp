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
    // CRITICAL: Preserve initial condition - never update x.col(0)!
    // Store initial condition to restore after forward pass
    tinyVector x0_initial = solver->work->x.col(0);
    
    for (int i = 0; i < solver->work->N - 1; i++)
    {
        // REVERT TO WORKING VERSION: Original ADMM blending that gave perfect results
        tinyVector u_lqr = -solver->cache->Kinf.lazyProduct(solver->work->x.col(i)) - solver->work->d.col(i);
        tinytype alpha = 0.9;
        solver->work->u.col(i) = alpha * solver->work->znew.col(i) + (1.0 - alpha) * u_lqr;
        // Blend dynamics with SDP projected states for consensus
        tinyVector x_dyn = solver->work->Adyn.lazyProduct(solver->work->x.col(i)) + solver->work->Bdyn.lazyProduct(solver->work->u.col(i)) + solver->work->fdyn;
        solver->work->x.col(i + 1) = alpha * solver->work->vnew.col(i + 1) + (1.0 - alpha) * x_dyn;
    }
    
    // RESTORE initial condition - this should NEVER change!
    solver->work->x.col(0) = x0_initial;
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
 * SDP projection method for augmented state with proper moment matrix constraints
 * Projects augmented state x̄ = [x, vec(xx^T)] onto PSD cone
 */
void project_sdp_augmented_state(TinySolver *solver) {
    // Only apply if we're using augmented state (dimension 20)
    if (solver->work->vnew.rows() != 20) {
        return; // Skip if not using augmented formulation
    }
    
    std::cout << "[SDP] Augmented state projection called!" << std::endl;
    
    // CRITICAL: Store initial condition to preserve it
    tinyVector x0_initial = solver->work->vnew.col(0);
    
    
    for (int k = 0; k < solver->work->N; k++) {
        // Get augmented state x̄ = [x, vec(xx^T)]
        tinyVector x_aug = solver->work->vnew.col(k);
        
        // Extract physical state (first 4 elements)
        tinyVector x_phys = x_aug.head(4);
        
        // Extract vectorized quadratic terms (elements 4-19)
        Eigen::Matrix<tinytype, 16, 1> vec_xx = x_aug.segment(4, 16);
        
        // Reconstruct 4x4 matrix from vec(xx^T)
        Eigen::Matrix<tinytype, 4, 4> XX;
        int idx = 0;
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                XX(i, j) = vec_xx(idx++);
            }
        }
        
        // Build 5x5 moment matrix M = [1, x^T; x, XX]
        Eigen::Matrix<tinytype, 5, 5> M;
        M.setZero();
        M(0, 0) = 1.0;
        M.block<1, 4>(0, 1) = x_phys.transpose();
        M.block<4, 1>(1, 0) = x_phys;
        M.block<4, 4>(1, 1) = XX;
        
        // Project onto PSD cone using eigenvalue clipping
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<tinytype, 5, 5>> es;
        es.compute(M, Eigen::ComputeEigenvectors);
        auto eigenvals = es.eigenvalues();
        
        // Clip negative eigenvalues to small positive value
        for (int i = 0; i < 5; ++i) {
            eigenvals(i) = eigenvals(i) < 1e-8 ? 1e-8 : eigenvals(i);
        }
        
        // Reconstruct PSD matrix
        auto M_psd = es.eigenvectors() * eigenvals.asDiagonal() * es.eigenvectors().transpose();
        
        // Extract projected physical state
        tinytype alpha = std::max(1e-12, M_psd(0, 0));
        tinyVector x_proj = M_psd.block<4, 1>(1, 0) / alpha;
        
        // SAFETY MARGIN: Additional obstacle enforcement on physical state
        // NOTE: This is NOT in the original Julia formulation - it's an ADMM robustness enhancement
        // Julia relies purely on the linear constraint: px² + py² + 10*px ≥ -21
        // We add this because ADMM is iterative/approximate, unlike direct SDP solvers
        /*
        tinytype px = x_proj(0), py = x_proj(1);
        tinytype dist_to_obs = sqrt((px + 5.0)*(px + 5.0) + py*py);  // distance to obstacle at [-5, 0]
        if (dist_to_obs < 2.5) {  // 2.5 > 2.0 SAFETY MARGIN (25% larger than obstacle radius)
            // Push away from obstacle center
            tinytype scale = 2.5 / dist_to_obs;
            x_proj(0) = -5.0 + scale * (px + 5.0);  // new px
            x_proj(1) = scale * py;                  // new py
            // velocity components unchanged: x_proj(2), x_proj(3)
        }
        */
        
        // Extract quadratic matrix from PSD projection (pure Julia approach)
        Eigen::Matrix<tinytype, 4, 4> XX_proj = M_psd.block<4, 4>(1, 1) / alpha;
        
        // Vectorize projected quadratic matrix
        Eigen::Matrix<tinytype, 16, 1> vec_xx_proj;
        idx = 0;
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                vec_xx_proj(idx++) = XX_proj(i, j);
            }
        }
        
        // Update augmented state
        solver->work->vnew.col(k).head(4) = x_proj;
        solver->work->vnew.col(k).segment(4, 16) = vec_xx_proj;
    }
    
    // RESTORE initial condition - this should NEVER change!
    solver->work->vnew.col(0) = x0_initial;
}

/**
 * Joint state-control SDP projection for stronger relaxation
 * Projects 7x7 joint moment matrix [1; x; u; X; XU; UX; UU] onto PSD cone
 * This couples cross-terms to both x and u, enforcing symmetry UX = XU^T
 */
void project_sdp_state_control(TinySolver *solver) {
    // Only apply if we're using augmented formulation
    if (solver->work->vnew.rows() != 20 || solver->work->znew.rows() != 22) {
        std::cout << "[SDP] State-control projection SKIPPED (wrong dimensions: vnew=" << solver->work->vnew.rows() << ", znew=" << solver->work->znew.rows() << ")" << std::endl;
        return;
    }
    
    std::cout << "[SDP] State-control projection called!" << std::endl;
    
    for (int k = 0; k < solver->work->N - 1; ++k) {  // N-1 because controls are only defined for k < N
        // Extract pieces from vnew (x, X) and znew (u, XU, UX, UU)
        const auto& x_aug = solver->work->vnew.col(k);
        const auto& u_aug = solver->work->znew.col(k);

        // x (4)
        Eigen::Matrix<tinytype, 4, 1> x = x_aug.head<4>();

        // X (4x4) from vec(xx^T), column-major
        Eigen::Matrix<tinytype, 4, 4> X;
        int idx = 4;
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                X(i, j) = x_aug(idx++);
            }
        }

        // u (2)
        Eigen::Matrix<tinytype, 2, 1> u = u_aug.head<2>();

        // XU (4x2) from vec(xu^T), column-major (positions 2..9)
        Eigen::Matrix<tinytype, 4, 2> XU;
        idx = 2;
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 4; ++i) {
                XU(i, j) = u_aug(idx++);
            }
        }

        // UX (2x4) from vec(ux^T), column-major (positions 10..17)
        // We stored vec(ux^T) as 8 numbers; reconstruct ux^T (4x2) then transpose to UX (2x4)
        Eigen::Matrix<tinytype, 4, 2> uxT;
        idx = 10;
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 4; ++i) {
                uxT(i, j) = u_aug(idx++);
            }
        }
        Eigen::Matrix<tinytype, 2, 4> UX = uxT.transpose();

        // UU (2x2) from vec(uu^T) (positions 18..21)
        Eigen::Matrix<tinytype, 2, 2> UU;
        idx = 18;
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                UU(i, j) = u_aug(idx++);
            }
        }

        // Assemble 7x7 joint moment matrix
        Eigen::Matrix<tinytype, 7, 7> M;
        M.setZero();
        M(0, 0) = 1.0;
        M.block<1, 4>(0, 1) = x.transpose();
        M.block<1, 2>(0, 5) = u.transpose();
        M.block<4, 1>(1, 0) = x;
        M.block<4, 4>(1, 1) = X;
        M.block<4, 2>(1, 5) = XU;
        M.block<2, 1>(5, 0) = u;
        M.block<2, 4>(5, 1) = UX;
        M.block<2, 2>(5, 5) = UU;

        // PSD projection via eigenvalue clipping
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<tinytype, 7, 7>> es;
        es.compute(M, Eigen::ComputeEigenvectors);
        auto evals = es.eigenvalues();
        for (int i = 0; i < 7; ++i) {
            evals(i) = std::max<tinytype>(evals(i), 1e-8);
        }
        Eigen::Matrix<tinytype, 7, 7> Mpsd = es.eigenvectors() * evals.asDiagonal() * es.eigenvectors().transpose();

        // De-homogenize by alpha = M(0,0)
        tinytype alpha = std::max<tinytype>(Mpsd(0, 0), 1e-12);

        // Read back projected blocks
        Eigen::Matrix<tinytype, 4, 1> xP = Mpsd.block<4, 1>(1, 0) / alpha;
        Eigen::Matrix<tinytype, 2, 1> uP = Mpsd.block<2, 1>(5, 0) / alpha;
        Eigen::Matrix<tinytype, 4, 4> XP = Mpsd.block<4, 4>(1, 1) / alpha;
        Eigen::Matrix<tinytype, 4, 2> XUP = Mpsd.block<4, 2>(1, 5) / alpha;
        Eigen::Matrix<tinytype, 2, 4> UXP = Mpsd.block<2, 4>(5, 1) / alpha;
        Eigen::Matrix<tinytype, 2, 2> UUP = Mpsd.block<2, 2>(5, 5) / alpha;

        // Write back to vnew (x, X)
        solver->work->vnew.col(k).head<4>() = xP;
        idx = 4;
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                solver->work->vnew.col(k)(idx++) = XP(i, j);
            }
        }

        // Write back to znew (u, vec(xu^T), vec(ux^T), vec(uu^T))
        solver->work->znew.col(k).head<2>() = uP;
        
        // vec(xu^T)
        idx = 2;
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 4; ++i) {
                solver->work->znew.col(k)(idx++) = XUP(i, j);
            }
        }
        
        // vec(ux^T) = vec((UXP)^T), column-major
        idx = 10;
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 4; ++i) {
                solver->work->znew.col(k)(idx++) = UXP.transpose()(i, j);
            }
        }
        
        // vec(uu^T)
        idx = 18;
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                solver->work->znew.col(k)(idx++) = UUP(i, j);
            }
        }
    }
}
/**
 * Project a vector z onto a hyperplane defined by a^T z = b
 * Implements equation (21): ΠH(z) = z - (⟨z, a⟩ − b)/||a||² * a
 * @param z Vector to project
 * @param a Normal vector of the hyperplane
 * @param b Offset of the hyperplane
 * @return Projection of z onto the hyperplane
 */
tinyVector project_hyperplane(const tinyVector& z, const tinyVector& a, tinytype b) {
    tinytype dist = (a.dot(z) - b) / a.squaredNorm();
    return z - dist * a;
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
                solver->work->vcnew.block(start, i, num_xs, 1) = project_soc(col, mu);
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
                solver->work->zcnew.block(start, i, num_us, 1) = project_soc(col, mu);
            }
        }
    }
    
    // SDP constraints for augmented state - project moment matrices onto PSD cone
    project_sdp_augmented_state(solver);   // [1;x;X] ⪰ 0
    project_sdp_state_control(solver);     // [1;x;u;X;XU;UX;UU] ⪰ 0 (joint projection)
    
    // FIX A: RE-APPLY BOX AFTER PSD (so exported slacks respect bounds)
    if (solver->settings->en_input_bound) {
        solver->work->znew = solver->work->u_max.cwiseMin(
                             solver->work->u_min.cwiseMax(solver->work->znew));
    }
    if (solver->settings->en_state_bound) {
        solver->work->vnew = solver->work->x_max.cwiseMin(
                             solver->work->x_min.cwiseMax(solver->work->vnew));
    }
    
    // Update linear constraint slack variables for state
    if (solver->settings->en_state_linear) {
        solver->work->vlnew = solver->work->x + solver->work->gl;
    }

    // Update linear constraint slack variables for input
    if (solver->settings->en_input_linear) {
        solver->work->zlnew = solver->work->u + solver->work->yl;
    }

    // Linear constraints on state
    if (solver->settings->en_state_linear) {
        for (int i=0; i<solver->work->N; i++) {
            for (int k=0; k<solver->work->numStateLinear; k++) {
                tinyVector a = solver->work->Alin_x.row(k);
                tinytype b = solver->work->blin_x(k);
                tinytype constraint_value = a.dot(solver->work->vlnew.col(i));
                if (constraint_value > b) {  // Only project if constraint is violated
                    solver->work->vlnew.col(i) = project_hyperplane(solver->work->vlnew.col(i), a, b);
                }
            }
        }
    }

    // Linear constraints on input
    if (solver->settings->en_input_linear) {
        for (int i=0; i<solver->work->N-1; i++) {
            for (int k=0; k<solver->work->numInputLinear; k++) {
                tinyVector a = solver->work->Alin_u.row(k);
                tinytype b = solver->work->blin_u(k);
                tinytype constraint_value = a.dot(solver->work->zlnew.col(i));
                if (constraint_value > b) {  // Only project if constraint is violated
                    solver->work->zlnew.col(i) = project_hyperplane(solver->work->zlnew.col(i), a, b);
                }
            }
        }
    }

    // Update time-varying linear constraint slack variables for state
    if (solver->settings->en_tv_state_linear) {
        solver->work->vlnew_tv = solver->work->x + solver->work->gl_tv;
    }

    // Update time-varying linear constraint slack variables for input
    if (solver->settings->en_tv_input_linear) {
        solver->work->zlnew_tv = solver->work->u + solver->work->yl_tv;
    }

    // Time-varying Linear constraints on state
    if (solver->settings->en_tv_state_linear) {
        for (int i=0; i<solver->work->N; i++) {
            for (int k=0; k<solver->work->numtvStateLinear; k++) {
                tinyVector a = solver->work->tv_Alin_x.row((solver->work->numtvStateLinear*i) + k);
                tinytype b = solver->work->tv_blin_x(k,i);
                tinytype constraint_value = a.dot(solver->work->vlnew_tv.col(i));
                if (constraint_value > b) {  // Only project if constraint is violated
                    solver->work->vlnew_tv.col(i) = project_hyperplane(solver->work->vlnew_tv.col(i), a, b);
                }
            }
        }
    }

    // Time-varying Linear constraints on input
    if (solver->settings->en_tv_input_linear) {
        for (int i=0; i<solver->work->N-1; i++) {
            for (int k=0; k<solver->work->numtvInputLinear; k++) {
                tinyVector a = solver->work->tv_Alin_u.row((solver->work->numtvInputLinear*i) + k);
                tinytype b = solver->work->tv_blin_u(k,i);
                tinytype constraint_value = a.dot(solver->work->zlnew_tv.col(i));
                if (constraint_value > b) {  // Only project if constraint is violated
                    solver->work->zlnew_tv.col(i) = project_hyperplane(solver->work->zlnew_tv.col(i), a, b);
                }
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
    
    // Update linear constraint dual variables for state
    if (solver->settings->en_state_linear) {
        solver->work->gl = solver->work->gl + solver->work->x - solver->work->vlnew;
    }

    // Update linear constraint dual variables for input
    if (solver->settings->en_input_linear) {
        solver->work->yl = solver->work->yl + solver->work->u - solver->work->zlnew;
    }
        
    // Update time-varying linear constraint dual variables for state
    if (solver->settings->en_tv_state_linear) {
        solver->work->gl_tv = solver->work->gl_tv + solver->work->x - solver->work->vlnew_tv;
    }

    // Update time-varying linear constraint dual variables for input
    if (solver->settings->en_tv_input_linear) {
        solver->work->yl_tv = solver->work->yl_tv + solver->work->u - solver->work->zlnew_tv;
    }
}

/**
 * Update linear control cost terms in the Riccati feedback using the changing
 * slack and dual variables from ADMM
*/
void update_linear_cost(TinySolver *solver)
{
    // Update state cost terms: Reference + ADMM (linear terms handled separately if needed)
    // FIX 3a: Missing factor of 2 - expansion of (x-Xref)'Q(x-Xref) gives -2*Q*Xref, not -Q*Xref
    solver->work->q = -2.0 * (solver->work->Xref.array().colwise() * solver->work->Q.array());
    (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vnew - solver->work->g);
    if (solver->settings->en_state_soc && solver->work->numStateCones > 0) {
        (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vcnew - solver->work->gc);
    }
    if (solver->settings->en_state_linear) {
        (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vlnew - solver->work->gl);
    }
    if (solver->settings->en_tv_state_linear) {
        (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vlnew_tv - solver->work->gl_tv);
    }

    // Update input cost terms: Reference + ADMM (linear terms handled separately if needed)
    // FIX 3a: Missing factor of 2 - expansion of (u-Uref)'R(u-Uref) gives -2*R*Uref, not -R*Uref
    solver->work->r = -2.0 * (solver->work->Uref.array().colwise() * solver->work->R.array());
    (solver->work->r).noalias() -= solver->cache->rho * (solver->work->znew - solver->work->y);
    if (solver->settings->en_input_soc && solver->work->numInputCones > 0) {
        (solver->work->r).noalias() -= solver->cache->rho * (solver->work->zcnew - solver->work->yc);
    }
    if (solver->settings->en_input_linear) {
        (solver->work->r).noalias() -= solver->cache->rho * (solver->work->zlnew - solver->work->yl);
    }
    if (solver->settings->en_tv_input_linear) {
        (solver->work->r).noalias() -= solver->cache->rho * (solver->work->zlnew_tv - solver->work->yl_tv);
    }

    // Update terminal cost
    solver->work->p.col(solver->work->N - 1) = -(solver->work->Xref.col(solver->work->N - 1).transpose().lazyProduct(solver->cache->Pinf));
    (solver->work->p.col(solver->work->N - 1)).noalias() -= solver->cache->rho * (solver->work->vnew.col(solver->work->N - 1) - solver->work->g.col(solver->work->N - 1));

    if (solver->settings->en_state_soc && solver->work->numStateCones > 0) {
        solver->work->p.col(solver->work->N - 1) -= solver->cache->rho * (solver->work->vcnew.col(solver->work->N - 1) - solver->work->gc.col(solver->work->N - 1));
    }
    if (solver->settings->en_state_linear) {
        solver->work->p.col(solver->work->N - 1) -= solver->cache->rho * (solver->work->vlnew.col(solver->work->N - 1) - solver->work->gl.col(solver->work->N - 1));
    }
    if (solver->settings->en_tv_state_linear) {
        solver->work->p.col(solver->work->N - 1) -= solver->cache->rho * (solver->work->vlnew_tv.col(solver->work->N - 1) - solver->work->gl_tv.col(solver->work->N - 1));
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

    // Initialize linear constraint slack variables if needed
    if (solver->settings->en_state_linear) {
        solver->work->vlnew = solver->work->x;
    }
    
    if (solver->settings->en_input_linear) {
        solver->work->zlnew = solver->work->u;
    }

    // Initialize time-varying linear constraint slack variables if needed
    if (solver->settings->en_tv_state_linear) {
        solver->work->vlnew_tv = solver->work->x;
    }
    
    if (solver->settings->en_tv_input_linear) {
        solver->work->zlnew_tv = solver->work->u;
    }

    for (int i = 0; i < solver->settings->max_iter; i++)
    {
        // Solve linear system with Riccati and roll out to get new trajectory
        backward_pass_grad(solver);

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
      
    }
    
    solver->solution->iter = solver->work->iter;
    solver->solution->solved = 0;
    solver->solution->x = solver->work->vnew;
    solver->solution->u = solver->work->znew;
    return 1;
}

} /* extern "C" */
