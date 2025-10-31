// Obstacle Avoidance with SDP Constraints (Julia translation) — ADMM debug harness

#define NX_PHYS 4
#define NU_PHYS 2
#define NX_AUG 20   // [x(4); vec(xx^T)(16)]
#define NU_AUG 22   // [u(2); vec(xu^T)(8); vec(ux^T)(8); vec(uu^T)(4)]

#define NHORIZON 31

#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <tinympc/tiny_api.hpp>
#include <vector>
#include <string>

using std::cout;
using std::endl;

typedef Matrix<tinytype, NX_AUG, NX_AUG> MatAA;
typedef Matrix<tinytype, NX_AUG, NU_AUG> MatAB;
typedef Matrix<tinytype, NX_AUG, 1> VecA;
typedef Matrix<tinytype, NU_AUG, 1> VecU;
typedef Matrix<tinytype, NX_PHYS, NX_PHYS> MatXX;
typedef Matrix<tinytype, NX_PHYS, 1> VecX;
typedef Matrix<tinytype, NU_PHYS, 1> VecUc;

// ---------------- Physical dynamics (matches Julia) ----------------
static const tinytype A_phys_data[NX_PHYS * NX_PHYS] = {
    1.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0
};

static const tinytype B_phys_data[NX_PHYS * NU_PHYS] = {
    0.5, 0.0,
    0.0, 0.5,
    1.0, 0.0,
    0.0, 1.0
};

template<int M, int N, int P, int Q>
static Eigen::Matrix<tinytype, M*P, N*Q> kron(const Eigen::Matrix<tinytype, M, N> &A,
                                             const Eigen::Matrix<tinytype, P, Q> &B) {
    Eigen::Matrix<tinytype, M*P, N*Q> K;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            K.template block<P, Q>(i*P, j*Q) = A(i,j)*B;
        }
    }
    return K;
}

static void build_augmented_A(MatAA &A_aug) {
    using namespace Eigen;
    Matrix<tinytype, NX_PHYS, NX_PHYS, RowMajor> A_phys =
        Map<Matrix<tinytype, NX_PHYS, NX_PHYS, RowMajor>>(const_cast<tinytype*>(A_phys_data));
    A_aug.setZero();
    A_aug.block<NX_PHYS, NX_PHYS>(0, 0) = A_phys;
    auto kron_AA = kron<NX_PHYS, NX_PHYS, NX_PHYS, NX_PHYS>(A_phys, A_phys);
    A_aug.block<16,16>(NX_PHYS, NX_PHYS) = kron_AA;
}

static void build_augmented_B(MatAB &B_aug) {
    using namespace Eigen;
    Matrix<tinytype, NX_PHYS, NX_PHYS, RowMajor> A_phys =
        Map<Matrix<tinytype, NX_PHYS, NX_PHYS, RowMajor>>(const_cast<tinytype*>(A_phys_data));
    Matrix<tinytype, NX_PHYS, NU_PHYS, RowMajor> B_phys =
        Map<Matrix<tinytype, NX_PHYS, NU_PHYS, RowMajor>>(const_cast<tinytype*>(B_phys_data));
    B_aug.setZero();
    B_aug.block<NX_PHYS, NU_PHYS>(0, 0) = B_phys;
    auto kron_BA = kron<NX_PHYS, NU_PHYS, NX_PHYS, NX_PHYS>(B_phys, A_phys);  // 16x8
    auto kron_AB = kron<NX_PHYS, NX_PHYS, NX_PHYS, NU_PHYS>(A_phys, B_phys);  // 16x8
    auto kron_BB = kron<NX_PHYS, NU_PHYS, NX_PHYS, NU_PHYS>(B_phys, B_phys);  // 16x4
    B_aug.block<16, 8>(NX_PHYS, NU_PHYS) = kron_BA;            // vec(xu^T)
    B_aug.block<16, 8>(NX_PHYS, NU_PHYS + 8) = kron_AB;        // vec(ux^T)
    B_aug.block<16, 4>(NX_PHYS, NU_PHYS + 16) = kron_BB;       // vec(uu^T)
}

// ---------------- Julia weights and linear terms ----------------
// reg = 1e-6; q_xx = 0.1; R_xx = 500.0; r_xx = 10.0
static constexpr tinytype REG = 1e-6;
static constexpr tinytype Q_xx = 0.1;
static constexpr tinytype R_xx = 500.0;
static constexpr tinytype r_xx = 10.0;

static void build_Q(VecA &Qdiag) {
    Qdiag.setConstant(REG);
}

static void build_q(VecA &q) {
    q.setZero();
    // vec(q_xx*I_4) into positions [4..19]
    // column-major packing of 4x4 identity scaled by Q_xx
    int idx = NX_PHYS;
    for (int j = 0; j < NX_PHYS; ++j) {
        for (int i = 0; i < NX_PHYS; ++i) {
            q(idx++) = (i == j) ? Q_xx : 0.0;
        }
    }
}

static void build_R(VecU &Rdiag) {
    Rdiag.setConstant(REG);
    // uu^T diagonal contribution at tail (positions 18 and 21)
    Rdiag(18) = R_xx + REG;
    Rdiag(21) = R_xx + REG;
}

static void build_r(VecU &r) {
    r.setZero();
    // vec(r_xx*I_2) into uu^T part (positions 18..21) — column-major
    r(18) = r_xx;
    r(21) = r_xx;
}

// ---------------- Collision avoidance constraint (Julia) ----------------
static constexpr tinytype OBS_CX = -5.0;
static constexpr tinytype OBS_CY =  0.0;
static constexpr tinytype OBS_R  =  2.0;

static void build_collision_constraint(Eigen::Matrix<tinytype, 1, NX_AUG> &m, tinytype &n) {
    m.setZero();
    // m = [-2*obs_x, -2*obs_y, 0, 0, 1, 0,0,0, 0,1, 0..0]
    m(0,0) = -2.0 * OBS_CX;  // px
    m(0,1) = -2.0 * OBS_CY;  // py
    m(0,4) = 1.0;            // px^2 (X00)
    m(0,9) = 1.0;            // py^2 (X11)
    n = -(OBS_CX*OBS_CX + OBS_CY*OBS_CY) + OBS_R*OBS_R;
}

// Utility: build augmented state from physical x
static VecA lift_state_from_phys(const VecX &x) {
    VecA x_aug;
    x_aug.head<NX_PHYS>() = x;
    MatXX XX = x * x.transpose();
    int idx = NX_PHYS;
    for (int j = 0; j < NX_PHYS; ++j) {
        for (int i = 0; i < NX_PHYS; ++i) {
            x_aug(idx++) = XX(i,j);
        }
    }
    return x_aug;
}

extern "C" {

static void parse_args(int argc, char** argv,
                      tinytype &rho,
                      tinytype &alpha,
                      int &state_sdp,
                      int &input_sdp,
                      tinytype &lin_scale) {
    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 1; i + 1 < args.size(); ++i) {
        if (args[i] == std::string("--rho")) rho = std::stod(args[i+1]);
        if (args[i] == std::string("--alpha")) alpha = std::stod(args[i+1]);
        if (args[i] == std::string("--state-sdp")) state_sdp = std::stoi(args[i+1]);
        if (args[i] == std::string("--input-sdp")) input_sdp = std::stoi(args[i+1]);
        if (args[i] == std::string("--lin-scale")) lin_scale = std::stod(args[i+1]);
    }
}

int main(int argc, char** argv) {
    cout << "[ADMM-DEBUG] Obstacle avoidance (Julia SDP)" << endl;
    // Tunables (override via CLI)
    tinytype rho = 100.0;         // --rho <value>
    tinytype alpha_blend = 0.6;   // --alpha <value>
    int state_sdp = 1;            // --state-sdp 0/1
    int input_sdp = 1;            // --input-sdp 0/1
    tinytype LIN_SCALE = 1.0;     // --lin-scale <value>
    parse_args(argc, argv, rho, alpha_blend, state_sdp, input_sdp, LIN_SCALE);

    // Augmented dynamics
    MatAA A_aug; MatAB B_aug; A_aug.setZero(); B_aug.setZero();
    build_augmented_A(A_aug);
    build_augmented_B(B_aug);
    VecA f_aug = VecA::Zero();

    // Costs (diagonal) and linear vectors
    VecA Qdiag; VecU Rdiag; VecA q; VecU r;
    build_Q(Qdiag);
    build_R(Rdiag);
    build_q(q);
    build_r(r);

    // Setup solver
    TinySolver *solver = nullptr;
    int status = tiny_setup(&solver,
                            A_aug, B_aug, f_aug,
                            Qdiag.asDiagonal(), Rdiag.asDiagonal(),
                            rho, NX_AUG, NU_AUG, NHORIZON, /*verbose=*/0);
    if (status) { cout << "setup failed: " << status << endl; return status; }

    // PURE Julia: only linear collision constraint
    Eigen::Matrix<tinytype, 1, NX_AUG> m;
    tinytype n;
    build_collision_constraint(m, n);
    tinyMatrix A_lin_x(NHORIZON, NX_AUG);
    tinyVector b_lin_x(NHORIZON);
    for (int k = 0; k < NHORIZON; ++k) { A_lin_x.row(k) = -m; b_lin_x(k) = -n; }
    tinyMatrix A_lin_u(0, NU_AUG);
    tinyVector b_lin_u(0);
    status = tiny_set_linear_constraints(solver, A_lin_x, b_lin_x, A_lin_u, b_lin_u);
    if (status) { cout << "set linear constraints failed: " << status << endl; return status; }

    // Parity with Julia: no box constraints
    solver->settings->en_state_bound = 0;
    solver->settings->en_input_bound = 0;
    solver->settings->en_state_linear = 1;
    solver->settings->en_input_linear = 0;
    solver->settings->en_tv_state_linear = 0;
    solver->settings->en_tv_input_linear = 0;
    // SDP projections (configurable)
    solver->settings->en_state_sdp = state_sdp;
    solver->settings->en_input_sdp = input_sdp;
    solver->settings->abs_pri_tol = 1e-3;
    solver->settings->abs_dua_tol = 1e-3;
    solver->settings->check_termination = 1;
    solver->settings->max_iter = 200;
    // Blend between projections and LQR
    solver->settings->forward_blend_alpha = alpha_blend;
    cout << "Params: rho=" << rho
         << ", alpha=" << alpha_blend
         << ", state_sdp=" << state_sdp
         << ", input_sdp=" << input_sdp
         << ", lin_scale=" << LIN_SCALE << endl;

    // Build Xref and Uref
    // Linear injection scaling; 1.0 mirrors Julia, 0.0 disables linear pull
    const tinytype LIN_SCALE_LOCAL = LIN_SCALE;
    tinyMatrix Xref = tinyMatrix::Zero(NX_AUG, NHORIZON);
    tinyMatrix Uref = tinyMatrix::Zero(NU_AUG, NHORIZON-1);
    VecA Xref_one; VecU Uref_one;
    for (int i = 0; i < NX_AUG; ++i) Xref_one(i) = (Qdiag(i) != 0.0) ? LIN_SCALE_LOCAL * (-0.5 * q(i) / Qdiag(i)) : 0.0;
    for (int i = 0; i < NU_AUG; ++i) Uref_one(i) = (Rdiag(i) != 0.0) ? LIN_SCALE_LOCAL * (-0.5 * r(i) / Rdiag(i)) : 0.0;
    for (int k = 0; k < NHORIZON; ++k) Xref.col(k) = Xref_one;
    for (int k = 0; k < NHORIZON-1; ++k) Uref.col(k) = Uref_one;
    solver->work->Xref = Xref;
    solver->work->Uref = Uref;

    // Initial and goal (physical)
    VecX x0_phys; x0_phys << -10.0, 0.1, 0.0, 0.0;
    VecX xg_phys; xg_phys << 0.0, 0.0, 0.0, 0.0;
    VecA x0_aug = lift_state_from_phys(x0_phys);
    VecA xg_aug = lift_state_from_phys(xg_phys);

    // Initialize trajectory (straight line in physical space, lifted)
    for (int k = 0; k < NHORIZON; ++k) {
        tinytype a = static_cast<tinytype>(k) / (NHORIZON - 1);
        VecX xi = (1.0 - a) * x0_phys + a * xg_phys;
        solver->work->x.col(k) = lift_state_from_phys(xi);
    }
    solver->work->x.col(0) = x0_aug;     // hard set initial
    solver->work->vnew.col(0) = x0_aug;  // initialize slack
    solver->work->Xref.col(0) = x0_aug;  // encourage equality at k=0
    // Terminal goal
    solver->work->Xref.col(NHORIZON-1) = xg_aug;

    // ---------------- Use library solve for parity ----------------
    auto t0 = std::chrono::high_resolution_clock::now();
    int rc = tiny_solve(solver);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cout << "Solve time (tiny_solve): " << ms << " ms, status=" << rc << ", iters=" << solver->solution->iter << "\n";

    // Populate solution like tiny_solve would
    solver->solution->solved = 1;
    solver->solution->iter = solver->work->iter;
    solver->solution->x = solver->work->vnew;
    solver->solution->u = solver->work->znew;

    // ---------------- Quick safety/consistency checks ----------------
    const auto &Xsol = solver->solution->x;
    int violations = 0; tinytype min_dist = 1e9;
    for (int k = 0; k < NHORIZON; ++k) {
        VecX xp = Xsol.col(k).head<NX_PHYS>();
        tinytype dx = xp(0) - OBS_CX, dy = xp(1) - OBS_CY;
        tinytype d = std::sqrt(dx*dx + dy*dy);
        min_dist = std::min(min_dist, d);
        if (d < OBS_R) violations++;
    }
    cout << "Safety violations: " << violations << "/" << NHORIZON
         << ", min distance: " << min_dist << endl;

    // Check initial condition preserved
    cout << "x[0] phys (solution): [" << Xsol(0,0) << ", " << Xsol(1,0)
         << ", " << Xsol(2,0) << ", " << Xsol(3,0) << "]\n";

    // Save physical rollout from controls (optional CSV)
    std::ofstream f("obstacle_avoidance_admm_debug.csv");
    f << "# k, px, py, vx, vy, ux, uy\n";
    MatXX Aphys = Map<const Matrix<tinytype, NX_PHYS, NX_PHYS, RowMajor>>(A_phys_data);
    Matrix<tinytype, NX_PHYS, NU_PHYS> Bphys = Map<const Matrix<tinytype, NX_PHYS, NU_PHYS, RowMajor>>(B_phys_data);
    VecX xroll = x0_phys;
    for (int k = 0; k < NHORIZON; ++k) {
        f << k << ", " << xroll(0) << ", " << xroll(1) << ", " << xroll(2) << ", " << xroll(3);
        if (k < NHORIZON-1) {
            VecUc u = solver->solution->u.col(k).head<NU_PHYS>();
            f << ", " << u(0) << ", " << u(1) << "\n";
            xroll = Aphys * xroll + Bphys * u;
        } else {
            f << ", 0, 0\n";
        }
    }
    f.close();
    cout << "Saved trajectory: obstacle_avoidance_admm_debug.csv" << endl;

    return 0;
}

} // extern "C"
