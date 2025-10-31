// Minimal ADMM solver for the Julia lifted SDP obstacle-avoidance problem
// Mirrors tiny_sdp_big_v2.jl but replaces Mosek with a custom ADMM.

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using std::cout;
using std::endl;

// Dimensions
static constexpr int NX_PHYS = 4;
static constexpr int NU_PHYS = 2;
static constexpr int NX_AUG  = 20;  // [x(4); vec(xx^T)(16)]
static constexpr int NU_AUG  = 22;  // [u(2); vec(xu^T)(8); vec(ux^T)(8); vec(uu^T)(4)]
static constexpr int N       = 31;  // horizon

// Julia parameters
static constexpr double q_xx = 0.1;
static constexpr double r_xx = 10.0;
static constexpr double R_xx_default = 500.0;
static constexpr double reg_default  = 1e-6;

// Obstacle
static constexpr double OBS_CX = -5.0;
static constexpr double OBS_CY =  0.0;
static constexpr double OBS_R  =  2.0;

// Physical dynamics (dt=1)
static const double A_phys_data[NX_PHYS*NX_PHYS] = {
    1,0,1,0,
    0,1,0,1,
    0,0,1,0,
    0,0,0,1
};
static const double B_phys_data[NX_PHYS*NU_PHYS] = {
    0.5,0,
    0,0.5,
    1,0,
    0,1
};

template<int M, int N_, int P, int Q>
static Eigen::Matrix<double, M*P, N_*Q> kron(const Eigen::Matrix<double,M,N_>& A,
                                             const Eigen::Matrix<double,P,Q>& B) {
    Eigen::Matrix<double, M*P, N_*Q> K;
    for (int i=0;i<M;++i) for (int j=0;j<N_;++j)
        K.template block<P,Q>(i*P, j*Q) = A(i,j)*B;
    return K;
}

static void build_augmented_A(Eigen::Matrix<double, NX_AUG, NX_AUG>& A_aug) {
    using MatA = Eigen::Matrix<double, NX_PHYS, NX_PHYS, Eigen::RowMajor>;
    MatA A_phys = Map<const MatA>(A_phys_data);
    A_aug.setZero();
    A_aug.template block<NX_PHYS,NX_PHYS>(0,0) = A_phys;
    auto KAA = kron<NX_PHYS,NX_PHYS,NX_PHYS,NX_PHYS>(A_phys, A_phys);
    A_aug.template block<16,16>(NX_PHYS,NX_PHYS) = KAA;
}

static void build_augmented_B(Eigen::Matrix<double, NX_AUG, NU_AUG>& B_aug) {
    using MatA = Eigen::Matrix<double, NX_PHYS, NX_PHYS, Eigen::RowMajor>;
    using MatB = Eigen::Matrix<double, NX_PHYS, NU_PHYS, Eigen::RowMajor>;
    MatA A_phys = Map<const MatA>(A_phys_data);
    MatB B_phys = Map<const MatB>(B_phys_data);
    B_aug.setZero();
    B_aug.template block<NX_PHYS,NU_PHYS>(0,0) = B_phys;
    auto KBA = kron<NX_PHYS,NU_PHYS,NX_PHYS,NX_PHYS>(B_phys, A_phys);  // 16x8
    auto KAB = kron<NX_PHYS,NX_PHYS,NX_PHYS,NU_PHYS>(A_phys, B_phys);  // 16x8
    auto KBB = kron<NX_PHYS,NU_PHYS,NX_PHYS,NU_PHYS>(B_phys, B_phys);  // 16x4
    B_aug.template block<16,8>(NX_PHYS, NU_PHYS)      = KBA;  // vec(xu^T)
    B_aug.template block<16,8>(NX_PHYS, NU_PHYS + 8)  = KAB;  // vec(ux^T)
    B_aug.template block<16,4>(NX_PHYS, NU_PHYS + 16) = KBB;  // vec(uu^T)
}

// Cost diags and linear terms
static void build_Qdiag(Eigen::Matrix<double, NX_AUG, 1>& Qdiag, double reg) { Qdiag.setConstant(reg); }
static void build_q(Eigen::Matrix<double, NX_AUG, 1>& q, double q_scale) {
    q.setZero();
    int idx = NX_PHYS;
    for (int j=0;j<NX_PHYS;++j)
        for (int i=0;i<NX_PHYS;++i)
            q(idx++) = (i==j) ? (q_scale * q_xx) : 0.0;
}
static void build_Rdiag(Eigen::Matrix<double, NU_AUG, 1>& Rdiag, double reg, double R_xx) {
    Rdiag.setConstant(reg);
    Rdiag(18) = R_xx + reg; // uu^T diag entries
    Rdiag(21) = R_xx + reg;
}
static void build_r(Eigen::Matrix<double, NU_AUG, 1>& r, double r_scale) {
    r.setZero();
    r(18) = r_scale * r_xx;
    r(21) = r_scale * r_xx;
}

// Collision: m x_bar >= n
static void build_collision(Eigen::Matrix<double,1,NX_AUG>& m, double& n) {
    m.setZero();
    m(0,0) = -2.0*OBS_CX;  // px
    m(0,1) = -2.0*OBS_CY;  // py
    m(0,4) = 1.0;          // px^2
    m(0,9) = 1.0;          // py^2
    n = -(OBS_CX*OBS_CX + OBS_CY*OBS_CY) + OBS_R*OBS_R;
}

static Eigen::Matrix<double, NX_AUG, 1> lift_state(const Eigen::Matrix<double,NX_PHYS,1>& x) {
    Eigen::Matrix<double, NX_AUG, 1> xa;
    xa.head<NX_PHYS>() = x;
    Eigen::Matrix<double, NX_PHYS, NX_PHYS> XX = x * x.transpose();
    int idx = NX_PHYS;
    for (int j=0;j<NX_PHYS;++j)
        for (int i=0;i<NX_PHYS;++i)
            xa(idx++) = XX(i,j);
    return xa;
}

// Projection helpers
static void project_halfspace(Eigen::Matrix<double,NX_AUG,1>& x,
                              const Eigen::Matrix<double,1,NX_AUG>& a,
                              double b) {
    double ax = (a * x)(0,0);
    if (ax < b) {
        double t = (ax - b) / a.squaredNorm();
        x = x - t * a.transpose();
    }
}

static void project_state_psd(Eigen::Matrix<double,NX_AUG,1>& x_aug) {
    Eigen::Matrix<double, NX_PHYS, 1> x = x_aug.head<NX_PHYS>();
    Eigen::Matrix<double,4,4> XX;
    int idx = NX_PHYS;
    for (int j=0;j<4;++j) for (int i=0;i<4;++i) XX(i,j)=x_aug(idx++);
    Eigen::Matrix<double,5,5> M; M.setZero();
    M(0,0)=1.0; M.block<1,4>(0,1)=x.transpose(); M.block<4,1>(1,0)=x; M.block<4,4>(1,1)=XX;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,5,5>> es(M);
    auto evals = es.eigenvalues();
    for (int i=0;i<5;++i) if (evals(i)<1e-8) evals(i)=1e-8;
    auto Mpsd = es.eigenvectors()*evals.asDiagonal()*es.eigenvectors().transpose();
    double alpha = std::max(1e-12, Mpsd(0,0));
    Eigen::Matrix<double,4,1> xP = Mpsd.block<4,1>(1,0)/alpha;
    Eigen::Matrix<double,4,4> XXP = Mpsd.block<4,4>(1,1)/alpha;
    // write back
    x_aug.head<4>() = xP;
    idx = NX_PHYS;
    for (int j=0;j<4;++j) for (int i=0;i<4;++i) x_aug(idx++) = XXP(i,j);
}

static void project_joint_psd(Eigen::Matrix<double,NX_AUG,1>& x_aug,
                              Eigen::Matrix<double,NU_AUG,1>& u_aug) {
    // extract
    Eigen::Matrix<double,4,1> x = x_aug.head<4>();
    Eigen::Matrix<double,4,4> X;
    int idx = 4; for (int j=0;j<4;++j) for (int i=0;i<4;++i) X(i,j) = x_aug(idx++);
    Eigen::Matrix<double,2,1> u = u_aug.head<2>();
    Eigen::Matrix<double,4,2> XU; idx = 2; for (int j=0;j<2;++j) for (int i=0;i<4;++i) XU(i,j) = u_aug(idx++);
    Eigen::Matrix<double,2,4> UX; {
        Eigen::Matrix<double,4,2> uxT; idx = 10; for (int j=0;j<2;++j) for (int i=0;i<4;++i) uxT(i,j)=u_aug(idx++);
        UX = uxT.transpose();
    }
    Eigen::Matrix<double,2,2> UU; idx = 18; for (int j=0;j<2;++j) for (int i=0;i<2;++i) UU(i,j)=u_aug(idx++);
    // assemble 7x7
    Eigen::Matrix<double,7,7> M; M.setZero();
    M(0,0)=1.0; M.block<1,4>(0,1)=x.transpose(); M.block<1,2>(0,5)=u.transpose();
    M.block<4,1>(1,0)=x; M.block<4,4>(1,1)=X; M.block<4,2>(1,5)=XU;
    M.block<2,1>(5,0)=u; M.block<2,4>(5,1)=UX; M.block<2,2>(5,5)=UU;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,7,7>> es(M);
    auto evals = es.eigenvalues(); for (int i=0;i<7;++i) if (evals(i)<1e-8) evals(i)=1e-8;
    auto Mpsd = es.eigenvectors()*evals.asDiagonal()*es.eigenvectors().transpose();
    double alpha = std::max(1e-12, Mpsd(0,0));
    // read back
    Eigen::Matrix<double,4,1> xP = Mpsd.block<4,1>(1,0)/alpha;
    Eigen::Matrix<double,2,1> uP = Mpsd.block<2,1>(5,0)/alpha;
    Eigen::Matrix<double,4,4> XP = Mpsd.block<4,4>(1,1)/alpha;
    Eigen::Matrix<double,4,2> XUP = Mpsd.block<4,2>(1,5)/alpha;
    Eigen::Matrix<double,2,4> UXP = Mpsd.block<2,4>(5,1)/alpha;
    Eigen::Matrix<double,2,2> UUP = Mpsd.block<2,2>(5,5)/alpha;
    // write back
    x_aug.head<4>() = xP;
    idx = 4; for (int j=0;j<4;++j) for (int i=0;i<4;++i) x_aug(idx++) = XP(i,j);
    u_aug.head<2>() = uP;
    idx = 2; for (int j=0;j<2;++j) for (int i=0;i<4;++i) u_aug(idx++) = XUP(i,j);
    idx = 10; for (int j=0;j<2;++j) for (int i=0;i<4;++i) u_aug(idx++) = UXP.transpose()(i,j);
    idx = 18; for (int j=0;j<2;++j) for (int i=0;i<2;++i) u_aug(idx++) = UUP(i,j);
}

int main(int argc, char** argv) {
    cout << "[ADMM-BAREBONE] Julia lifted SDP via ADMM" << endl;
    // Build augmented dynamics
    Eigen::Matrix<double, NX_AUG, NX_AUG> A_aug; build_augmented_A(A_aug);
    Eigen::Matrix<double, NX_AUG, NU_AUG> B_aug; build_augmented_B(B_aug);

    // Costs (tunable via CLI)
    double reg_val = reg_default;           // --reg <float>
    double Rxx_val = R_xx_default;          // --Rxx <float>
    double q_scale = 1.0;                   // --q-scale <float>
    double r_scale = 1.0;                   // --r-scale <float>
    {
        std::vector<std::string> args(argv, argv + argc);
        auto get = [&](const std::string& flag, std::string &out){
            for (size_t i=1;i+1<args.size();++i) if (args[i]==flag){ out=args[i+1]; return true;} return false; };
        std::string sval;
        if (get("--reg", sval)) reg_val = std::stod(sval);
        if (get("--Rxx", sval)) Rxx_val = std::stod(sval);
        if (get("--q-scale", sval)) q_scale = std::stod(sval);
        if (get("--r-scale", sval)) r_scale = std::stod(sval);
    }
    Eigen::Matrix<double, NX_AUG, 1> Qdiag; build_Qdiag(Qdiag, reg_val);
    Eigen::Matrix<double, NU_AUG, 1> Rdiag; build_Rdiag(Rdiag, reg_val, Rxx_val);
    Eigen::Matrix<double, NX_AUG, 1> q; build_q(q, q_scale);
    Eigen::Matrix<double, NU_AUG, 1> r; build_r(r, r_scale);

    // Linear constraint
    Eigen::Matrix<double,1,NX_AUG> m; double n; build_collision(m, n);

    // Initial/goal (physical)
    Eigen::Matrix<double, NX_PHYS, 1> x0_phys; x0_phys << -10.0, 0.1, 0.0, 0.0;
    Eigen::Matrix<double, NX_PHYS, 1> xg_phys; xg_phys << 0.0, 0.0, 0.0, 0.0;
    Eigen::Matrix<double, NX_AUG, 1> x0_aug = lift_state(x0_phys);

    // Stack variable p = [X1..XN, U1..U_{N-1}]
    const int NXN   = NX_AUG * N;
    const int NUNm1 = NU_AUG * (N-1);
    const int PDIM  = NXN + NUNm1;

    // Build cost diagonal (Qhat diag) and linear c
    VectorXd Hdiag(PDIM); Hdiag.setZero();
    VectorXd c(PDIM);      c.setZero();
    for (int k=0;k<N;++k) {
        Hdiag.segment(k*NX_AUG, NX_AUG) = Qdiag;
        c.segment(k*NX_AUG, NX_AUG)     = q;
    }
    for (int k=0;k<N-1;++k) {
        Hdiag.segment(NXN + k*NU_AUG, NU_AUG) = Rdiag;
        c.segment(NXN + k*NU_AUG, NU_AUG)     = r;
    }

    // Equality constraints M p = b: initial condition, dynamics, terminal goal
    const int MEQ = NX_AUG /*x1=x0*/ + NX_AUG*(N-1) /*dynamics*/ + NX_AUG /*xN = xg*/;
    MatrixXd M = MatrixXd::Zero(MEQ, PDIM);
    VectorXd b = VectorXd::Zero(MEQ);
    int row = 0;
    // x1 = x0
    M.block(row, 0, NX_AUG, NX_AUG) = MatrixXd::Identity(NX_AUG, NX_AUG);
    b.segment(row, NX_AUG) = x0_aug;
    row += NX_AUG;
    // x_{k+1} = A x_k + B u_k
    for (int k=0;k<N-1;++k) {
        int col_xk   = k*NX_AUG;
        int col_xkp1 = (k+1)*NX_AUG;
        int col_uk   = NXN + k*NU_AUG;
        // x_{k+1}
        M.block(row, col_xkp1, NX_AUG, NX_AUG) += MatrixXd::Identity(NX_AUG, NX_AUG);
        // -A x_k
        M.block(row, col_xk, NX_AUG, NX_AUG)   -= A_aug;
        // -B u_k
        M.block(row, col_uk, NX_AUG, NU_AUG)   -= B_aug;
        row += NX_AUG;
    }
    // x_N = xg (lifted)
    Eigen::Matrix<double, NX_AUG, 1> xg_aug = lift_state(xg_phys);
    M.block(row, (N-1)*NX_AUG, NX_AUG, NX_AUG) = MatrixXd::Identity(NX_AUG, NX_AUG);
    b.segment(row, NX_AUG) = xg_aug;

    // ADMM parameters (CLI tunable)
    double rho = 50.0;                 // --rho <float>
    double rho_psd = -1.0;             // --rho-psd <float>
    double rho_lin = -1.0;             // --rho-lin <float>
    int max_iter = 1000;               // --max-iter <int>
    double eps_pri = 1e-3;             // --eps-pri <float>
    double eps_dua = 1e-3;             // --eps-dua <float>
    double relax = 1.5;                // --relax <float> (1.0 = none)
    int use_joint_psd = 0;             // --joint-psd 0/1
    int alt_proj_passes = 2;           // --alt-proj <int> alternating PSD<->halfspace passes
    double z_prox = 0.0;               // --z-prox <float> (damping on z-update)
    int warmup_joint_after = 200;      // --warmup-joint <iters> (enable joint PSD after warmup)

    std::vector<std::string> args(argv, argv + argc);
    auto get = [&](const std::string& flag, std::string &out){
        for (size_t i=1;i+1<args.size();++i) if (args[i]==flag){ out=args[i+1]; return true;} return false; };
    std::string sval;
    if (get("--rho", sval)) rho = std::stod(sval);
    if (get("--max-iter", sval)) max_iter = std::stoi(sval);
    if (get("--eps-pri", sval)) eps_pri = std::stod(sval);
    if (get("--eps-dua", sval)) eps_dua = std::stod(sval);
    if (get("--relax", sval)) relax = std::stod(sval);
    if (get("--rho-psd", sval)) rho_psd = std::stod(sval);
    if (get("--rho-lin", sval)) rho_lin = std::stod(sval);
    if (get("--joint-psd", sval)) use_joint_psd = std::stoi(sval);
    if (get("--alt-proj", sval)) alt_proj_passes = std::stoi(sval);
    if (get("--z-prox", sval)) z_prox = std::stod(sval);
    if (get("--warmup-joint", sval)) warmup_joint_after = std::stoi(sval);

    if (rho_psd < 0) rho_psd = rho;
    if (rho_lin < 0) rho_lin = rho;

    cout << "Params: rho=" << rho << " (psd=" << rho_psd << ", lin=" << rho_lin << ")"
         << ", relax=" << relax
         << ", joint_psd=" << use_joint_psd
         << ", alt_proj=" << alt_proj_passes
         << ", z_prox=" << z_prox
         << ", warmup_joint_after=" << warmup_joint_after
         << ", max_iter=" << max_iter
         << ", eps(pri,dua)=(" << eps_pri << "," << eps_dua << ")" << endl;

    auto build_system = [&](double rho_val_psd, double rho_val_lin,
                            VectorXd &Hinvd_out,
                            MatrixXd &MH_out,
                            Eigen::LDLT<MatrixXd> &ldlt_out) {
        double rho_sum = rho_val_psd + rho_val_lin;
        Hinvd_out = (Hdiag.array() + rho_sum).inverse().matrix();
        MH_out = M;
        for (int j=0;j<PDIM;++j) MH_out.col(j) *= Hinvd_out(j);
        MatrixXd S = MH_out * M.transpose();
        ldlt_out.compute(S);
    };

    VectorXd Hinvd; MatrixXd MH; Eigen::LDLT<MatrixXd> S_ldlt;
    build_system(rho_psd, rho_lin, Hinvd, MH, S_ldlt);

    // Variables
    VectorXd p = VectorXd::Zero(PDIM);
    VectorXd z_psd = VectorXd::Zero(PDIM);
    VectorXd z_lin = VectorXd::Zero(PDIM);
    VectorXd u_psd = VectorXd::Zero(PDIM);
    VectorXd u_lin = VectorXd::Zero(PDIM);

    auto idx_x = [&](int k){ return k*NX_AUG; };
    auto idx_u = [&](int k){ return NXN + k*NU_AUG; };

    // Initialize z with a simple straight-line lift for states, zeros for controls
    for (int k=0;k<N;++k) {
        double a = double(k)/double(N-1);
        Eigen::Matrix<double,NX_PHYS,1> xi = (1.0-a)*x0_phys + a*xg_phys;
        auto xi_aug = lift_state(xi);
        z_psd.segment(idx_x(k), NX_AUG) = xi_aug;
        z_lin.segment(idx_x(k), NX_AUG) = xi_aug;
    }

    cout << "Starting ADMM..." << endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it=0; it<max_iter; ++it) {
        // y-update: minimize
        // 0.5 p'Hp + c'p + (rho_psd/2)||p - z_psd + u_psd||^2 + (rho_lin/2)||p - z_lin + u_lin||^2
        // s.t. M p = b
        // H' = H + (rho_psd + rho_lin) I,  g' = rho_psd*(z_psd - u_psd) + rho_lin*(z_lin - u_lin) - c
        VectorXd g = rho_psd*(z_psd - u_psd) + rho_lin*(z_lin - u_lin) - c;
        VectorXd DHg = Hinvd.array() * g.array();   // H'^-1 g'
        // Solve (M H'^-1 M^T) μ = b - M H'^-1 g'
        VectorXd rhs = b - MH * g;
        VectorXd mu  = S_ldlt.solve(rhs);
        // Recover p = H'^-1 (g' + M^T μ)
        p = DHg + (Hinvd.array() * (M.transpose()*mu).array()).matrix();

        // Over-relaxation
        VectorXd p_hat = relax * p + (1.0 - relax) * ((z_psd + z_lin) * 0.5);

        // z_psd update: PSD projections only
        VectorXd z_psd_old = z_psd;
        for (int k=0;k<N;++k) {
            Eigen::Matrix<double,NX_AUG,1> xk = p_hat.segment(idx_x(k), NX_AUG) + u_psd.segment(idx_x(k), NX_AUG);
            if (k==0) xk = x0_aug;
            if (k < N-1) {
                Eigen::Matrix<double,NU_AUG,1> uk = p_hat.segment(idx_u(k), NU_AUG) + u_psd.segment(idx_u(k), NU_AUG);
                int passes = std::max(1, alt_proj_passes);
                for (int ap=0; ap<passes; ++ap) {
                    bool do_joint = use_joint_psd || (it >= warmup_joint_after);
                    if (do_joint) project_joint_psd(xk, uk);
                    else project_state_psd(xk);
                }
                z_psd.segment(idx_x(k), NX_AUG) = xk;
                z_psd.segment(idx_u(k), NU_AUG) = uk;
            } else {
                int passes = std::max(1, alt_proj_passes);
                for (int ap=0; ap<passes; ++ap) project_state_psd(xk);
                z_psd.segment(idx_x(k), NX_AUG) = xk;
            }
        }

        // z_lin update: halfspace projections only (states)
        VectorXd z_lin_old = z_lin;
        for (int k=0;k<N;++k) {
            Eigen::Matrix<double,NX_AUG,1> xk = p_hat.segment(idx_x(k), NX_AUG) + u_lin.segment(idx_x(k), NX_AUG);
            if (k==0) xk = x0_aug;
            int passes = std::max(1, alt_proj_passes);
            for (int ap=0; ap<passes; ++ap) project_halfspace(xk, m, n);
            z_lin.segment(idx_x(k), NX_AUG) = xk;
            if (k < N-1) {
                z_lin.segment(idx_u(k), NU_AUG) = p_hat.segment(idx_u(k), NU_AUG) + u_lin.segment(idx_u(k), NU_AUG);
            }
        }

        // Optional z-prox damping (convex combination towards previous z)
        if (z_prox > 0.0) {
            double gamma_psd = rho_psd / (rho_psd + z_prox);
            double gamma_lin = rho_lin / (rho_lin + z_prox);
            z_psd = gamma_psd * z_psd + (1.0 - gamma_psd) * z_psd_old;
            z_lin = gamma_lin * z_lin + (1.0 - gamma_lin) * z_lin_old;
        }

        // dual updates
        u_psd = u_psd + (p - z_psd);
        u_lin = u_lin + (p - z_lin);

        // residuals (max over both constraints)
        double pri = std::max((p - z_psd).lpNorm<Eigen::Infinity>(), (p - z_lin).lpNorm<Eigen::Infinity>());
        double dua = std::max((rho_psd * (z_psd - z_psd_old)).lpNorm<Eigen::Infinity>(),
                              (rho_lin * (z_lin - z_lin_old)).lpNorm<Eigen::Infinity>());

        if (it % 10 == 0)
            cout << "iter " << it << ": pri=" << pri << ", dua=" << dua << endl;

        if (pri < eps_pri && dua < eps_dua) {
            cout << "Converged at iter " << it << endl; break;
        }

        // Adaptive rho every 25 iterations
        if (it>0 && it%25==0) {
            double scale_up = 2.0, scale_down = 0.5;
            bool rebuild = false;
            if (pri > 10.0*dua) { rho_psd *= scale_up; rho_lin *= scale_up; rebuild = true; }
            else if (dua > 10.0*pri) { rho_psd *= scale_down; rho_lin *= scale_down; rebuild = true; }
            if (rebuild) {
                build_system(rho_psd, rho_lin, Hinvd, MH, S_ldlt);
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cout << "ADMM solve time: " << ms << " ms" << endl;

    // ============ DIAGNOSTICS ============
    cout << "\n=== SOLUTION DIAGNOSTICS ===" << endl;
    
    // Check constraint violations and compute cost
    double total_cost = 0.0;
    double max_constraint_viol = 0.0;
    double min_obs_dist = 1e9;
    
    Eigen::Matrix<double,NX_PHYS,NX_PHYS,Eigen::RowMajor> Aphys = Map<const Eigen::Matrix<double,NX_PHYS,NX_PHYS,Eigen::RowMajor>>(A_phys_data);
    Eigen::Matrix<double,NX_PHYS,NU_PHYS,Eigen::RowMajor> Bphys = Map<const Eigen::Matrix<double,NX_PHYS,NU_PHYS,Eigen::RowMajor>>(B_phys_data);
    
    for (int k=0; k<N; ++k) {
        Eigen::Matrix<double,NX_AUG,1> xk = z_psd.segment(idx_x(k), NX_AUG);
        double px = xk(0), py = xk(1);
        
        // Obstacle distance
        double dist = std::sqrt((px - OBS_CX)*(px - OBS_CX) + (py - OBS_CY)*(py - OBS_CY));
        min_obs_dist = std::min(min_obs_dist, dist);
        
        // Constraint: m*x >= n  =>  px² + py² + 10*px >= -21
        double constr_val = (m * xk)(0,0);
        double viol = n - constr_val;
        if (viol > max_constraint_viol) max_constraint_viol = viol;
        
        // State cost
        total_cost += (Qdiag.array() * xk.array() * xk.array()).sum() + q.dot(xk);
        
        // Control cost
        if (k < N-1) {
            Eigen::Matrix<double,NU_AUG,1> uk = z_psd.segment(idx_u(k), NU_AUG);
            total_cost += (Rdiag.array() * uk.array() * uk.array()).sum() + r.dot(uk);
        }
    }
    
    cout << "Total Cost: " << total_cost << endl;
    cout << "Min Obstacle Distance: " << min_obs_dist << " (obstacle radius: " << OBS_R << ")" << endl;
    cout << "Max Constraint Violation: " << max_constraint_viol << " (should be <= 0)" << endl;
    if (min_obs_dist < OBS_R) {
        cout << "WARNING: Trajectory COLLIDES with obstacle!" << endl;
    } else {
        cout << "SUCCESS: Trajectory avoids obstacle" << endl;
    }
    
    // Export trajectory to CSV (physical only) using PSD copy (z_psd)
    std::ofstream f("obstacle_avoidance_admm_barebone.csv");
    f << "# k, px, py, vx, vy, ux, uy\n";
    Eigen::Matrix<double,NX_PHYS,1> xroll = x0_phys;
    for (int k=0;k<N;++k) {
        f << k << ", " << xroll(0) << ", " << xroll(1) << ", " << xroll(2) << ", " << xroll(3);
        if (k < N-1) {
            Eigen::Matrix<double,NU_PHYS,1> up = (z_psd.segment(idx_u(k), NU_AUG)).head<NU_PHYS>();
            f << ", " << up(0) << ", " << up(1) << "\n";
            xroll = Aphys * xroll + Bphys * up;
        } else {
            f << ", 0, 0\n";
        }
    }
    f.close();
    cout << "\nSaved obstacle_avoidance_admm_barebone.csv" << endl;
    return 0;
}
