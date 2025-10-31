using Pkg
Pkg.activate(".")

using LinearAlgebra 
using Plots 
using Convex 
using Mosek 
using MosekTools 

# ------------------------------- Problem setup ------------------------------ #
#problem parameters
N = 31 #number of timesteps
x_initial = [-10; 0.1; 0; 0]
x_obs = [-5.0, 0.0]  #position of the obstacle center
r_obs = 2 #radius of the obstacle 


q_xx = 0.1
r_xx = 10.0
R_xx = 500.0
reg = 1e-6

# ------------------------------ System dynamics ----------------------------- #

#dimensions
nx = 4  # state dimension x = [position, velocity]
nu = 2 # controls dimension u = [acceleration]

nxx = 16 #number of elements in xx'
nxu = 8 #number of elements in ux'
nux = 8 #number of elements in xu'
nuu = 4 #number of elements in uu'

Ad = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1]
Bd = [0.5 0; 0 0.5; 1 0; 0 1]

A = [Ad zeros(nx, nxx); 
     zeros(nxx, nx) kron(Ad, Ad)]

B = [Bd zeros(nx, nxu + nux + nuu); 
     zeros(nxx, nu) kron(Bd, Ad) kron(Ad, Bd) kron(Bd, Bd)]

# ------------------------- SDP generation and solve ------------------------- #
#weights 
Q = zeros(nx + nxx, nx + nxx) + reg*Matrix(I, nx + nxx, nx + nxx)
q = [zeros(nx); vec(q_xx*Matrix(I, nx, nx))]

R = Diagonal([zeros(nu); zeros(nxu+nux); vec(R_xx*Matrix(I, nu, nu))])+reg*Matrix(I, nu + nxu + nux + nuu, nu + nxu + nux + nuu)
r = [zeros(nu + nxu + nux); vec(r_xx*Matrix(I, nu, nu))]

#decision variables
x_bar = Variable(nx + nxx, N)
u_bar = Variable(nu + nxu + nux + nuu, N-1) 

obj = 0
constraints = []
for k=1:N

    # initial condition
    if k == 1
        push!(constraints, x_bar[:,1] == [x_initial; vec(x_initial*x_initial')])
    end

    #dynamics constraints 
    if k < N
        push!(constraints, x_bar[:,k+1] == A*x_bar[:,k] + B*u_bar[:,k])
    end

    #PSD constraints 
    x = x_bar[1:nx,k]
    XX = reshape(x_bar[nx+1:end,k], nx, nx)
    if k < N
        u = u_bar[1:nu,k]
        XU =  reshape(u_bar[nu+1: nu + nxu, k], nx, nu)
        UX = reshape(u_bar[nu+nxu + 1: nu + nxu + nux, k], nu, nx)
        UU = reshape(u_bar[nu+nxu+nux+1:end, k], nu, nu)
        push!(constraints, [1 x' u';
                            x XX XU;
                            u UX UU]⪰ 0)
    else
        push!(constraints, [1 x'; 
                            x XX] ⪰ 0)
    end

    # collision avoidance
    m = [-2*x_obs[1]; -2*x_obs[2]; zeros(2); 1; zeros(4); 1; zeros(10)]'
    n = -x_obs'*x_obs + r_obs^2
    push!(constraints, m*x_bar[:,k] >= n)


    # cost function
    global obj += quadform(x_bar[:,k], Q) + q'*x_bar[:,k]
    if k < N
        global obj += quadform(u_bar[:,k], R) + r'*u_bar[:,k] 
    end

end

#solve problem
problem = minimize(obj, constraints)
solve!(problem, () -> Mosek.Optimizer())
println("Problem Status: ", problem.status)

# --------------------------------- Analysis --------------------------------- #
#extract solution
x_bar_opt = x_bar.value
u_bar_opt = u_bar.value 
x_opt = x_bar_opt[1:4,:]
u_opt = u_bar_opt[1:2,:]
Xres = zeros(nx, nx, N-1)
check = zeros(N)
for i=1:N 
    check[i] = sum(x_opt[:,i]*x_opt[:,i]' - reshape(x_bar_opt[nx+1:end, i], nx,nx))
end

# plot trajectory
θ = range(0, 2π, length=200)  # angles
p0 = plot(aspect_ratio=1, title="double integrator trajectory")


x_obstacle = x_obs[1] .+ r_obs .* cos.(θ)
y_obstacle = x_obs[2] .+ r_obs .* sin.(θ)
plot!(p0, x_obstacle, y_obstacle, seriestype=:shape, label="obstacle", color=:grey)

# p0 = plot(x_obstacle, y_obstacle, aspect_ratio=1, seriestype=:shape, label="obstacle", color=:grey, title="double integrator trajectory")
scatter!(p0, x_opt[1,:], x_opt[2,:], label="trajectory")
scatter!(p0, [x_initial[1]], [x_initial[2]], label="initial position")
scatter!(p0, [0], [0], label="final position")

# plot states
time_steps = 1:N
p1 = plot(time_steps, x_opt[1,:], label="x₁ (position x)", linewidth=2)
plot!(p1, time_steps, x_opt[2,:], label="x₂ (position y)", linewidth=2)
plot!(p1, time_steps, x_opt[3,:], label="x₃ (velocity x)", linewidth=2)
plot!(p1, time_steps, x_opt[4,:], label="x₄ (velocity y)", linewidth=2)
title!(p1, "States (x)")
xlabel!(p1, "Time Step")
ylabel!(p1, "State Value")

# Control inputs plot
p2 = plot(1:N-1, u_opt[1,:], label="u₁ (acceleration x)", linewidth=2)
plot!(p2, 1:N-1, u_opt[2,:], label="u₂ (acceleration y)", linewidth=2)
title!(p2, "Controls (u)")
xlabel!(p2, "Time Step")
ylabel!(p2, "Control Value")

# Residual check plot
p3 = plot(time_steps, check, label="residual", linewidth=2)
title!(p3, "Residual Check")
xlabel!(p3, "Time Step")
ylabel!(p3, "‖X-x@x'‖")

# Combine all plots
comb_plots = plot(p0, p1, p2, p3, layout=(4,1), size=(800, 1200))

savefig(comb_plots, "combined_plots.png") 