# # Differentiating a solve
#
# The [companion page](parameter_identification.md) differentiates a constitutive model at a
# material point. This one differentiates a *solve* — a discretised boundary value problem,
# with a mesh, a Newton loop and a time loop between the parameter and the answer.
#
# That is the harder claim, and it is where a library either delivers or does not. A
# constitutive law that accepts a `Dual` is worth little if the assembly allocates its
# element arrays as `Float64`, or if the solver's convergence test takes a square root of a
# residual on its way to zero. Both were true here until this page was written; both are
# recorded below rather than quietly fixed, because they are the failure modes anyone doing
# this will meet.

using PoroMechanics
using Ferrite
using Tensors
using VoronoiFVM
using ExtendableGrids
using ForwardDiff
using SparseArrays
using Printf

# ## Finite volumes, steady state
#
# A one-dimensional unsaturated column with a pressure imposed at each end, solved with
# `VoronoiFVM` through [`fvm_system`](@ref). The parameters differentiated are the intrinsic
# permeability and the coefficients of the retention and relative-permeability curves.
#
# The system is built with `valuetype = eltype(θ)`, so the whole solve — assembly, Newton,
# linear algebra — happens in dual numbers.

Base.@kwdef struct SteadyColumn{T, R, K} <: AbstractPoroModel
    rho_l::T = 1.0e3        # liquid density [kg/m³]
    k_int::T = 1.0e-18      # intrinsic permeability [m²]
    mu_l::T = 1.0e-3        # dynamic viscosity [Pa·s]
    p_g::T = 1.0e5          # gas pressure [Pa]
    p_left::T = -5.0e6      # suction imposed at x = 0 [Pa]
    retention::R = VanGenuchten(1.5e6, 0.06)
    rel_perm::K = Mualem(3.0e6, 0.5)
end

## Promote rather than require a single type: a Dual in one parameter leaves the rest Float64.
function SteadyColumn(rho_l, k_int, mu_l, p_g, p_left, retention, rel_perm)
    return SteadyColumn(promote(rho_l, k_int, mu_l, p_g, p_left)..., retention, rel_perm)
end

PoroMechanics.nspecies(::SteadyColumn) = 1
PoroMechanics.species_names(::SteadyColumn) = [:p_l]

conductivity(m::SteadyColumn, pc) =
    m.rho_l * m.k_int / m.mu_l * relative_permeability(m.rel_perm, pc)

function PoroMechanics.flux!(f, u, edge, m::SteadyColumn, ::Any)
    f[1] = conductivity(m, m.p_g - (u[1, 1] + u[1, 2]) / 2) * (u[1, 1] - u[1, 2])
    return nothing
end

PoroMechanics.storage!(f, u, ::Any, m::SteadyColumn, ::Any) = (f[1] = 0 * u[1]; nothing)

function PoroMechanics.bcondition!(f, u, bnode, m::SteadyColumn, ::Any)
    boundary_dirichlet!(f, u, bnode; species = 1, region = 1, value = m.p_left)
    boundary_dirichlet!(f, u, bnode; species = 1, region = 2, value = m.p_g)
    return nothing
end

"Steady mass flux through the column, as a function of four scaled parameters."
function column_flux(θ; N = 41, L = 0.2)
    T = eltype(θ)
    m = SteadyColumn(
        k_int = θ[1] * 1.0e-18,
        retention = VanGenuchten(θ[2] * 1.5e6, θ[3] * 0.06),
        rel_perm = Mualem(θ[4] * 3.0e6, 0.5),
    )
    grid = simplexgrid(range(0.0, L; length = N))
    sys = fvm_system(m, grid; valuetype = T)
    inival = unknowns(sys)
    inival[1, :] .= T(-1.0e6)
    sol = solve(sys; inival, control = VoronoiFVM.SolverControl(; reltol = 1.0e-10, verbose = ""))
    u = sol[1, :]
    return conductivity(m, m.p_g - (u[1] + u[2]) / 2) * (u[1] - u[2]) / (L / (N - 1))
end

flux_names = ("k_int", "van Genuchten a", "van Genuchten m", "Mualem a")
θ_flux = ones(4)
grad_flux = ForwardDiff.gradient(column_flux, θ_flux)
grad_flux_fd = map(eachindex(θ_flux)) do i
    h = 1.0e-6
    θp = copy(θ_flux); θp[i] += h
    θm = copy(θ_flux); θm[i] -= h
    (column_flux(θp) - column_flux(θm)) / (2h)
end

@printf("steady flux: %.6e kg/m²/s\n\n", column_flux(θ_flux))
println("  parameter          automatic differentiation   central differences")
for i in eachindex(θ_flux)
    @printf("  %-17s  %+.8e             %+.8e\n", flux_names[i], grad_flux[i], grad_flux_fd[i])
end

# The permeability and the Mualem coefficient agree to ten digits. The two retention
# parameters come back as exactly zero from both — and here that is not a subtlety but a
# statement about the problem: at steady state the retention curve enters only through the
# storage term, which is identically zero. No amount of measurement of a steady flux will
# ever determine them.

# ## Finite volumes, transient — blocked upstream
#
# The same trick does not work for a transient `VoronoiFVM` solve, and the reason is worth
# recording because it is not a property of this package.
#
# `VoronoiFVM` builds its boundary-node object with the *coordinate* type rather than the
# value type,
#
# ```julia
# # JF: We need to be able to distinguish between dirichlet type and value type.
# # So far we will use Tc for the dirichlet type instead of the valuetype.
# BNode(sys::AbstractSystem{Tv,Tc,Ti,Tm}, time, embedparam; partition = 1) =
#     BNode{Tc, Tc, Ti}(sys, time, embedparam)
# ```
#
# — an acknowledged compromise in `vfvm_geometryitems.jl`. In a stationary solve nothing
# notices. In a transient one the solver promotes the time to the system's value type, and
# a `Dual` time cannot be stored in a node whose fields are `Float64`. The solve fails at
# initialisation with a conversion error.
#
# Two ways round it exist and neither is free. `VoronoiFVM` supports declared parameters
# (`nparams`, `parameters(node)`) and assembles ``\partial R/\partial\theta`` alongside the
# Jacobian, which is the raw material for propagating sensitivities forward across time
# steps by the implicit function theorem — the storage Jacobian at the previous step has to
# be supplied separately. Or the time loop can be written outside the package. Both are more
# than this page needs, and the honest summary today is: steady state yes, transient not
# through this backend.

# ## Finite elements, transient and nonlinear
#
# Where the time loop belongs to this package, none of that applies. What follows is an
# axisymmetric elastoplastic problem taken through twelve load steps: at each step a global
# Newton with a line search, and at every quadrature point of every iteration a return
# mapping — itself a Newton solve. The whole thing is differentiated with respect to three
# Barcelona Basic Model parameters.

grid_fe = Ferrite.generate_grid(
    Ferrite.Quadrilateral, (2, 2), Ferrite.Vec(1.0, 0.0), Ferrite.Vec(2.0, 1.0)
)
ip_fe = Ferrite.Lagrange{Ferrite.RefQuadrilateral, 1}()^2
dh_fe = Ferrite.DofHandler(grid_fe)
Ferrite.add!(dh_fe, :u, ip_fe)
Ferrite.close!(dh_fe)
qr_fe = Ferrite.QuadratureRule{Ferrite.RefQuadrilateral}(2)
cv_fe = Ferrite.CellValues(qr_fe, ip_fe, Ferrite.Lagrange{Ferrite.RefQuadrilateral, 1}())

"Mean stress carried by the sample after a prescribed compaction, as a function of κ, λ(0), e₀."
function mobilised_stress(θ; nsteps = 12, εv_total = 0.08)
    T = eltype(θ)
    m = BBM(κ = θ[1] * 0.011, λ0 = θ[2] * 0.065, e0 = θ[3] * (1 / 3))
    nq = Ferrite.getnquadpoints(cv_fe)
    σ0 = -1.0e3 * one(SymmetricTensor{2, 3, T})
    fresh() = [
        [initial_state(m, σ0, T(4.0e4); suction = zero(T)) for _ in 1:nq]
            for _ in 1:Ferrite.getncells(grid_fe)
    ]
    states, states_old = fresh(), fresh()

    ## The sparsity pattern comes from the mesh; only the numbers become dual.
    K = convert(SparseMatrixCSC{T, Int}, Ferrite.allocate_matrix(dh_fe))
    f = zeros(T, Ferrite.ndofs(dh_fe))
    u = zeros(T, Ferrite.ndofs(dh_fe))

    for k in 1:nsteps
        εv = εv_total * k / nsteps
        ch = Ferrite.ConstraintHandler(dh_fe)
        for (set, comp) in (("left", 1), ("right", 1), ("bottom", 2), ("top", 2))
            Ferrite.add!(
                ch, Ferrite.Dirichlet(
                    :u, Ferrite.getfacetset(grid_fe, set), (x, t) -> -εv / 3 * x[comp], [comp]
                )
            )
        end
        Ferrite.close!(ch)
        Ferrite.update!(ch, 0.0)
        newton_solve!(
            u, K, f, dh_fe, cv_fe, m, states, states_old, ch, 1.0;
            linsolve = (A, b) -> Matrix(A) \ b,
        )
        for c in eachindex(states)
            states_old[c] .= states[c]
        end
    end
    return mean_pressure(states_old[1][1].σ)
end

fe_names = ("κ", "λ(0)", "e₀")
θ_fe = ones(3)
grad_fe = ForwardDiff.gradient(mobilised_stress, θ_fe)
grad_fe_fd = map(eachindex(θ_fe)) do i
    h = 1.0e-6
    θp = copy(θ_fe); θp[i] += h
    θm = copy(θ_fe); θm[i] -= h
    (mobilised_stress(θp) - mobilised_stress(θm)) / (2h)
end

@printf("mean stress after 8 %% compaction: %.4f kPa\n\n", mobilised_stress(θ_fe) / 1.0e3)
println("  parameter   automatic differentiation   central differences")
for i in eachindex(θ_fe)
    @printf("  %-9s   %+.9e            %+.9e\n", fe_names[i], grad_fe[i], grad_fe_fd[i])
end

# Agreement to about ``5\times10^{-10}``, through a path that has no closed form and through
# two nested Newton solves.
#
# The signs are the physics: a larger ``\kappa`` is a softer elastic law, so the sample
# reaches a given compaction at a lower stress; a larger ``\lambda(0)`` is a flatter virgin
# compression line, likewise; a larger void ratio stiffens both ``K = \bar p(1+e_0)/\kappa``
# and the hardening, so the stress rises.

# ## What it took
#
# Four things had to be true, and three of them were not.
#
# **Model parameters must be typed by a parameter, not by `Float64`.** The constitutive
# layer already obeyed this; the model structs in `examples/` did not, so a `Dual` could
# enter a retention curve but not a permeability. They now carry a type parameter and a
# promoting constructor, which is what lets one field become a `Dual` while the rest stay
# `Float64`.
#
# **Element arrays must take their type from the system.** `assemble_axisymmetric!`
# allocated `zeros(n, n)`, which confines differentiability to the constitutive layer and
# stops it at the mesh — silently, with a conversion error far from the cause.
#
# **Convergence tests must not take square roots of quantities going to zero.** `√x` has an
# infinite slope at the origin, so a residual norm that reaches zero returns a correct value
# and a `NaN` derivative. Both the global Newton and the stress-control loop now compare
# squares. The same shape of problem, at the apex of the deviatoric cone, is what
# [`equivalent_stress`](@ref) guards against.
#
# **The linear solver must be replaceable.** Sparse factorisations are `Float64`-only, so a
# tangent full of `Dual`s cannot go to UMFPACK. `newton_solve!` takes a `linsolve` keyword
# for exactly this.
#
# None of these is deep. All of them are invisible until something tries to differentiate
# through the whole stack, which is the argument for having a page that does.
