# Conservation harness — does a scheme conserve what it claims to transport?
#
# `element_balance_error` in `examples/chloride_ingress/element_balance.jl` compares a
# state before and after a chemical step. That is the right question for a **closed** cell
# and the wrong one for a cell the transport runs through, where matter legitimately
# crosses the faces. The identity that holds in general is
#
#     d/dt ∫_Ω T dx  =  Σ_Γ  ∫_Γ J·n ds
#
# and `VoronoiFVM` computes the right-hand side exactly, through its test functions:
# `integrate(sys, tf, u, uold, Δt)` with `tf = 1` on one boundary region and `0` on the
# others returns the influx through that region, already accounting for the storage
# change, the reaction term and the source.
#
# ## The point of the second testset
#
# Applied to a scheme's *own* storage function, the identity is exact by construction and
# the harness measures nothing — `integrate` and the assembly use the same expression. It
# only bites when the inventory is defined **independently of the scheme**, from the
# physics. That is what separates
#
#     T = φc + S(c)              the inventory
#     T = (φ + K_d) c            the linearized retardation, K_d frozen over a step
#
# which agree to first order and not at all as a conserved quantity. The second testset
# below is written to fail if the harness cannot see that difference, because a
# conservation check that never fails is not a check.

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using LinearAlgebra: norm

# ── The harness ───────────────────────────────────────────────────────────────

"""
    inventory(sys, u, F = sys.physics.storage) -> Vector

`∫_Ω F(u) dx` per species, summed over cell regions. Pass `F` explicitly to integrate an
inventory the scheme does **not** use as its storage — which is the only way the harness
can contradict the scheme.
"""
function inventory(sys, u, F = sys.physics.storage)
    I = VoronoiFVM.integrate(sys, F, u)
    return [sum(@view I[i, :]) for i in axes(I, 1)]
end

"""
    boundary_influx(sys, u, uold, Δt, regions) -> Matrix   (nspec × nregion)

Influx through each boundary region, by the test-function identity. Column `k` is region
`regions[k]`: its test function is 1 there and 0 on every other listed region.
"""
function boundary_influx(sys, u, uold, Δt, regions)
    factory = TestFunctionFactory(sys)
    cols = map(regions) do r
        tf = testfunction(factory, [q for q in regions if q != r], [r])
        VoronoiFVM.integrate(sys, tf, u, uold, Δt)
    end
    return reduce(hcat, cols)
end

"""
    conservation_defect(sys, tsol, regions; F) -> (worst, step, scale)

Worst relative violation of `d/dt ∫T = Σ_Γ influx` over the steps of `tsol`, the step it
occurs at, and the scale it is relative to. `F` defaults to the scheme's own storage, in
which case the identity is exact by construction — pass the physical inventory instead to
make the check bite.
"""
function conservation_defect(sys, tsol, regions; F = sys.physics.storage)
    worst, at, scale = 0.0, 0, 0.0
    for k in 2:length(tsol.t)
        u, uold = tsol.u[k], tsol.u[k - 1]
        Δt = tsol.t[k] - tsol.t[k - 1]
        Δt > 0 || continue
        rate = (inventory(sys, u, F) .- inventory(sys, uold, F)) ./ Δt
        influx = sum(boundary_influx(sys, u, uold, Δt, regions); dims = 2) |> vec
        ## The scale is the size of the terms being differenced, not of their difference:
        ## a defect is only meaningful next to the flux that produced it.
        s = max(norm(rate), norm(influx))
        d = norm(rate .- influx) / max(s, eps())
        d > worst && ((worst, at, scale) = (d, k, s))
    end
    return worst, at, scale
end

# ── A model whose adsorption is not linear ────────────────────────────────────

"""
Tracer with a Langmuir isotherm `S(c) = K c / (1 + b c)`, in two storage flavours:

  * `:inventory`   — `storage = φc + S(c)`, the conserved quantity itself;
  * `:retardation` — `storage = (φ + K_d) c` with `K_d = S'(c★)` frozen at `c★`, which is
    what `run_4.jl` does between two chemistry passes.

Both are legitimate discretisations of the same PDE and they differ in what they conserve.
"""
Base.@kwdef struct SorbingTracer{T, B} <: PoroMechanics.AbstractPoroModel
    phi::T = 0.3
    D::T = 1.0e-9
    K::T = 2.0
    b::T = 5.0
    flavour::Symbol = :inventory
    kd::T = 0.0                     # frozen tangent, used by :retardation only
    ## A type parameter, not `::Any`: a heterogeneous boundary tuple behind an abstract
    ## field boxes, and VoronoiFVM reports the allocations per facet and per Newton step.
    dirichlet::B = ((1, 1.0),)
end

PoroMechanics.nspecies(::SorbingTracer) = 1
PoroMechanics.species_names(::SorbingTracer) = [:c]

## Langmuir rather than a Freundlich `K√c`: the latter has an infinite slope at `c = 0`,
## which is exactly the initial condition here, and `ForwardDiff` returns `Inf` there.
sorbed(m::SorbingTracer, c) = m.K * c / (one(c) + m.b * c)

function PoroMechanics.storage!(f, u, node, m::SorbingTracer, ::Any)
    return f[1] = m.flavour === :inventory ?
        m.phi * u[1] + sorbed(m, u[1]) :
        (m.phi + m.kd) * u[1]
end

function PoroMechanics.flux!(f, u, edge, m::SorbingTracer, ::Any)
    return f[1] = m.D * (u[1, 1] - u[1, 2])
end

function PoroMechanics.bcondition!(f, u, bnode, m::SorbingTracer, ::Any)
    return PoroMechanics.apply_dirichlet!(f, u, bnode, m.dirichlet)
end

## The physical inventory, as a node function, whatever the scheme stores.
true_inventory!(m) = (f, u, node, data = nothing) -> (f[1] = m.phi * u[1] + sorbed(m, u[1]))

# ── Tests ─────────────────────────────────────────────────────────────────────

@testset "conservation harness" begin
    grid = simplexgrid(range(0, 1; length = 41))
    ctrl = VoronoiFVM.SolverControl(;
        Δt = 1.0e3, Δt_min = 1.0e-3, Δt_max = 1.0e7, Δu_opt = 1.0
    )

    @testset "the identity holds for the scheme's own storage" begin
        ## Fick, `c = 1` imposed at region 1, sealed at region 2. Nothing here is
        ## approximate: `integrate` and the assembly evaluate the same storage.
        m = FickModel(; phi = 0.3, D = 1.0e-9, dirichlet = ((1, 1.0),))
        sys = fvm_system(m, grid)
        inival = unknowns(sys)
        inival .= 0.0
        tsol = solve(sys; inival, times = [0.0, 1.0e8], control = ctrl)

        worst, at, _ = conservation_defect(sys, tsol, [1, 2])
        @test worst < 1.0e-12

        ## The sealed boundary passes nothing, and the inventory only grows through
        ## region 1 — the two statements the harness rests on.
        k = length(tsol.t)
        Δt = tsol.t[k] - tsol.t[k - 1]
        flux = boundary_influx(sys, tsol.u[k], tsol.u[k - 1], Δt, [1, 2])
        @test abs(flux[1, 2]) < 1.0e-20 * max(abs(flux[1, 1]), 1.0)
        @test flux[1, 1] > 0
    end

    @testset "and fails for a frozen retardation" begin
        ## Same PDE, same isotherm, two storages. The inventory form conserves the
        ## inventory; the frozen-tangent form conserves `(φ + K_d)c`, which is not it.
        common = (phi = 0.3, D = 1.0e-9, K = 2.0, b = 5.0, dirichlet = ((1, 1.0),))
        m_inv = SorbingTracer(; common..., flavour = :inventory)

        ## `K_d = S'(c★)` at a representative concentration — exactly the tangent
        ## `run_4.jl` freezes, and the choice that makes the comparison fair.
        c_star = 0.5
        kd = m_inv.K / (1 + m_inv.b * c_star)^2
        m_ret = SorbingTracer(; common..., flavour = :retardation, kd = kd)

        defects = map((m_inv, m_ret)) do m
            sys = fvm_system(m, grid)
            inival = unknowns(sys)
            inival .= 0.0
            tsol = solve(sys; inival, times = [0.0, 1.0e8], control = ctrl)
            first(conservation_defect(sys, tsol, [1, 2]; F = true_inventory!(m)))
        end

        ## The inventory form is exact.
        @test defects[1] < 1.0e-12

        ## The frozen retardation is not, and by a margin no tolerance would absorb.
        ## This assertion is the harness's own test: if it ever passes silently, the
        ## harness has stopped measuring anything.
        @test defects[2] > 1.0e-3
        @test defects[2] > 1.0e6 * defects[1]
        @info "conservation" inventaire = defects[1] retardation_gelée = defects[2]

        ## Read the second number for what it is. `K_d` is frozen here for the **whole**
        ## transient, at a `c★` the solution spends most of its time far from, so 0.79 is
        ## an upper bound on the mechanism, not a claim about `run_4.jl` — which refreshes
        ## `K_d` at every chemistry pass and therefore freezes it over a much shorter
        ## interval. Measuring the real figure on `run_4` is what this harness now makes
        ## possible, and it belongs to the SNIA/SIA comparison of step D.
    end
end
