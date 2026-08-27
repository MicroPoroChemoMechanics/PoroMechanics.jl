# # Biot Consolidation 2D — Ternay Dam
#
# Hydromechanical response of the Ternay gravity dam (concrete body on a rock foundation)
# to a sudden reservoir filling, on
# [Ferrite.jl](https://ferrite-fem.github.io/Ferrite.jl/stable/). Unlike the VoronoiFVM
# examples, this model uses the FEM interface: `element_matrices!` and `facet_load!` take
# the place of `storage!` / `flux!` / `bcondition!`.
#
# The unknowns are the solid displacement ``\mathbf{u}`` [m] and the pore pressure
# ``p_l`` [Pa].
#
# ## Governing equations
#
# Plane strain, zero gravity, saturated medium (``S_l = 1``). Mechanical equilibrium:
#
# ```math
# \nabla \cdot \boldsymbol{\sigma} = \mathbf{0}, \qquad
#   \boldsymbol{\sigma} = \lambda \operatorname{tr}(\boldsymbol{\varepsilon})\,\mathbf{I}
#   + 2\mu\,\boldsymbol{\varepsilon} - b\,p_l\,\mathbf{I}
# ```
#
# Liquid mass conservation (Biot):
#
# ```math
# \frac{\partial}{\partial t}\!\left(b\,\nabla\cdot\mathbf{u} + N p_l\right)
#   - \nabla \cdot \left(\frac{k_\text{int}}{\mu_l}\,\nabla p_l\right) = 0
# ```
#
# with ``\boldsymbol{\varepsilon} = \tfrac{1}{2}(\nabla\mathbf{u} + \nabla^\top\mathbf{u})``
# the infinitesimal strain, ``b`` the Biot coefficient and ``N`` [Pa⁻¹] the storage modulus
# at constant strain.
#
# | Region | Location | Condition |
# |---|---|---|
# | 101–105, 121 | Upstream face | ``p_l = \rho_l g (H - y)`` (hydrostatic) |
# | 106–112, 125 | Downstream face | ``p_l = 0`` (drainage) |
# | 101–105, 121 | Upstream face | traction ``\mathbf{t} = -p_\text{hydro}(y)\,\mathbf{n}`` |
# | 122, 124 | Lateral rock faces | ``u_1 = 0`` |
# | 123 | Base of foundation (``y = 455`` m) | ``u_1 = u_2 = 0`` (clamped) |
#
# The initial state is ``\mathbf{u} = \mathbf{0}``, ``p_l = 0``: a dry dam filled
# instantaneously at ``t = 0``.
#
# ## Geometry and mesh
#
# The mesh `ternay.msh` is in GMSH 2.2 format — 479 nodes, 860 triangles, physical
# surfaces `"1"` (concrete) and `"2"` (rock foundation), boundary lines `"101"`–`"125"`.
# The reservoir surface is at ``H = 517`` m NGF and the upstream base at ``y \approx 476``
# m, so the maximum hydrostatic pressure is about
# ``10^4 \times (517 - 476) = 0.41`` MPa.
#
# ## Material parameters
#
# | Symbol | Concrete | Rock | Unit | Description |
# |---|---|---|---|---|
# | ``E`` | ``1.4 \times 10^{10}`` | ``1.8 \times 10^{10}`` | Pa | Young's modulus |
# | ``\nu`` | ``0.15`` | ``0.15`` | — | Poisson's ratio |
# | ``k_\text{int}`` | ``10^{-14}`` | ``10^{-11}`` | m² | Intrinsic permeability |
# | ``\mu_l`` | ``10^{-3}`` | ``10^{-3}`` | Pa·s | Dynamic viscosity |
# | ``b`` | ``0.4`` | ``0.2`` | — | Biot coefficient |
# | ``N`` | ``10^{-10}`` | ``10^{-10}`` | Pa⁻¹ | Storage modulus |
#
# With ``c_v = (k_\text{int}/\mu_l) / (b^2/(\lambda+2\mu) + N)`` and ``T_c = L^2/c_v``, the
# rock consolidates in seconds while the concrete takes about eight days — the concrete is
# the long-time bottleneck.

using PoroMechanics
using Ferrite
using FerriteGmsh
using LinearAlgebra
using SparseArrays
using Printf

# ## Model definition

"""Parameters of model M7 (Biot poroelasticity, saturated medium)."""
Base.@kwdef struct BiotModel <: AbstractPoroModel
    ## concrete (surface "1")
    E_b     :: Float64 = 1.4e10    # Young's modulus [Pa]
    nu_b    :: Float64 = 0.15      # Poisson's ratio [-]
    k_b     :: Float64 = 1.0e-14   # intrinsic permeability [m²]
    b_b     :: Float64 = 0.4       # Biot coefficient [-]
    N_b     :: Float64 = 1.0e-10   # storage modulus [Pa⁻¹]
    ## rock (surface "2")
    E_r     :: Float64 = 1.8e10
    nu_r    :: Float64 = 0.15
    k_r     :: Float64 = 1.0e-11
    b_r     :: Float64 = 0.2
    N_r     :: Float64 = 1.0e-10
    ## fluid (common)
    mu_l    :: Float64 = 1.0e-3    # dynamic viscosity [Pa·s]
    rho_l   :: Float64 = 1000.0    # density [kg/m³]
    ## hydraulic datum: p_l(y) = ρ_l·g·(H−y)
    H       :: Float64 = 517.0     # water table elevation [m NGF]
    rho_g   :: Float64 = 10_000.0  # ρ_l·g [Pa/m]
end

PoroMechanics.nspecies(::BiotModel) = 3   # u₁, u₂, p
PoroMechanics.species_names(::BiotModel) = [:u1, :u2, :p]

"""Result of the M7 simulation: solution vector and DofHandler."""
struct BiotSolution
    x  :: Vector{Float64}
    dh :: DofHandler
end

function Base.show(io::IO, r::BiotSolution)
    print(io, "BiotSolution: $(length(r.x)) DOFs — reach them through .x and .dh")
end

"Lamé coefficients λ, μ in plane strain from (E, ν)."
function lame_coeffs(E, nu)
    λ = E * nu / ((1 + nu) * (1 - 2nu))
    μ = E / (2 * (1 + nu))
    return λ, μ
end

"Hydrostatic reservoir pressure at elevation y [Pa]."
p_hydro(m::BiotModel, y::Real) = m.rho_g * (m.H - y)

# ## Element assembly

"""
    PoroMechanics.element_matrices!(ke1, ke2, is_beton::Bool, m::BiotModel, cv_u, cv_p)

Computes the steady element matrix `ke1` and the storage element matrix `ke2`
for a P1/P1 triangular element of the Biot model.

`is_beton` selects the concrete parameters (`true`) or the rock ones (`false`).

Blocks of ke1 (terms independent of Δt):
  ke1[u,u] = K_uu   — elastic stiffness
  ke1[u,p] = −K_up  — mechanical coupling
  ke1[p,p] = K_pp   — Darcy conductivity

Blocks of ke2 (divided by Δt during time integration):
  ke2[p,u] = +K_up^T — couplage hydraulique
  ke2[p,p] = M_pp    — storage compressibility
"""
function PoroMechanics.element_matrices!(ke1, ke2, is_beton::Bool, m::BiotModel, cv_u, cv_p)
    fill!(ke1, 0.0)
    fill!(ke2, 0.0)

    E  = is_beton ? m.E_b  : m.E_r
    nu = is_beton ? m.nu_b : m.nu_r
    k  = is_beton ? m.k_b  : m.k_r
    b  = is_beton ? m.b_b  : m.b_r
    N  = is_beton ? m.N_b  : m.N_r

    λ, μ = lame_coeffs(E, nu)
    K_l  = k / m.mu_l    # hydraulic conductivity [m²/(Pa·s)]

    nu_l = getnbasefunctions(cv_u)   # 6 (P1 × 2 composantes)
    np_l = getnbasefunctions(cv_p)   # 3

    for q in 1:getnquadpoints(cv_u)
        dΩ = getdetJdV(cv_u, q)

        ## Mechanical stiffness K_uu : ∫ ε(δu) : C : ε(u) dΩ
        for i in 1:nu_l
            εᵢ = symmetric(shape_gradient(cv_u, q, i))
            for j in 1:nu_l
                εⱼ = symmetric(shape_gradient(cv_u, q, j))
                σⱼ = λ * tr(εⱼ) * one(εⱼ) + 2μ * εⱼ
                ke1[i, j] += (εᵢ ⊡ σⱼ) * dΩ
            end
        end

        ## Biot coupling
        ## ke1[u,p] = −b ∫ (∇·δu) p_j dΩ
        ## ke2[p,u] = +b ∫ (∇·u_j) δp dΩ
        for i in 1:nu_l
            div_δu = tr(shape_gradient(cv_u, q, i))
            for j in 1:np_l
                Np  = shape_value(cv_p, q, j)
                val = b * div_δu * Np * dΩ
                ke1[i,          nu_l + j] -= val   # K1[u,p]
                ke2[nu_l + j,   i       ] += val   # K2[p,u]
            end
        end

        ## Darcy K_pp and storage M_pp
        for i in 1:np_l
            ∇Npi = shape_gradient(cv_p, q, i)
            Npi  = shape_value(cv_p, q, i)
            for j in 1:np_l
                ∇Npj = shape_gradient(cv_p, q, j)
                Npj  = shape_value(cv_p, q, j)
                ke1[nu_l + i, nu_l + j] += K_l * (∇Npi ⋅ ∇Npj) * dΩ   # K1[p,p]
                ke2[nu_l + i, nu_l + j] += N   * Npi * Npj * dΩ          # K2[p,p]
            end
        end
    end
end

# ## Surface loading

"""
    PoroMechanics.facet_load!(fe, facet, m::BiotModel, fv_u)

Adds the hydrostatic thrust  t = −p_hydro(y)·n  on the upstream face.
`fe` is a local vector of size `ndofs_per_cell`.
"""
function PoroMechanics.facet_load!(fe, facet, m::BiotModel, fv_u)
    coords = getcoordinates(facet)
    nu_l   = getnbasefunctions(fv_u)
    for q in 1:getnquadpoints(fv_u)
        x  = spatial_coordinate(fv_u, q, coords)
        n  = getnormal(fv_u, q)
        dΓ = getdetJdV(fv_u, q)
        t  = -p_hydro(m, x[2]) * n   # inward traction
        for i in 1:nu_l
            Nu = shape_value(fv_u, q, i)
            fe[i] += (Nu ⋅ t) * dΓ
        end
    end
end

# ## Solving

"""
    run_biot(; dt, n_steps, mesh_path)

Simulates the consolidation of the Ternay dam by Biot poroelasticity (M7).

Returns `(x, dh)`: the solution vector at the last step and the DofHandler,
so that results can be post-processed later (extracting p or u per node).

## Keyword arguments
- `dt`        : time step [s] (default: 100 s)
- `n_steps`   : number of steps (default: 20 → t_max = 2000 s ≈ 1.8 × T_c concrete)
- `mesh_path` : path to `ternay.msh` (default: the script's own directory)
"""
function run_biot(;
    dt        = 100.0,
    n_steps   = 20,
    mesh_path = joinpath(@__DIR__, "ternay.msh"),
)
    m = BiotModel()

    ## ── Mesh ─────────────────────────────────────────────────────────────────
    grid = togrid(mesh_path)
    @printf("Mesh: %d nodes, %d elements\n", getnnodes(grid), getncells(grid))

    beton_cells = getcellset(grid, "1")   # concrete elements

    ## ── DofHandler : P1 vectoriel (u₁,u₂) + P1 scalaire (p) ─────────────────
    ip_geo = Lagrange{RefTriangle, 1}()
    ip_u   = Lagrange{RefTriangle, 1}()^2
    ip_p   = Lagrange{RefTriangle, 1}()

    dh = DofHandler(grid)
    add!(dh, :u, ip_u)
    add!(dh, :p, ip_p)
    close!(dh)

    n_loc = ndofs_per_cell(dh)
    n_tot = ndofs(dh)
    @printf("DOFs: %d total (%d per element)\n", n_tot, n_loc)

    ## ── Quadrature ───────────────────────────────────────────────────────────
    qr     = QuadratureRule{RefTriangle}(3)
    qr_fac = FacetQuadratureRule{RefTriangle}(2)

    cv_u = CellValues(qr, ip_u, ip_geo)
    cv_p = CellValues(qr, ip_p, ip_geo)
    fv_u = FacetValues(qr_fac, ip_u, ip_geo)

    ## ── Dirichlet conditions ──────────────────────────────────────────────────
    ch = ConstraintHandler(dh)

    upstream_tags   = ["101","102","103","104","105","121"]
    downstream_tags = ["106","107","108","109","110","111","112","125"]

    upstream_hyd   = reduce(union, getfacetset(grid, r) for r in upstream_tags)
    downstream_hyd = reduce(union, getfacetset(grid, r) for r in downstream_tags)

    add!(ch, Dirichlet(:p, upstream_hyd,   (x, t) -> p_hydro(m, x[2])))
    add!(ch, Dirichlet(:p, downstream_hyd, (x, t) -> 0.0))

    for reg in ["122","123","124"]
        add!(ch, Dirichlet(:u, getfacetset(grid, reg), (x, t) -> 0.0, [1]))
    end
    add!(ch, Dirichlet(:u, getfacetset(grid, "123"), (x, t) -> 0.0, [2]))

    close!(ch)
    update!(ch, 0.0)
    @printf("Contraintes Dirichlet : %d DDL prescrits\n", length(ch.prescribed_dofs))

    ## ── Global assembly of K1 and K2 ─────────────────────────────────────────
    K1 = allocate_matrix(dh)
    K2 = allocate_matrix(dh)
    as1 = start_assemble(K1)
    as2 = start_assemble(K2)

    ke1_buf = zeros(n_loc, n_loc)
    ke2_buf = zeros(n_loc, n_loc)

    for cell in CellIterator(dh)
        reinit!(cv_u, cell)
        reinit!(cv_p, cell)
        is_beton = cellid(cell) ∈ beton_cells
        PoroMechanics.element_matrices!(ke1_buf, ke2_buf, is_beton, m, cv_u, cv_p)
        assemble!(as1, celldofs(cell), ke1_buf)
        assemble!(as2, celldofs(cell), ke2_buf)
    end
    println("K1 and K2 assembled.")

    ## ── Surface loading (upstream hydrostatic thrust) ─────────────────────────
    f_ext    = zeros(n_tot)
    u_range  = dof_range(dh, :u)
    nu_facet = getnbasefunctions(fv_u)   # = 6 for a P1 triangle (2 components × 3 nodes)
    fe_u     = zeros(nu_facet)

    upstream_mec = reduce(union, getfacetset(grid, r) for r in upstream_tags)

    for facet in FacetIterator(dh, upstream_mec)
        reinit!(fv_u, facet)
        fill!(fe_u, 0.0)
        PoroMechanics.facet_load!(fe_u, facet, m, fv_u)
        dofs = celldofs(facet)
        for (i, d) in enumerate(u_range)
            f_ext[dofs[d]] += fe_u[i]
        end
    end
    println("Surface loading assembled.")

    ## ── Condition initiale ────────────────────────────────────────────────────
    x_vec = zeros(n_tot)
    apply!(x_vec, ch)   # Dirichlet values consistent with t=0

    ## ── Time loop — implicit Euler ────────────────────────────────────────────
    println("\nM7 Biot 2D — Ternay dam  (Δt = $(dt) s, $(n_steps) steps)")
    println("─"^66)
    println("Step |     t [d] | p_max concrete [MPa] | u₁_max [mm] | u₂_max [mm]")
    println("─"^66)

    x = copy(x_vec)
    p_range = dof_range(dh, :p)

    for step in 1:n_steps
        t_step = step * dt
        x_prev = copy(x)

        A   = K1 + (1.0/dt) .* K2
        rhs = copy(f_ext)
        mul!(rhs, K2, x_prev, 1.0/dt, 1.0)

        update!(ch, t_step)
        apply!(A, rhs, ch)

        x = A \ rhs

        ## — diagnostics —
        p_beton_max = -Inf
        for ci in beton_cells
            d = celldofs(dh, ci)
            for k in p_range
                p_beton_max = max(p_beton_max, x[d[k]])
            end
        end

        u1_max = 0.0;  u2_max = 0.0
        for ci in 1:getncells(grid)
            d = celldofs(dh, ci)
            for k in 1:2:length(u_range)
                u1_max = max(u1_max, abs(x[d[u_range[k]]]))
            end
            for k in 2:2:length(u_range)
                u2_max = max(u2_max, abs(x[d[u_range[k]]]))
            end
        end

        @printf("%4d | %9.4f | %+17.4f | %+11.4f | %+11.4f\n",
                step, t_step/86400.0, p_beton_max/1e6, u1_max*1e3, u2_max*1e3)
    end

    println("─"^66)
    println("Simulation finished.")

    return BiotSolution(x, dh)
end

# ## Results

result = run_biot()

# ## Key points
#
# - **Two-matrix split** — the linear Biot system separates into a stationary part ``K_1``
#   and a storage part ``K_2``, both assembled once. Only the right-hand side changes from
#   step to step, so ``A = K_1 + K_2/\Delta t`` can be factored once when ``\Delta t`` is
#   constant.
# - **P1/P1 mixed elements** — equal-order interpolation for ``\mathbf{u}`` and ``p_l`` is
#   not inf-sup stable in general, but works here because the Biot storage term ``N > 0``
#   regularises the pressure block, leaving no spurious pressure modes.
# - **`is_beton::Bool` dispatch** — the material is read from the Ferrite `cellset` at
#   assembly time and passed as a `Bool`, avoiding a runtime dictionary lookup.
# - **Verification** — as ``t \to \infty`` the pore pressure satisfies ``\Delta p_l = 0``
#   with the upstream and downstream Dirichlet values, and the displacement field is the
#   fully drained elastic response of the dam.
