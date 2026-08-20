# Example M7 — Biot poroelasticity 2D (Ternay dam)
#
# Biot poroelasticity 2D via Ferrite.jl.
# Validation case: Ternay gravity dam under hydromechanical loading.
# Test case: Ternay dam (triangular P1/P1 mesh).
#
# PDEs (plane strain, no gravity, saturated medium S_l = 1):
#   Mechanical equilibrium : ∇·(C:ε(u) − b·p·I) = 0
#   Mass conservation      : ∂(b·∇·u + N·p)/∂t  − ∇·(k_int/μ·∇p) = 0
#
# Discretisation: triangular P1/P1 elements, implicit Euler.
# Assembly: two matrices K1 (stiffness + Darcy) and K2 (coupling + storage)
# such that  A·xⁿ⁺¹ = f_ext + (K2/Δt)·xⁿ  with A = K1 + K2/Δt.
#
# Materials: concrete (physical surface "1") + rock (physical surface "2").
# Mesh     : examples/M7_Biot/ternay.msh (GMSH 2.2, 479 nodes).
#
# Dependencies (added to the root Project.toml):
#   Ferrite, FerriteGmsh, LinearAlgebra, SparseArrays, Printf

using PoroMechanics
using Ferrite
using FerriteGmsh
using LinearAlgebra
using SparseArrays
using Printf

# ── Material parameters ───────────────────────────────────────────────────────

"""Parameters of model M7 (Biot poroelasticity, saturated medium)."""
Base.@kwdef struct M7Model <: AbstractPoroModel
    # concrete (surface "1")
    E_b     :: Float64 = 1.4e10    # Young's modulus [Pa]
    nu_b    :: Float64 = 0.15      # Poisson's ratio [-]
    k_b     :: Float64 = 1.0e-14   # intrinsic permeability [m²]
    b_b     :: Float64 = 0.4       # Biot coefficient [-]
    N_b     :: Float64 = 1.0e-10   # storage modulus [Pa⁻¹]
    # rock (surface "2")
    E_r     :: Float64 = 1.8e10
    nu_r    :: Float64 = 0.15
    k_r     :: Float64 = 1.0e-11
    b_r     :: Float64 = 0.2
    N_r     :: Float64 = 1.0e-10
    # fluid (common)
    mu_l    :: Float64 = 1.0e-3    # dynamic viscosity [Pa·s]
    rho_l   :: Float64 = 1000.0    # density [kg/m³]
    # hydraulic datum: p_l(y) = ρ_l·g·(H−y)
    H       :: Float64 = 517.0     # water table elevation [m NGF]
    rho_g   :: Float64 = 10_000.0  # ρ_l·g [Pa/m]
end

PoroMechanics.nspecies(::M7Model) = 3   # u₁, u₂, p
PoroMechanics.species_names(::M7Model) = [:u1, :u2, :p]

"""Result of the M7 simulation: solution vector and DofHandler."""
struct M7Result
    x  :: Vector{Float64}
    dh :: DofHandler
end

function Base.show(io::IO, r::M7Result)
    print(io, "M7Result: $(length(r.x)) DOFs — reach them through .x and .dh")
end

"Lamé coefficients λ, μ in plane strain from (E, ν)."
function lame_coeffs(E, nu)
    λ = E * nu / ((1 + nu) * (1 - 2nu))
    μ = E / (2 * (1 + nu))
    return λ, μ
end

"Hydrostatic reservoir pressure at elevation y [Pa]."
p_hydro(m::M7Model, y::Real) = m.rho_g * (m.H - y)

# ── Element assembly ──────────────────────────────────────────────────────────

"""
    PoroMechanics.element_matrices!(ke1, ke2, is_beton::Bool, m::M7Model, cv_u, cv_p)

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
function PoroMechanics.element_matrices!(ke1, ke2, is_beton::Bool, m::M7Model, cv_u, cv_p)
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

        # Mechanical stiffness K_uu : ∫ ε(δu) : C : ε(u) dΩ
        for i in 1:nu_l
            εᵢ = symmetric(shape_gradient(cv_u, q, i))
            for j in 1:nu_l
                εⱼ = symmetric(shape_gradient(cv_u, q, j))
                σⱼ = λ * tr(εⱼ) * one(εⱼ) + 2μ * εⱼ
                ke1[i, j] += (εᵢ ⊡ σⱼ) * dΩ
            end
        end

        # Biot coupling
        # ke1[u,p] = −b ∫ (∇·δu) p_j dΩ
        # ke2[p,u] = +b ∫ (∇·u_j) δp dΩ
        for i in 1:nu_l
            div_δu = tr(shape_gradient(cv_u, q, i))
            for j in 1:np_l
                Np  = shape_value(cv_p, q, j)
                val = b * div_δu * Np * dΩ
                ke1[i,          nu_l + j] -= val   # K1[u,p]
                ke2[nu_l + j,   i       ] += val   # K2[p,u]
            end
        end

        # Darcy K_pp and storage M_pp
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

# ── Surface loading ───────────────────────────────────────────────────────────

"""
    PoroMechanics.facet_load!(fe, facet, m::M7Model, fv_u)

Adds the hydrostatic thrust  t = −p_hydro(y)·n  on the upstream face.
`fe` is a local vector of size `ndofs_per_cell`.
"""
function PoroMechanics.facet_load!(fe, facet, m::M7Model, fv_u)
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

# ── Solve ─────────────────────────────────────────────────────────────────────

"""
    run_M7(; dt, n_steps, mesh_path)

Simulates the consolidation of the Ternay dam by Biot poroelasticity (M7).

Returns `(x, dh)`: the solution vector at the last step and the DofHandler,
so that results can be post-processed later (extracting p or u per node).

# Keyword arguments
- `dt`        : time step [s] (default: 100 s)
- `n_steps`   : number of steps (default: 20 → t_max = 2000 s ≈ 1.8 × T_c concrete)
- `mesh_path` : path to `ternay.msh` (default: the script's own directory)
"""
function run_M7(;
    dt        = 100.0,
    n_steps   = 20,
    mesh_path = joinpath(@__DIR__, "ternay.msh"),
)
    m = M7Model()

    # ── Mesh ─────────────────────────────────────────────────────────────────
    grid = togrid(mesh_path)
    @printf("Mesh: %d nodes, %d elements\n", getnnodes(grid), getncells(grid))

    beton_cells = getcellset(grid, "1")   # concrete elements

    # ── DofHandler : P1 vectoriel (u₁,u₂) + P1 scalaire (p) ─────────────────
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

    # ── Quadrature ───────────────────────────────────────────────────────────
    qr     = QuadratureRule{RefTriangle}(3)
    qr_fac = FacetQuadratureRule{RefTriangle}(2)

    cv_u = CellValues(qr, ip_u, ip_geo)
    cv_p = CellValues(qr, ip_p, ip_geo)
    fv_u = FacetValues(qr_fac, ip_u, ip_geo)

    # ── Dirichlet conditions ──────────────────────────────────────────────────
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

    # ── Global assembly of K1 and K2 ─────────────────────────────────────────
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

    # ── Surface loading (upstream hydrostatic thrust) ─────────────────────────
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

    # ── Condition initiale ────────────────────────────────────────────────────
    x_vec = zeros(n_tot)
    apply!(x_vec, ch)   # Dirichlet values consistent with t=0

    # ── Time loop — implicit Euler ────────────────────────────────────────────
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

        # — diagnostics —
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

    return M7Result(x, dh)
end

# ── Entry point ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    run_M7()
end
