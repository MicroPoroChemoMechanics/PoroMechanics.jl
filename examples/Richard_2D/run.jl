# examples/Richard_2D/run.jl
# Richards equation 2D — drainage of a composite column
# Richards equation 2D on VoronoiFVM.jl.
#
# Physical reference: Richards, L.A. (1931). Capillary conduction of liquids
#   through porous mediums. Physics, 1(5), 318–333.
# Validation case   : drainage of a composite column

# ──────────────────────────────────────────────────────────────────────────────
# 1. Paquets
# ──────────────────────────────────────────────────────────────────────────────
using VoronoiFVM
using ExtendableGrids
using SimplexGridFactory   # building the composite 2D mesh
using Triangulate           # triangulation backend (Shewchuk)
using Plots
using Printf
using DelimitedFiles
using Statistics

# ──────────────────────────────────────────────────────────────────────────────
# 2. Physical and geometric parameters
# ──────────────────────────────────────────────────────────────────────────────
const gravite = -9.81      # gravity [m/s²]
const phi     = 0.38       # porosity [-] (same in both zones)
const rho_l   = 1000.0     # liquid density [kg/m³]
const mu_l    = 1.0e-3     # dynamic viscosity [Pa·s]
const pg_ref  = 0.0        # reference gas pressure [Pa]

# Intrinsic permeabilities per region (outer zone and inclusion)
const k_int_outer = 8.9e-12    # m² — outer zone (region 1)
const k_int_inner = 8.9e-13    # m² — inclusion  (region 2)
const k_int_par_region = [k_int_outer, k_int_inner]

# Geometry (matching columncomposite.geo)
const W  = 0.02     # total width [m]
const Hg = 0.20     # total height [m]
const w  = W  / 2   # inclusion width [m]  = 0.01
const h  = Hg / 2   # inclusion height [m] = 0.10
const x0 = (W  - w) / 2   # left x of the inclusion = 0.005 m
const y0 = (Hg - h) / 2   # bottom y of the inclusion = 0.050 m

const U_P_L = 1   # index of the unknown in VoronoiFVM

@printf("Outer zone (region 1) : k_int = %.2e m²\n", k_int_outer)
@printf("Inclusion  (region 2) : k_int = %.2e m²\n", k_int_inner)
@printf("Permeability ratio    : k_out/k_inc = %.0f×\n", k_int_outer/k_int_inner)
@printf("Domaine       : W × H  = %.3f × %.3f m\n", W, Hg)
@printf("Inclusion     : w × h  = %.3f × %.3f m, bottom-left corner (%.3f, %.3f)\n",
        w, h, x0, y0)
@printf("hydrostatic pₗ at the base (y=0) = ρₗ |g| H = %.0f Pa\n",
        rho_l * abs(gravite) * Hg)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Retention curves — tabulated data file
# ──────────────────────────────────────────────────────────────────────────────
# Three columns: p_c [Pa], S_l [-], k_rl [-]. The table is not shipped with the
# repository; point RICHARDS_2D_DATA at your own measured or fitted curve.
billes_path = get(ENV, "RICHARDS_2D_DATA", joinpath(@__DIR__, "retention_table.dat"))
isfile(billes_path) || error("""
    Retention curve table not found: $billes_path
    Provide a 3-column table (p_c [Pa], S_l [-], k_rl [-]) at that path, or set
    the RICHARDS_2D_DATA environment variable to point at one.""")
billes_raw  = readdlm(billes_path)

# const is MANDATORY: without it Julia cannot infer the type of these arrays
# from flux!/storage!, which causes an allocation on every edge and a wrong
# ForwardDiff Jacobian → Newton diverges.
const pc_tab  = billes_raw[:, 1]
const sl_tab  = billes_raw[:, 2]
const krl_tab = billes_raw[:, 3]

@printf("Retention table: %d points, p_c ∈ [%.0f, %.0f] Pa\n",
        length(pc_tab), minimum(pc_tab), maximum(pc_tab))

"""
Bounded linear interpolation, ForwardDiff-compatible (Dual numbers).

The `zero(x - xv[1])` term preserves the exact type of x: it returns 0.0 for
Float64 and Dual(0, 0...) for a Dual, avoiding any type instability.
"""
function interp1(xv::Vector{Float64}, yv::Vector{Float64}, x)
    if x ≤ xv[1]
        return yv[1] + zero(x - xv[1])    # type-stable: Float64 or Dual
    elseif x ≥ xv[end]
        return yv[end] + zero(x - xv[1])
    end
    i = searchsortedfirst(xv, x) - 1
    t = (x - xv[i]) / (xv[i+1] - xv[i])
    return yv[i] + t * (yv[i+1] - yv[i])
end

"""
S_l(p_c) — returns 1 for p_c ≤ 0.
`1.0 + zero(pc)`: for pc::Float64 → 1.0; for pc::Dual → Dual(1, 0...).
Guarantees that the return type always matches the input type.
"""
function Sl(pc)
    pc ≤ 0.0 && return 1.0 + zero(pc)
    return interp1(pc_tab, sl_tab, pc)
end

"""k_rl(p_c) — idem, type-stable."""
function krl(pc)
    pc ≤ 0.0 && return 1.0 + zero(pc)
    return interp1(pc_tab, krl_tab, pc)
end

@printf("Sl(0)    = %.6f  (expected 1)\n",    Sl(0.0))
@printf("Sl(577)  = %.6f  (air entry)\n", Sl(577.0))
@printf("Sl(1000) = %.6f  (residual)\n",  Sl(1000.0))

# ──────────────────────────────────────────────────────────────────────────────
# 4. Callbacks VoronoiFVM
# ──────────────────────────────────────────────────────────────────────────────
const pl_base_ini = rho_l * abs(gravite) * Hg   # 1962 Pa

"""Pressure imposed at the base as a function of time (reference-case ramp)."""
pl_base(t::Float64) = t ≤ 360.0 ? pl_base_ini * max(0.0, 1.0 - t/360.0) : 0.0

"""
Numerical flux on an edge (2D, y axis vertical).
  f[1] = k_l · (H_{l,1} − H_{l,2}),   H_l = p_l − ρ_l g y
Effective permeability: harmonic mean of the regions of the two adjacent cells.
"""
function flux!(f, u, edge, _data)
    pl1 = u[U_P_L, 1]
    pl2 = u[U_P_L, 2]
    y1  = edge.coord[2, 1]   # y coordinate of node 1 (vertical axis)
    y2  = edge.coord[2, 2]   # y coordinate of node 2

    pc_avg = pg_ref - 0.5 * (pl1 + pl2)

    # Intrinsic permeability per region — harmonic mean at the interface
    k1    = k_int_par_region[edge.cellregions[1]]
    k2    = k_int_par_region[edge.cellregions[2]]
    k_eff = 2 * k1 * k2 / (k1 + k2)   # = k1 within a region, harmonic mean otherwise

    k_l = rho_l * k_eff / mu_l * krl(pc_avg)

    hl1 = pl1 - rho_l * gravite * y1
    hl2 = pl2 - rho_l * gravite * y2

    f[U_P_L] = k_l * (hl1 - hl2)
end

"""Stockage : m_l = ρ_l φ S_l(p_c),  p_c = −p_l  (p_g = 0)."""
function storage!(f, u, _node, _data)
    f[U_P_L] = rho_l * phi * Sl(pg_ref - u[U_P_L])
end

"""
Boundary conditions:
  - Region 11 (base, y=0) : Dirichlet p_l = pl_base(t)
  - All other regions     : zero flux (homogeneous Neumann, the default)
"""
function bcondition!(f, u, bnode, ::Any)
    boundary_dirichlet!(f, u, bnode;
        species = U_P_L,
        region  = 11,
        value   = pl_base(bnode.time))
end

println("Callbacks defined.")
@printf("  pl_base(0)   = %.1f Pa\n", pl_base(0.0))
@printf("  pl_base(360) = %.1f Pa\n", pl_base(360.0))
@printf("  pl_base(361) = %.1f Pa\n", pl_base(361.0))

# ──────────────────────────────────────────────────────────────────────────────
# 5. Composite 2D mesh (matching columncomposite.geo)
# ──────────────────────────────────────────────────────────────────────────────
b = SimplexGridBuilder(; Generator=Triangulate)

# Corners of the outer domain
p1 = point!(b, 0,    0   )
p2 = point!(b, W,    0   )
p3 = point!(b, W,    Hg  )
p4 = point!(b, 0,    Hg  )

# Corners of the centred inclusion
p5 = point!(b, x0,       y0      )
p6 = point!(b, x0 + w,   y0      )
p7 = point!(b, x0 + w,   y0 + h  )
p8 = point!(b, x0,       y0 + h  )

# Outer boundaries
# Note: VoronoiFVM requires boundary regions ≥ 1.
# We use 11 for the base (Dirichlet) and 1 for every other face
# (zero flux by default — no BC is applied on region 1).
facetregion!(b, 11); facet!(b, p1, p2)   # base  → Dirichlet (region 11)
facetregion!(b,  1); facet!(b, p2, p3)   # right → zero flux
facetregion!(b,  1); facet!(b, p3, p4)   # top   → zero flux
facetregion!(b,  1); facet!(b, p4, p1)   # left  → zero flux

# Inclusion interface (region 1 — internal boundary, no BC)
facetregion!(b, 1)
facet!(b, p5, p6)
facet!(b, p6, p7)
facet!(b, p7, p8)
facet!(b, p8, p5)

# Interior points used to identify the two regions
cellregion!(b, 1); regionpoint!(b, x0/2,         Hg/2)   # outer zone (left)
cellregion!(b, 2); regionpoint!(b, x0 + w/2,     Hg/2)   # inclusion (centre)

# Maximum triangle volume (controls mesh refinement)
grid2d = simplexgrid(b; maxvolume=5e-7)

nn = num_nodes(grid2d)
nc = num_cells(grid2d)
coords  = grid2d[Coordinates]    # 2×nn : coords[1,:]=x, coords[2,:]=y
cregions = grid2d[CellRegions]
cnodes   = grid2d[CellNodes]

@printf("2D mesh generated: %d nodes, %d triangles\n", nn, nc)
println("BFaceRegion 11 → base (y=0) : Dirichlet pₗ = pl_base(t)")
println("Region 1 → outer zone (k_int = 8.9e-12 m²)")
println("Region 2 → inclusion  (k_int = 8.9e-13 m²)")

# ──────────────────────────────────────────────────────────────────────────────
# 6. VoronoiFVM system and time integration
# ──────────────────────────────────────────────────────────────────────────────
sys = VoronoiFVM.System(
    grid2d;
    flux       = flux!,
    storage    = storage!,
    bcondition = bcondition!,
    species    = [U_P_L],
)

# Condition initiale : distribution hydrostatique p_l = ρ_l |g| (H - y)
inival = unknowns(sys)
for i in 1:nn
    inival[U_P_L, i] = rho_l * abs(gravite) * (Hg - coords[2, i])
end

@printf("System: %d nodes, 1 unknown (pₗ)\n", nn)
@printf("Condition initiale : pₗ ∈ [%.1f, %.1f] Pa → Sₗ = 1 partout\n",
        minimum(inival), maximum(inival))

# Output times (0, 200, 400, … 3000 s)
tsave = collect(0.0:200.0:3000.0)

control = VoronoiFVM.SolverControl(;
    Δt      = 1.0,
    Δt_max  = 1000.0,
    Δt_min  = 1.0e-6,
    # The column starts fully saturated, where dS_l/dp_l = 0: the storage term
    # drops out and the first step is elliptic, so its Δu does not shrink with
    # Δt. Δu_opt has to clear that one-off jump (1962 Pa) or the controller
    # halves Δt down to Δt_min and gives up.
    Δu_opt  = 2.5e3,
    reltol  = 1.0e-6,
    abstol  = 1.0e-10,
    verbose = false,
)

println("Solving...")
@time tsol = solve(sys; inival, times=tsave, control)
println("Done. Time steps taken: $(length(tsol.t) - 1)")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Post-traitement
# ──────────────────────────────────────────────────────────────────────────────

"""Saturation degree at every node for output index it."""
function saturation_field(it)
    [Sl(pg_ref - tsol[U_P_L, i, it]) for i in 1:nn]
end

"""2D scatter coloured by saturation, in mm."""
function carte_saturation(it; titre="", clims=(0.0, 1.0))
    sl = saturation_field(it)
    scatter(coords[1,:] .* 1e3, coords[2,:] .* 1e3;
        marker_z          = sl,
        clims             = clims,
        c                 = cgrad(:Blues; rev = true),
        colorbar          = true,
        colorbar_title    = "Sₗ",
        markersize        = 1.2,
        markerstrokewidth = 0,
        xlabel            = "x [mm]",
        ylabel            = "y [mm]",
        title             = titre,
        aspect_ratio      = :equal,
        legend            = false)
end

# 7.1 2D saturation maps at 4 output times
t_maps  = [0.0, 600.0, 1800.0, 3000.0]
it_maps = [argmin(abs.(tsol.t .- t)) for t in t_maps]
sl_all  = vcat([saturation_field(it) for it in it_maps]...)
clims_glob = (minimum(sl_all), 1.0)

cartes = [carte_saturation(it_maps[k];
            titre  = "Sₗ — t = $(Int(t_maps[k])) s",
            clims  = clims_glob)
          for k in 1:4]
p_cartes = plot(cartes...; layout=(1,4), size=(1100, 550))
display(p_cartes)

# 7.2 Vertical profiles — outer zone vs inclusion
"""Indices of the nodes closest to x_target, sorted by y."""
function profil_vertical(x_cible; n_top=15)
    dists_x = abs.(coords[1, :] .- x_cible)
    idx = partialsortperm(dists_x, 1:min(n_top, nn))
    return idx[sortperm(coords[2, idx])]
end

idx_ext = profil_vertical(x0/2;       n_top=20)
idx_inc = profil_vertical(x0 + w/2;   n_top=20)

t_profils = [0.0, 600.0, 1800.0, 3000.0]
colors_t  = [:steelblue, :darkorange, :red, :crimson]
labels_t  = ["t = $(Int(t)) s" for t in t_profils]

p_ext = plot(; xlabel="Sₗ [-]", ylabel="y [mm]",
               title="Vertical profile — outer zone (x≈2.5 mm)",
               legend=:bottomleft, size=(380, 500))
p_inc = plot(; xlabel="Sₗ [-]", ylabel="y [mm]",
               title="Vertical profile — inclusion (x≈10 mm)",
               legend=:bottomleft, size=(380, 500))

for (k, t_req) in enumerate(t_profils)
    it = argmin(abs.(tsol.t .- t_req))

    y_e  = coords[2, idx_ext] .* 1e3
    sl_e = [Sl(pg_ref - tsol[U_P_L, i, it]) for i in idx_ext]
    plot!(p_ext, sl_e, y_e; lw=2, color=colors_t[k], label=labels_t[k])

    y_i  = coords[2, idx_inc] .* 1e3
    sl_i = [Sl(pg_ref - tsol[U_P_L, i, it]) for i in idx_inc]
    plot!(p_inc, sl_i, y_i; lw=2, color=colors_t[k], label=labels_t[k])
end

hspan!(p_inc, [y0*1e3, (y0+h)*1e3]; alpha=0.08, color=:darkorange, label="")
annotate!(p_inc, [(0.92, (y0 + h/2)*1e3, text("inclusion", 7, :darkorange, :center))])
display(plot(p_ext, p_inc; layout=(1,2), size=(760, 500)))

# 7.3 Time evolution — outer zone vs inclusion
"""Node closest to the point (xt, yt)."""
function nœud_proche(xt, yt)
    d = @. sqrt((coords[1,:] - xt)^2 + (coords[2,:] - yt)^2)
    argmin(d)
end

i_ext = nœud_proche(x0/2,       Hg/2)
i_inc = nœud_proche(x0 + w/2,   Hg/2)

@printf("Outer zone node: (%.4f, %.4f) m\n", coords[1, i_ext], coords[2, i_ext])
@printf("Inclusion node : (%.4f, %.4f) m\n", coords[1, i_inc], coords[2, i_inc])

t_all  = tsol.t
sl_ext = [Sl(pg_ref - tsol[U_P_L, i_ext, it]) for it in eachindex(t_all)]
sl_inc = [Sl(pg_ref - tsol[U_P_L, i_inc, it]) for it in eachindex(t_all)]
pc_ext = [pg_ref - tsol[U_P_L, i_ext, it]     for it in eachindex(t_all)]
pc_inc = [pg_ref - tsol[U_P_L, i_inc, it]     for it in eachindex(t_all)]

p_sl_ev = plot(; xlabel="t [s]", ylabel="Sₗ [-]",
    title="Saturation — outer zone vs inclusion", legend=:bottomleft, size=(650, 320))
plot!(p_sl_ev, t_all, sl_ext; lw=2, color=:steelblue,  label="Outer zone (x≈2.5 mm, y=H/2)")
plot!(p_sl_ev, t_all, sl_inc; lw=2, color=:darkorange, label="Inclusion    (x=W/2,  y=H/2)")
vline!(p_sl_ev, [360.0]; lw=1, ls=:dash, color=:grey, label="t=360 s")
hline!(p_sl_ev, [Sl(1000.0)]; lw=0.8, ls=:dot, color=:grey, label="residual Sₗ")

p_pc_ev = plot(; xlabel="t [s]", ylabel="pᶜ = −pₗ [Pa]",
    title="Capillary pressure — outer zone vs inclusion", legend=:topleft, size=(650, 320))
plot!(p_pc_ev, t_all, pc_ext; lw=2, color=:steelblue,  label="Outer zone")
plot!(p_pc_ev, t_all, pc_inc; lw=2, color=:darkorange, label="Inclusion")
hline!(p_pc_ev, [577.0]; lw=0.8, ls=:dash, color=:red, label="air-entry pᶜ ≈ 577 Pa")
vline!(p_pc_ev, [360.0]; lw=1, ls=:dash, color=:grey, label="t=360 s")
display(plot(p_sl_ev, p_pc_ev; layout=(2,1), size=(700, 620)))

# 7.4 Horizontal profile at mid-height
"""Indices of the nodes closest to y_target, sorted by x."""
function profil_horizontal(y_cible; n_top=20)
    dists_y = abs.(coords[2, :] .- y_cible)
    idx = partialsortperm(dists_y, 1:min(n_top, nn))
    return idx[sortperm(coords[1, idx])]
end

idx_hori = profil_horizontal(Hg/2; n_top=30)

p_hori = plot(; xlabel="x [mm]", ylabel="Sₗ [-]",
    title="Horizontal profile at y = H/2 — effect of the inclusion",
    legend=:bottomright, size=(700, 320))

for (k, t_req) in enumerate([600.0, 1800.0, 3000.0])
    it   = argmin(abs.(tsol.t .- t_req))
    x_h  = coords[1, idx_hori] .* 1e3
    sl_h = [Sl(pg_ref - tsol[U_P_L, i, it]) for i in idx_hori]
    plot!(p_hori, x_h, sl_h; lw=2, color=colors_t[k+1], label="t = $(Int(t_req)) s")
end

vspan!(p_hori, [x0*1e3, (x0+w)*1e3]; alpha=0.12, color=:darkorange, label="")
annotate!(p_hori, [((x0 + w/2)*1e3, 0.97, text("inclusion", 8, :darkorange, :center))])
display(p_hori)

# ──────────────────────────────────────────────────────────────────────────────
# 8. Physical checks
# ──────────────────────────────────────────────────────────────────────────────
println("═"^65)
println("Physical checks")
println("═"^65)

# 1. Dirichlet condition at the base (t = 360 s → p_l ≈ 0)
it360    = argmin(abs.(tsol.t .- 360.0))
idx_base = findall(i -> coords[2, i] < 1e-6, 1:nn)
pl_base_obs = tsol[U_P_L, idx_base, it360]
@printf("\n1. Dirichlet BC at the base (t=360 s):\n")
@printf("   pₗ_max = %.4f Pa  (expected ≈ 0)\n", maximum(abs.(pl_base_obs)))
maximum(abs.(pl_base_obs)) < 1.0 && println("   ✓ Dirichlet condition satisfied")

# 2. Drainage delay
it3000     = length(tsol.t)
sl_ext_fin = Sl(pg_ref - tsol[U_P_L, i_ext, it3000])
sl_inc_fin = Sl(pg_ref - tsol[U_P_L, i_inc, it3000])
@printf("\n2. Saturation at t=3000 s (y=H/2):\n")
@printf("   Outer zone : Sₗ = %.6f\n", sl_ext_fin)
@printf("   Inclusion  : Sₗ = %.6f\n", sl_inc_fin)
sl_inc_fin > sl_ext_fin &&
    println("   ✓ Inclusion more saturated than the outer zone (drainage delay)")

# 3. Approximate mass balance
area_total = W * Hg
vol_nœud   = area_total / nn
m_ini = sum(rho_l * phi * Sl(pg_ref - tsol[U_P_L, i, 1])      * vol_nœud for i in 1:nn)
m_fin = sum(rho_l * phi * Sl(pg_ref - tsol[U_P_L, i, it3000]) * vol_nœud for i in 1:nn)
@printf("\n3. Integrated liquid mass (approximation):\n")
@printf("   t = 0 s     : m = %.4f kg/m\n", m_ini)
@printf("   t = 3000 s  : m = %.4f kg/m\n", m_fin)
@printf("   Drainage total ≈ %.2f%%\n", (m_ini - m_fin)/m_ini * 100)

# 4. Time at which desaturation starts (Sl < 0.99)
function temps_desaturation(i_nœud, seuil=0.99)
    for it in eachindex(tsol.t)
        Sl(pg_ref - tsol[U_P_L, i_nœud, it]) < seuil && return tsol.t[it]
    end
    return Inf
end

t_desat_ext = temps_desaturation(i_ext)
t_desat_inc = temps_desaturation(i_inc)
@printf("\n4. Time at which desaturation starts (Sₗ < 0.99):\n")
@printf("   Outer zone : t = %.1f s\n", t_desat_ext)
@printf("   Inclusion  : t = %.1f s\n", t_desat_inc)
t_desat_inc > t_desat_ext &&
    @printf("   ✓ Inclusion delay: Δt = %.1f s\n",
            t_desat_inc - t_desat_ext)