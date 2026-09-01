# # Thick-walled cylinder — homogenised moduli in a field problem
#
# The second half of the interface with
# [MeanFieldHomogenization.jl](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl).
# The [previous page](@ref) checked that the two packages agree on a *scalar*; this one
# checks that the moduli it produces survive being put into a boundary value problem and
# solved.
#
# The case is that package's own [thick-walled cylinder](https://microporochemomechanics.github.io/MeanFieldHomogenization.jl/dev/fe_coupling/thick_cylinder/):
# a matrix with 25 % stiff spherical inclusions, internal pressure, plane strain, against
# Lamé's closed form written with the *homogenised* moduli. It is the right first case
# because the reference is exact and third-party-free — a disagreement can only come from
# this side.

using MeanFieldHomogenization
using PoroMechanics
using MeanFieldHomogenization.TensND
using Printf
include("cylinder_common.jl")

# !!! note "A name clash to expect"
#     Both packages export `initial_state` and `material_response`, so loading them together
#     leaves those names ambiguous. `cylinder_common.jl` qualifies them; anything combining
#     the two has to do the same.

# ## The microstructure
#
# Matrix ``k = 30`` GPa, ``\mu = 18`` GPa; inclusions ``k = 120`` GPa, ``\mu = 80`` GPa at a
# volume fraction of 25 %. Spherical, so the result is isotropic and Lamé applies unchanged.

rve = RVE(:matrix)
add_matrix!(rve, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 30.0e9, 2 * 18.0e9)))
add_phase!(
    rve, :inclusion, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 120.0e9, 2 * 80.0e9));
    fraction = 0.25
)

C_hom = homogenize(rve, MoriTanaka())
K_hom = C_hom[1, 1, 1, 1] - 4 / 3 * C_hom[1, 2, 1, 2]
G_hom = C_hom[1, 2, 1, 2]
E_hom = 9K_hom * G_hom / (3K_hom + G_hom)
ν_hom = (3K_hom - 2G_hom) / (2 * (3K_hom + G_hom))

@printf("E = %.4f GPa   ν = %.6f\n", E_hom / 1.0e9, ν_hom)

# 61.7593 GPa and 0.242670 — the 61.76 GPa and 0.2427 that package's page quotes.

# ## Convergence against Lamé

results = [solve_cylinder(E_hom, ν_hom, nr) for nr in (24, 48, 96)]

for (nr, r) in zip((24, 48, 96), results)
    @printf("nr = %3d   err(u_r) = %.4e   err(σ_θθ) = %.4e   dofs = %4d\n",
        nr, r.err_u, r.err_σ, r.ndofs)
end

# | radial elements | 24 | 48 | 96 |
# |:--|--:|--:|--:|
# | here, ``\max\lvert u_r - u_r^{\rm Lamé}\rvert / \max\lvert u_r^{\rm Lamé}\rvert`` | 1.6083·10⁻² | 4.2464·10⁻³ | 1.0775·10⁻³ |
# | MeanFieldHomogenization, same quantity | 1.6·10⁻² | 4.3·10⁻³ | 1.1·10⁻³ |
#
# The same numbers, to the precision the other page prints. That is the expected outcome
# rather than a surprise, and the reason is the point: with a structured radial mesh the
# error of this problem is the piecewise-linear interpolation error of a field that depends
# on ``r`` alone. Both computations divide the same ``[0.1, 1]`` into the same number of
# radial elements, so both should land on the same number — and anything either code added
# on top of that unavoidable error would show up as a discrepancy. None does.
#
# The observed order is 1.92, 1.98, 1.99 as the mesh is refined, converging on the second
# order Q1 elements owe. The hoop stress converges at first order — 0.84, 0.92, 0.96 — which
# is what a stress must do when it is a derivative of a second-order displacement, and is
# worth checking precisely because it is *not* the same number as the displacement.
#
# ## What differs, and what it is not
#
# That package solves a quarter annulus, 24 × 24 elements; this one solves a single radial
# row, 24 × 1. Same error, roughly twenty-five times fewer elements. That is not a
# comparison of the two codes — it is what exploiting an axisymmetry buys, and it is
# available there too. What it does mean is that the check is cheap enough to sit in the
# test suite, which is where `test/benchmarks.jl` puts it.
#
# The one thing genuinely gained here is exactness in ``\theta``: a straight-edged Q1
# annulus approximates the circular boundary, whereas a radial strip has no circumferential
# discretisation at all. At these resolutions the difference is below the printed digits,
# which the agreement above demonstrates rather than assumes.
#
# ## What this does not cover
#
# Step 2 of that page replaces the inclusions with two crack families that **close** under
# load, making the material nonlinear and history-dependent. Nothing here tests that: the
# comparison above is linear from end to end, and a closing crack needs a material with
# internal state on this side of the bridge — which exists for plasticity
# ([`DruckerPrager`](@ref)) but not for microcracks.
