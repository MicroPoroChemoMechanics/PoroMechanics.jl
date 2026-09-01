# # Poroelastic coefficients from a microstructure
#
# Where the homogenisation stops and this package starts.
#
# [MeanFieldHomogenization.jl](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl)
# owns the estimate: given a microstructure and a scheme it returns a drained stiffness
# ``\mathbb{C}^{\rm hom}``, and from it the Biot tensor ``\boldsymbol{B}`` and modulus
# ``M``. This package owns the field problem: [`BiotPoroelastic`](@ref) consumes a Biot
# coefficient `b` and a storage modulus `N`, and knows nothing about where they came from.
# The division is deliberate — a microstructure is not a boundary value problem, and the
# two questions have different right answers about what to be general in.
#
# This page checks that the two halves meet. Nothing here couples the packages at run time;
# the point is that the numbers agree on the one case where both have a closed form, so
# that handing one to the other is a defensible thing to do.
#
# !!! note "What this is not"
#     Not a validation of the homogenisation, which is that package's business and is
#     validated there. This checks the **interface**: the same coefficient, computed twice
#     from different directions, with the conventions each package writes it in.

using MeanFieldHomogenization
using PoroMechanics
using Printf

# ## A porous solid
#
# Spherical pores in a uniform solid, Mori-Tanaka. Deliberately the simplest microstructure
# that has a textbook answer: the pore space is isotropic, so ``\boldsymbol{B} = b\,
# \boldsymbol{1}`` and the classical relations hold.

E_s, ν_s, φ = 60.0e9, 0.25, 0.20        # solid grain, and the porosity
C_s = iso_stiffness_E_nu(E_s, ν_s)
k_s = E_s / (3 * (1 - 2ν_s))            # bulk modulus of the solid phase

rve = RVE(:solid)
add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_s))
## A void is an inclusion with no stiffness. Exactly zero would be singular for the scheme,
## so it is given 1 Pa against the solid's 60 GPa — eleven orders down, and the result is
## unchanged at the printed precision by making it smaller.
add_phase!(
    rve, :pore, Ellipsoid(1.0), Dict(:C => iso_stiffness_E_nu(1.0, 0.0)); fraction = φ
)

C_hom = homogenize(rve, MoriTanaka())
par = poroelastic_parameters(C_hom, C_s, φ)

# The drained constants the field problem needs, read off the homogenised stiffness:

k_hom = C_hom[1, 1, 1, 1] - 4 / 3 * C_hom[1, 2, 1, 2]
G_hom = C_hom[1, 2, 1, 2]
E_hom = 9k_hom * G_hom / (3k_hom + G_hom)
ν_hom = (3k_hom - 2G_hom) / (2 * (3k_hom + G_hom))

@printf("drained:  E = %.4f GPa   ν = %.6f   K = %.4f GPa\n", E_hom / 1.0e9, ν_hom, k_hom / 1.0e9)

# ## The coefficient, computed twice
#
# MeanFieldHomogenization forms ``\boldsymbol{B}`` and ``1/M`` from the full tensors,
#
# ```math
# \boldsymbol{B} = \boldsymbol{1} : \left(\mathbb{I} - \mathbb{S}_{\rm s} : \mathbb{C}^{\rm hom}\right),
# \qquad
# \frac{1}{M} = \boldsymbol{1} : \mathbb{S}_{\rm s} : \left(\boldsymbol{B} - \varphi\,\boldsymbol{1}\right),
# ```
#
# which for an isotropic pore space collapses to the two scalars every poromechanics text
# writes down. Computing those independently is the check:

b_closed = 1 - k_hom / k_s
N_closed = (b_closed - φ) / k_s

@printf("  MeanFieldHomogenization:  b = %.12f   1/M = %.6e Pa⁻¹\n", par.B[1, 1], par.inverse_modulus)
@printf("  closed form:              b = %.12f   1/M = %.6e Pa⁻¹\n", b_closed, N_closed)
@printf("  off-diagonal of B:        %.3e  (isotropic pore space ⇒ zero)\n", par.B[1, 2])

# Identical. The interface is sound in the direction that matters: `par.B[1,1]` is this
# package's `b`, and `par.inverse_modulus` is its `N`, with no conversion in between.

# ## What the field problem then does with them
#
# The microstructure now *sets* the consolidation, rather than a deck doing it:

m = BiotPoroelastic(;
    E = E_hom, nu = ν_hom, k = 1.0e-16, mu_l = 1.0e-3,
    b = par.B[1, 1], N = par.inverse_modulus
)

@printf("c_v = %.4e m²/s    Skempton = %.6f    ν_u = %.6f\n",
    consolidation_coefficient(m), skempton(m), undrained_poisson(m))

# The Skempton coefficient comes out **above one**, which is not a bug in either package
# and is worth pausing on. MeanFieldHomogenization's own documentation warns of it: its
# ``1/M`` assumes an incompressible fluid, and with compressible grains the pore volume is
# held fixed while the grains themselves compress, so an isotropic total stress can raise
# the pore pressure by more than itself. The formula here — `skempton(m) = b / (N·K + b²)`
# — was written from Biot's relations with no knowledge of that discussion, and reproduces
# it. Two independent routes to the same consequence is the strongest corroboration this
# page has to offer.
#
# It also means the pairing carries an assumption that has to travel with it: a compressible
# fluid adds ``\varphi/k_f`` to the storage, and neither `N` above nor `BiotPoroelastic`
# will add it for you.

# ## Where the bridge stops
#
# ``\boldsymbol{B}`` is a tensor, and it is anisotropic whenever the pore space is — which
# is the normal case, not the exotic one. Penny-shaped cracks normal to ``e_3``:

rve_c = RVE(:solid)
add_matrix!(rve_c, Ellipsoid(1.0), Dict(:C => C_s))
add_phase!(rve_c, :crack, PennyCrack(1.0), Dict(:C => C_s); density = 0.08)
B_c = biot_tensor(homogenize(rve_c, MoriTanaka()), C_s)

@printf("cracked:  B₁₁ = %.6f   B₃₃ = %.6f   ratio = %.4f\n",
    B_c[1, 1], B_c[3, 3], B_c[3, 3] / B_c[1, 1])

# A factor of three between the crack normal and the crack plane. `BiotPoroelastic` holds a
# **scalar** `b` and cannot carry that: the bridge built above closes only for an isotropic
# pore space, and a cracked medium needs a poroelastic material with a tensorial Biot
# coefficient before any of this transfers. That material does not exist here yet, and
# saying which microstructures the current one may legitimately be fed is the useful half of
# this page.
