# # Homogenisation — a mean-field estimate against Bil's FE² cell
#
# `base/MechaMic` is computational homogenisation in the full sense: at every macroscopic
# integration point Bil solves a complete microscopic boundary value problem. Its deck says
# `Method = Microstructure plasticell0`, and `plasticell0` is a two-phase cell — a
# Drucker-Prager matrix around a stiffer, nearly incompressible elastic inclusion.
#
# This page compares that against something Bil has no equivalent of: the analytical
# mean-field estimate from
# [MeanFieldHomogenization.jl](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl).
#
# It is therefore a different kind of check from the rest of this directory. Everywhere
# else two codes solve the same equations and are asked to agree. Here two *methods* — one
# analytical, one full-field — are asked whether they describe the same material. A
# disagreement would be informative in both directions.
#
# ## The cell
#
# Read from the deck and from `composite0.msh`:
#
# | phase | model in Bil | ``E`` | ``\nu`` | volume fraction |
# |---|---|---:|---:|---:|
# | matrix | `Plast` (Drucker-Prager, ``c`` = 1.5 MPa, ``\varphi = \psi`` = 25°) | 2713 MPa | 0.339 | 0.8232 |
# | inclusion | `Elast` | 5000 MPa | 0.490 | **0.1768** |
#
# The fraction is not quoted anywhere: it is integrated from the mesh, four regions of
# 0.0442 against four of 0.2058, summing to 1.0000 exactly.
#
# ## Finding an elastic point
#
# The shipped dates are useless for this. At ``t = 5`` the deck imposes
# ``\sigma_{22} = -16`` MPa and the cell reports ``\varepsilon_{11}/|\varepsilon_{22}| =
# 2.6`` — an apparent Poisson ratio no elastic material can have, so the matrix has yielded
# well before. At ``t = 10``, unloaded, a residual strain remains.
#
# Under uniaxial compression the Drucker-Prager matrix yields at about 4.7 MPa, so the load
# was reduced and the response checked for linearity:
#
# | applied | ``\sigma_{22}/\varepsilon_{22}`` | ``-\varepsilon_{11}/\varepsilon_{22}`` |
# |---:|---:|---:|
# | 0.5 MPa | 3.4816e9 | 0.5697 |
# | 1 MPa | 3.4816e9 | 0.5697 |
# | 2 MPa | 3.4816e9 | 0.5697 |
# | 16 MPa | 1.3926e9 | 2.6059 |
#
# Identical over a factor of four, then not: the first three are elastic. In plane strain
# those two numbers give ``E = 3.0230`` GPa and ``\nu = 0.3630``.
#
# ## The comparison
#
# | scheme | ``E`` [GPa] | ``\nu`` |
# |---|---:|---:|
# | Reuss — lower bound | 2.9517 | 0.3548 |
# | Mori-Tanaka | 3.0160 | 0.3622 |
# | **self-consistent** | **3.0209** | **0.3629** |
# | Voigt — upper bound | 3.3184 | 0.4676 |
# | **Bil, FE²** | **3.0230** | **0.3630** |
#
# Bil's cell sits inside the bounds, and the self-consistent scheme reproduces it to
# **0.07 % on ``E`` and 0.03 % on ``\nu``**. Mori-Tanaka is 0.23 % out.
#
# That the self-consistent estimate wins is not a surprise worth much on its own — at 17.7 %
# the two schemes are close and the ordering could invert with the morphology. What the
# agreement does establish is more useful than a ranking:
#
# * the microstructure is read correctly — the volume fraction integrated from the mesh and
#   the moduli taken from the deck reproduce the cell's stiffness without a fitted parameter;
# * Bil's homogenised elastic response is right, independently confirmed;
# * and there is now a **bound to judge a full-field implementation against**. Writing an
#   FE² backend here — the plan's step 9b, which needs periodic boundary conditions and a
#   nested solve per quadrature point — would otherwise have had nothing to be checked
#   against but Bil itself.
#
# ## Step 9b: the cell solved here
#
# With the bound in hand, the full-field computation was written — `PeriodicCell` in
# `src/Backends/Homogenization.jl`, an `AbstractMaterial` whose response is a finite element
# solve on a periodic cell. Run on Bil's own `composite0.msh` with the deck's phases:
#
# | | ``E`` [GPa] | ``\nu`` | ``\sigma_{22}/\varepsilon_{22}`` | ``-\varepsilon_{11}/\varepsilon_{22}`` |
# |---|---:|---:|---:|---:|
# | this package | **3.0230** | **0.3630** | 3.4816e9 | 0.5697 |
# | Bil, FE² | **3.0230** | **0.3630** | 3.4816e9 | 0.5697 |
#
# Identical to five figures, down to the intermediate quantities. The effective stiffness
# comes out symmetric — ``C_{12} = C_{21} = 2.9369`` GPa — with ``C_{11} = C_{22}``, which
# the cell's four-fold symmetry requires and nothing in the assembly enforces.
#
# Two details of the periodic constraints cost more time than the physics. Both directions
# must go into a **single** `PeriodicDirichlet`: adding one per direction looks equivalent
# and is not, because a corner node belongs to both mappings and the resulting nested affine
# constraint is refused. And the node pinned to remove the rigid translation has to be an
# **interior** one, since every boundary node already carries a periodic constraint.
#
# ## What this does not say
#
# Nothing about plasticity. The whole comparison lives below 4.7 MPa, and the cell exists to
# be driven past that: the deck's own loading is more than three times the yield. A
# mean-field scheme has no equivalent of the return mapping happening in each element of the
# cell, and the 16 MPa point — where Bil reports an apparent Poisson ratio of 2.6 — is
# outside anything this page can reach.

using MeanFieldHomogenization
using Printf

# ## The estimate
#
# `iso_stiffness_E_nu` builds the isotropic stiffness; the package works in
# [TensND.jl](https://github.com/MicroPoroChemoMechanics/TensND.jl) tensors rather than
# Tensors.jl, which is worth knowing before passing it a `SymmetricTensor`.

C_matrix = iso_stiffness_E_nu(2713.0e6, 0.339)
C_inclusion = iso_stiffness_E_nu(5000.0e6, 0.490)

rve = RVE(:matrix)
add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_matrix))
add_phase!(rve, :inclusion, Ellipsoid(1.0), Dict(:C => C_inclusion); fraction = 0.1768)

"Isotropic `E` and `ν` of a homogenised stiffness."
function elastic_constants(C)
    K = (C[1, 1, 1, 1] + 2C[1, 1, 2, 2]) / 3
    G = C[1, 2, 1, 2]
    return 9K * G / (3K + G), (3K - 2G) / (2 * (3K + G))
end

@printf("  %-22s %12s %10s\n", "scheme", "E [GPa]", "ν")
for (name, scheme) in (
        "Reuss" => Reuss(),
        "Mori-Tanaka" => MoriTanaka(),
        "self-consistent" => SelfConsistent(),
        "Voigt" => Voigt(),
    )
    E, ν = elastic_constants(homogenize(rve, scheme, :C))
    @printf("  %-22s %12.4f %10.4f\n", name, E / 1.0e9, ν)
end
@printf("  %-22s %12.4f %10.4f\n", "Bil, FE²", 3.0230, 0.3630)
