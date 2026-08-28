# Shared by the Gardner benchmarks: the exponential-law column, its VoronoiFVM callbacks,
# and the closed-form steady profile. The steady and transient cases differ only in what
# they impose at the top and how they march, not in the physics.

include("laplace.jl")

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using LinearAlgebra
using Printf

# ## Model

Base.@kwdef struct GardnerColumn{R, K} <: AbstractPoroModel
    k_int::Float64 = 1.0e-12     # intrinsic permeability [m²]
    mu_l::Float64 = 1.0e-3       # dynamic viscosity [Pa·s]
    rho_l::Float64 = 1.0e3       # liquid density [kg/m³]
    gravite::Float64 = -9.81     # gravity, signed, z upward [m/s²]
    phi::Float64 = 0.35          # porosity [-]
    p_g::Float64 = 0.0           # gas pressure [Pa]
    alpha::Float64 = 5.0e-4      # Gardner exponent [Pa⁻¹]
    q_top::Float64 = -2.0e-7     # imposed flux at the top, upward positive [m/s]
    L::Float64 = 2.0             # column height above the water table [m]

    retention::R = Gardner(5.0e-4)
    rel_perm::K = GardnerKrl(5.0e-4)
end

PoroMechanics.nspecies(::GardnerColumn) = 1
PoroMechanics.species_names(::GardnerColumn) = [:p_l]

"""Saturated hydraulic conductivity in pressure form, ``K_s = k_\\text{int}/\\mu_l`` [m²/(Pa·s)]."""
saturated_conductivity(m::GardnerColumn) = m.k_int / m.mu_l

Sl(m::GardnerColumn, pc) = saturation(m.retention, pc)
krl(m::GardnerColumn, pc) = relative_permeability(m.rel_perm, pc)
Kl(m::GardnerColumn, pc) = saturated_conductivity(m) * krl(m, pc)

# ## Reference solution

"""
    gardner_profile(m, z) -> p_l [Pa]

Steady liquid pressure at height `z` above the water table. Returns `NaN` where the
bracket turns negative, i.e. where the imposed upward flux cannot be sustained.
"""
function gardner_profile(m::GardnerColumn, z)
    β = m.alpha * m.rho_l * abs(m.gravite)
    Q = m.q_top / (saturated_conductivity(m) * m.rho_l * abs(m.gravite))
    arg = (1 + Q) * exp(-β * z) - Q
    return arg <= 0 ? NaN : log(arg) / m.alpha
end

# ## Constitutive behaviour
#
# The gravity term follows the sign convention of the Richards example: `gravite` is signed
# (negative for `z` upward), and the flux returned to VoronoiFVM is divided by the edge
# length internally.

function PoroMechanics.flux!(f, u, edge, m::GardnerColumn, ::Any)
    pl1, pl2 = u[1, 1], u[1, 2]
    pc_avg = m.p_g - (pl1 + pl2) / 2
    Kl_avg = Kl(m, pc_avg)
    dx = edge.coord[1, 2] - edge.coord[1, 1]
    f[1] = Kl_avg * (pl1 - pl2) + Kl_avg * m.rho_l * m.gravite * dx
end

"""Storage: volumetric water content ``\\phi S_l(p_c)``."""
function PoroMechanics.storage!(f, u, ::Any, m::GardnerColumn, ::Any)
    f[1] = m.phi * Sl(m, m.p_g - u[1])
end

"""
Boundary conditions:
  Region 1 (z = 0) : Dirichlet p_l = 0 — the water table
  Region 2 (z = L) : Neumann, the imposed infiltration flux
"""
function PoroMechanics.bcondition!(f, u, bnode, m::GardnerColumn, ::Any)
    boundary_dirichlet!(f, u, bnode; species = 1, region = 1, value = m.p_g)
    ## VoronoiFVM's Neumann `value` is an *inflow*, while `q_top` is signed upward.
    ## Downward infiltration (q_top < 0) is therefore an inflow of −q_top.
    boundary_neumann!(f, u, bnode; species = 1, region = 2, value = -m.q_top)
end

