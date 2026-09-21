"""
    DryingMaterial(; phi, k_int, lam_s, C_s, retention, rel_perm)

Properties of one rigid porous material: porosity [-], intrinsic permeability [m²],
solid thermal conductivity [W/(m·K)], volumetric solid heat capacity [J/(m³·K)],
retention and liquid relative-permeability laws. Numerical coefficients are promoted
so that material parameters can carry `ForwardDiff.Dual` values.
"""
Base.@kwdef struct DryingMaterial{T, R, K}
    phi::T             # porosity [-]
    k_int::T           # intrinsic permeability [m²]
    lam_s::T           # solid thermal conductivity [W/(m·K)]
    C_s::T             # volumetric heat capacity [J/(m³·K)]
    retention::R       # S_l(p_c), regularized near saturation
    rel_perm::K        # k_rl(p_c)
end

## Promote rather than require a single type: differentiating with respect to one
## coefficient makes that field a `Dual` while the others stay `Float64`.
function DryingMaterial(phi, k_int, lam_s, C_s, retention, rel_perm)
    return DryingMaterial(promote(phi, k_int, lam_s, C_s)..., retention, rel_perm)
end

saturation(mat::DryingMaterial, pc) = saturation(mat.retention, pc)
dsaturation_dpc(mat::DryingMaterial, pc) = dsaturation_dpc(mat.retention, pc)
relative_permeability(mat::DryingMaterial, pc) = relative_permeability(mat.rel_perm, pc)

"""
    DryingParameters(; kwargs...)

Shared water, vapor, dry-air and thermal-retention coefficients for
[`DryingModel`](@ref). Defaults describe the existing non-isothermal drying model.
All pressures are in Pa, temperature in K, densities in kg/m³, viscosities in Pa·s,
conductivities in W/(m·K), heat capacities in J/(kg·K), and latent heat in J/kg.
`M_vsR` and `M_asR` are molar mass divided by the gas constant [kg·K/J].
`D_av0` is the reference air-vapor diffusivity [m²/s], `H_a` the Henry constant [Pa],
and `alpha_T` the thermal-retention coefficient [1/K].

Use positive dry-air pressure and temperature. Thermal retention requires
`1 - alpha_T * (T - T_0) > 0`. These are constitutive-domain restrictions,
not a guarantee that Newton trial states will remain admissible.
"""
Base.@kwdef struct DryingParameters{T}
    rho_l::T = 1000.0       # liquid density [kg/m³]
    mu_l::T = 1.0e-3       # liquid viscosity [Pa·s]
    mu_g::T = 1.8e-5       # gas viscosity [Pa·s]
    M_vsR::T = 0.00216      # water molar mass / R [kg·K/J]
    M_asR::T = 0.00346      # air molar mass / R [kg·K/J]
    p_l0::T = 1.0e5        # reference liquid pressure [Pa]
    p_v0::T = 2460.0       # saturated vapor pressure at T₀ [Pa]
    p_a0::T = 97540.0      # reference air pressure [Pa]
    T_0::T = 293.0        # reference temperature [K]
    D_av0::T = 0.00248      # air-vapor diffusion at T₀ [m²/s]
    lam_l::T = 0.6          # liquid thermal conductivity [W/(m·K)]
    lam_g::T = 0.026        # gas thermal conductivity [W/(m·K)]
    C_pl::T = 4180.0       # liquid specific heat [J/(kg·K)]
    C_pv::T = 1800.0       # vapor specific heat [J/(kg·K)]
    C_pa::T = 1000.0       # dry air specific heat [J/(kg·K)]
    L_0::T = 2.45e6       # latent heat of vaporization [J/kg]
    alpha_T::T = 0.003        # thermal variation coeff. of S_l [1/K]
    H_a::T = 1.0e10       # Henry constant of air in water [Pa] (Olivella et al., 1994)
end

function DryingParameters(rho_l, mu_l, mu_g, M_vsR, M_asR, p_l0, p_v0, p_a0, T_0, D_av0, lam_l, lam_g, C_pl, C_pv, C_pa, L_0, alpha_T, H_a)
    values = promote(rho_l, mu_l, mu_g, M_vsR, M_asR, p_l0, p_v0, p_a0, T_0, D_av0, lam_l, lam_g, C_pl, C_pv, C_pa, L_0, alpha_T, H_a)
    return DryingParameters{typeof(first(values))}(values...)
end

"""
    vapor_pressure(parameters::DryingParameters, p_l, T)

Vapor pressure [Pa] from the modified Kelvin relation, including the temperature
variation of latent heat. At `(p_l0, T_0)` it returns `p_v0`.
"""
function vapor_pressure(p::DryingParameters, pl::Real, T::Real)
    θ = T - p.T_0
    return p.p_v0 * exp(
        p.M_vsR / T * (
            (pl - p.p_l0) / p.rho_l
                + p.L_0 * θ / p.T_0
                + (p.C_pl - p.C_pv) * (θ - T * log(T / p.T_0))
        )
    )
end

_drying_dissolved_air(p::DryingParameters, pa) = pa / p.H_a * p.M_asR / p.M_vsR

const _drying_gauss_a = (0.9324695142, 0.66120938646, 0.23861918608)
const _drying_gauss_w = (0.17132449237, 0.36076157304, 0.46791393457)

function _drying_saturation_dT(p::DryingParameters, mat::DryingMaterial, pc::Real, θ::Real)
    at = 1.0 - p.alpha_T * θ
    at <= 0 && return zero(at + pc)
    pc0 = pc / at
    return dsaturation_dpc(mat, pc0) * pc0 * p.alpha_T / at
end

function _drying_capillary_integral(p::DryingParameters, mat::DryingMaterial, pc::Real, θ::Real)
    ## The regularized retention curve also varies for pc < 0; integrate with the
    ## signed interval there as well, consistently with storage and its Jacobian.
    h = pc / 2.0
    dU = zero(pc + θ)
    for j in 1:3
        dU += _drying_gauss_w[j] * (
            _drying_saturation_dT(p, mat, h * (1.0 + _drying_gauss_a[j]), θ) +
                _drying_saturation_dT(p, mat, h * (1.0 - _drying_gauss_a[j]), θ)
        )
    end
    return dU * h
end
