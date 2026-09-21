"""
    DryingModel(; materials, parameters=DryingParameters(),
                dirichlet=((), (), ()), heat_flux=())

Non-isothermal water and air transport in a rigid porous medium. The unknowns are
liquid pressure `p_l` [Pa], dry-air pressure `p_a` [Pa], and temperature `T` [K].
Storage contains total water mass, total dry-air mass (including dissolved air),
and volumetric entropy. Fluxes combine Darcy flow, vapor diffusion, dissolved-air
advection, and heat conduction. Gravity and mechanical deformation are omitted.
The entropy balance retains the reference model's approximation without a separate
volumetric entropy-production term.

`materials[region]` supplies a [`DryingMaterial`](@ref) for each cell region; a tuple,
vector or indexable mapping can be used. Geometry, initial conditions and time
controls are supplied separately to the grid and solver.

`dirichlet` contains three boundary tuples, one per unknown, with entries
`(boundary_region, value)`; a value is a number or `t -> value`.
`heat_flux` contains `(boundary_region, Q)` entries with incoming heat flux `Q`
in W/m², also constant or time-dependent. Heat is imposed as entropy flux `Q/T`.
The denominator is floored at 200 K during trial evaluations, as in the original
model; this is a numerical guard, not a physical extension to low temperatures.
Unspecified boundaries have zero flux. Use disjoint temperature-Dirichlet and
heat-flux boundaries. The time integrator must stop at discontinuities of `Q`.
"""
Base.@kwdef struct DryingModel{M, P, D, H} <: AbstractPoroModel
    materials::M
    parameters::P = DryingParameters()
    dirichlet::D = ((), (), ())
    heat_flux::H = ()
end

nspecies(::DryingModel) = 3
species_names(::DryingModel) = [:p_l, :p_a, :T]

"""
    drying_material(model, region)

Material on the specified cell region. At an interface, query each side separately:
VoronoiFVM assembles its storage contributions with their respective region labels.
"""
drying_material(m::DryingModel, region) = m.materials[region]

"""
    vapor_pressure(model::DryingModel, p_l, T)

Evaluate the Kelvin relation using the model's shared [`DryingParameters`](@ref).
"""
vapor_pressure(m::DryingModel, pl, T) = vapor_pressure(m.parameters, pl, T)

"""Stored water mass, dry-air mass and entropy, per unit volume of medium."""
function storage!(f, u, node, m::DryingModel, ::Any)
    p = m.parameters
    mat = drying_material(m, node.region)
    φ = mat.phi
    Cs = mat.C_s

    pl = u[1];  pa = u[2];  T = u[3]
    θ = T - p.T_0
    pv = vapor_pressure(p, pl, T)
    pg = pv + pa
    pc = pg - pl
    at = 1.0 - p.alpha_T * θ
    pc0 = pc / at
    sl = saturation(mat, pc0);  sg = 1.0 - sl

    ρv = pv * p.M_vsR / T
    ρa = pa * p.M_asR / T
    Ml = p.rho_l * φ * sl
    Mv = ρv * φ * sg
    Ma = ρa * φ * sg
    Mad = Ml * _drying_dissolved_air(p, pa)

    s_l = p.C_pl * log(T / p.T_0)
    s_v = p.C_pv * log(T / p.T_0) - log(pv / p.p_v0) / p.M_vsR + p.L_0 / p.T_0
    s_a = p.C_pa * log(T / p.T_0) - log(pa / p.p_a0) / p.M_asR
    dU = _drying_capillary_integral(p, mat, pc, θ)

    ## Dissolved air is given the entropy of gaseous air: the heat of dissolution is neglected.
    f[1] = Ml + Mv
    f[2] = Ma + Mad
    f[3] = Cs * log(T / p.T_0) - φ * dU + Ml * s_l + Mv * s_v + (Ma + Mad) * s_a
    return nothing
end

function flux!(f, u, edge, m::DryingModel, ::Any)
    p = m.parameters
    mat = drying_material(m, edge.region)
    φ = mat.phi
    ki = mat.k_int
    λs = mat.lam_s

    pl1, pl2 = u[1, 1], u[1, 2]
    pa1, pa2 = u[2, 1], u[2, 2]
    T1, T2 = u[3, 1], u[3, 2]
    plm = (pl1 + pl2) / 2.0
    pam = (pa1 + pa2) / 2.0
    Tm = (T1 + T2) / 2.0

    θm = Tm - p.T_0
    at = 1.0 - p.alpha_T * θm
    pvm = vapor_pressure(p, plm, Tm)
    pgm = pvm + pam
    pcm = pgm - plm
    pc0m = pcm / at
    slm = saturation(mat, pc0m);  sgm = 1.0 - slm

    ρvm = pvm * p.M_vsR / Tm
    ρam = pam * p.M_asR / Tm
    ρgm = ρvm + ρam
    cvm = ρvm / ρgm;  cam = 1.0 - cvm

    ## Darcy conductivities
    Kl = p.rho_l * ki / p.mu_l * relative_permeability(mat, pc0m)
    krg = gas_relative_permeability(slm)
    KDv = ρvm * ki / p.mu_g * krg
    KDa = ρam * ki / p.mu_g * krg

    ## Fick diffusion — Millington-Quirk tortuosity
    τ = φ^(1.0 / 3.0) * max(sgm, 0.0)^(7.0 / 3.0)
    Dav = p.D_av0 * (p.p_v0 + p.p_a0) / pgm * (Tm / p.T_0)^1.88
    Def = φ * sgm * τ * Dav
    KFv = ρgm * Def;  KFa = KFv
    bar = Def * cvm * cam / Tm
    KDv += bar * (p.M_asR - p.M_vsR)
    KDa += bar * (p.M_vsR - p.M_asR)

    ## Thermal conductivity — Johansen geometric mean
    KTH = λs^(1.0 - φ) * p.lam_l^(φ * slm) * p.lam_g^(φ * sgm)

    ## Pressures and mass fractions at the nodes
    pv1 = vapor_pressure(p, pl1, T1);  pg1 = pv1 + pa1
    pv2 = vapor_pressure(p, pl2, T2);  pg2 = pv2 + pa2
    ρg1 = pv1 * p.M_vsR / T1 + pa1 * p.M_asR / T1
    ρg2 = pv2 * p.M_vsR / T2 + pa2 * p.M_asR / T2
    cv1 = pv1 * p.M_vsR / T1 / ρg1;  ca1 = 1.0 - cv1
    cv2 = pv2 * p.M_vsR / T2 / ρg2;  ca2 = 1.0 - cv2

    ## Elementary fluxes
    Wl = Kl * (pl1 - pl2)
    Wv = KDv * (pg1 - pg2) + KFv * (cv1 - cv2)
    Wa = KDa * (pg1 - pg2) + KFa * (ca1 - ca2)

    ## Dissolved air travels with the liquid, taken from the upstream node
    Wad = (Wl >= 0 ? _drying_dissolved_air(p, pa1) : _drying_dissolved_air(p, pa2)) * Wl

    ## Entropy flux
    s_lm = p.C_pl * log(Tm / p.T_0)
    s_vm = p.C_pv * log(Tm / p.T_0) - log(pvm / p.p_v0) / p.M_vsR + p.L_0 / p.T_0
    s_am = p.C_pa * log(Tm / p.T_0) - log(pam / p.p_a0) / p.M_asR
    Js = KTH / Tm * (T1 - T2) + s_lm * Wl + s_vm * Wv + s_am * (Wa + Wad)

    f[1] = Wl + Wv
    f[2] = Wa + Wad
    f[3] = Js
    return nothing
end

function bcondition!(f, u, bnode, m::DryingModel, ::Any)
    apply_dirichlet!(f, u, bnode, m.dirichlet[1]; species = 1)
    apply_dirichlet!(f, u, bnode, m.dirichlet[2]; species = 2)
    apply_dirichlet!(f, u, bnode, m.dirichlet[3]; species = 3)
    for (region, heat) in m.heat_flux
        if bnode.region == region
            Q = dirichlet_value(heat, bnode.time)
            VoronoiFVM.boundary_neumann!(
                f, u, bnode;
                species = 3, region, value = Q / max(u[3], 200.0)
            )
        end
    end
    return nothing
end

"""
    liquid_saturation(model::DryingModel, u, region)
    liquid_saturation(model::DryingModel, profiles, node, region)

Liquid saturation from a local `(p_l, p_a, T)` state, or column `node` of a profile
matrix. Applies the Kelvin relation and thermal shift before evaluating the
region's retention curve. Pressures can be continuous across a material interface
while saturation differs between its two sides.
"""
function liquid_saturation(m::DryingModel, u, region)
    pl, pa, T = u
    p = m.parameters
    pc0 = (vapor_pressure(p, pl, T) + pa - pl) / (1 - p.alpha_T * (T - p.T_0))
    return saturation(drying_material(m, region), pc0)
end
liquid_saturation(m::DryingModel, u, i, region) = liquid_saturation(m, @view(u[:, i]), region)
