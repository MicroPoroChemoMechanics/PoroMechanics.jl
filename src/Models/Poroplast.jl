"""
Poroplasticity in one dimension with axial symmetry: a Biot medium with a plastic skeleton,
posed on a radius.

This is the borehole problem — a cavity in a stressed, saturated ground, whose support and
internal pressure are released — and it is the shape `base/Poroplast` takes. Two unknowns
live on the radius, the radial displacement ``u(r)`` and the pore pressure ``p_l(r)``, and
two balances couple them:

```math
\\frac{1}{r}\\frac{\\partial}{\\partial r}\\big(r\\,\\sigma_{rr}\\big)
  - \\frac{\\sigma_{\\theta\\theta}}{r} = 0, \\qquad
\\frac{\\partial m_l}{\\partial t} + \\frac{1}{r}\\frac{\\partial}{\\partial r}(r\\,w_l) = 0
```

with ``w_l = -k_h\\,\\partial p_l/\\partial r`` and ``m_l = \\rho_l\\,\\phi``.

The kinematics are the whole reason this needs its own assembly rather than the plane one in
`src/Backends/FEM.jl`: with only ``u_r`` unknown, the strain is

```math
\\varepsilon_{rr} = \\frac{\\partial u}{\\partial r}, \\qquad
\\varepsilon_{\\theta\\theta} = \\frac{u}{r}, \\qquad
\\varepsilon_{zz} = 0
```

so the hoop strain is a *value* of the shape function divided by ``r``, not a derivative of
it. A 1D assembly that forgets the ``u/r`` term produces a perfectly plausible field that is
wrong everywhere except on a straight cavity wall.
"""

"""
    PoroplastModel(; material, phi0, rho_l0, k_l, p_l0)

| field | meaning |
|---|---|
| `material` | a [`BiotPlastic`](@ref): the skeleton, `b`, `beta`, `N`, `k`, `mu_l` |
| `phi0` | porosity at the reference state [-] |
| `rho_l0` | fluid density at `p_l0` [kg/m³] |
| `k_l` | fluid bulk modulus [Pa] |
| `p_l0` | reference pore pressure [Pa] |

The fluid is compressible, ``\\rho_l = \\rho_{l0}\\,(1 + (p_l - p_{l0})/k_l)``, which is
what makes the storage term more than the porosity change.
"""
Base.@kwdef struct PoroplastModel{M, T} <: AbstractPoroModel
    material::M
    phi0::T = 0.15
    rho_l0::T = 1.0e3
    k_l::T = 2.0e9
    p_l0::T = 4.7e6
end

nspecies(::PoroplastModel) = 2
species_names(::PoroplastModel) = [:u, :p_l]

"""
    fluid_density(m::PoroplastModel, p) -> ρ_l

``\\rho_{l0}\\,(1 + (p - p_{l0})/k_l)``.
"""
fluid_density(m::PoroplastModel, p) = m.rho_l0 * (1 + (p - m.p_l0) / m.k_l)

"""
    mobility(m::PoroplastModel, p) -> k_h

``\\rho_l k_{int}/\\mu_l``, the coefficient in front of ``\\partial p/\\partial r`` in the
mass flux.
"""
mobility(m::PoroplastModel, p) = fluid_density(m, p) * m.material.k / m.material.mu_l

"""
    liquid_mass(m::PoroplastModel, ε, εp, p) -> m_l

``\\rho_l\\,\\phi``, with the porosity from [`porosity`](@ref) — so an elastic and a plastic
volume change enter it with their own coefficients.
"""
function liquid_mass(m::PoroplastModel, ε, εp, p)
    return fluid_density(m, p) * porosity(m.material, m.phi0, ε, εp, p, m.p_l0)
end

"""
    PoroplastState(solid, p)

What one quadrature point carries between steps: the skeleton's own state — stress, strain,
plastic strain, prestress — and the pore pressure it was last converged at.

The pressure is here rather than derived because two terms need the *previous* one: the
storage term needs the mass it implies, and the mobility is evaluated there rather than at
the current iterate. That second choice is Bil's, not a simplification —
`k_h = val_n.Permeability_liquid` in `Poroplast.cpp`, exactly as in its `Richards`.
"""
struct PoroplastState{S, T}
    solid::S
    p::T
end

"""
    axisymmetric_strain_1d(du_dr, u, r) -> SymmetricTensor{2,3}

``\\mathrm{diag}(\\partial u/\\partial r,\\; u/r,\\; 0)`` — the strain of a radial
displacement field under plane-strain conditions along the axis.
"""
function axisymmetric_strain_1d(du_dr, u, r)
    z = zero(du_dr)
    return Tensors.SymmetricTensor{2, 3}((i, j) -> i != j ? z : (i == 1 ? du_dr : (i == 2 ? u / r : z)))
end

"""
    poroplast_element_residual(m, dofs, dofs_n, r1, r2, states_n, Δt) -> (residual, states)

Residual of one two-node element, and the quadrature states it leaves behind.

`dofs` is `[u₁, p₁, u₂, p₂]`. Two-point Gauss quadrature, with the axisymmetric weight `r`
folded into each contribution.

The returned residual is the **internal** one; boundary tractions are added by the caller,
which is where they belong — a traction is a property of the problem, not of an element.
"""
function poroplast_element_residual(m::PoroplastModel, dofs, dofs_n, r1, r2, states_n, Δt)
    T = eltype(dofs)
    J = (r2 - r1) / 2
    ξs = (-1 / sqrt(3), 1 / sqrt(3))

    res = zeros(T, 4)
    states = Vector{Any}(undef, 2)

    for (q, ξ) in pairs(ξs)
        N1, N2 = (1 - ξ) / 2, (1 + ξ) / 2
        dN1, dN2 = -1 / (2 * J), 1 / (2 * J)
        r = N1 * r1 + N2 * r2

        u = N1 * dofs[1] + N2 * dofs[3]
        du_dr = dN1 * dofs[1] + dN2 * dofs[3]
        p = N1 * dofs[2] + N2 * dofs[4]
        dp_dr = dN1 * dofs[2] + dN2 * dofs[4]

        state_n = states_n[q]
        ε = axisymmetric_strain_1d(du_dr, u, r)

        ## The skeleton is driven by the effective stress σ' = σ + β p I; `poro_response`
        ## returns the total stress, so the yield check inside has already seen the right
        ## one.
        σ, _, _, solid_new = poro_response(m.material, ε, p, state_n.solid, Δt)

        ## Storage: the mass now against the mass the point converged at.
        m_now = liquid_mass(m, ε, solid_new.εp, p)
        m_old = liquid_mass(m, state_n.solid.ε, state_n.solid.εp, state_n.p)

        ## Mobility at the previous pressure — Bil's lag, kept so that a comparison measures
        ## the physics and not the difference between two linearisations.
        k_h = mobility(m, state_n.p)

        w = J * r                              # Gauss weight 1, times the axisymmetric r
        for (i, (Ni, dNi)) in enumerate(((N1, dN1), (N2, dN2)))
            res[2i - 1] += w * (dNi * σ[1, 1] + Ni * σ[2, 2] / r)
            res[2i] += w * (Ni * (m_now - m_old) / Δt + dNi * k_h * dp_dr)
        end

        states[q] = PoroplastState(solid_new, p)
    end
    return res, states
end

"""
    poroplast_initial_states(m, nodes, σ0_total) -> Vector{Vector{PoroplastState}}

Two quadrature states per element, from the **total** initial stress a deck quotes.

The conversion lives here because it is the step that is easy to get wrong and impossible to
notice: the skeleton's prestress is the *effective* stress

```math
\\boldsymbol{\\sigma}'_0 = \\boldsymbol{\\sigma}_0 + \\beta\\, p_{l0}\\, \\mathbf{I}
```

and `base/Poroplast` quotes sigma_0 = -11.5 MPa with p_0 = 4.7 MPa, so the skeleton starts
at -7.74 MPa, not -11.5. Handing the total stress straight to the skeleton leaves the
initial state out of equilibrium by beta*p_0 — 3.76 MPa here — and the first step then
produces a perfectly smooth, entirely spurious wave of displacement and pressure.
"""
function poroplast_initial_states(m::PoroplastModel, nodes, σ0_total)
    σ0_eff = σ0_total + m.material.beta * m.p_l0 * one(σ0_total)
    return [
        [
            PoroplastState(initial_state(m.material, σ0_eff), m.p_l0)
                for _ in 1:2
        ] for _ in 1:(length(nodes) - 1)
    ]
end

"""
    poroplast_step!(m, nodes, states, dofs, Δt; σ_inner, σ_outer, p_inner, p_outer) -> dofs

One implicit step of the coupled problem, by Newton.

The element Jacobian is `ForwardDiff.jacobian` of
[`poroplast_element_residual`](@ref) — four degrees of freedom per element, so
differentiating it costs almost nothing and it cannot disagree with the residual it
linearizes. The quadrature states are frozen at their converged values during the
differentiation, which is what makes the tangent consistent rather than continuum.

The global matrix is dense. With two unknowns on a hundred elements that is a 202×202
factorization, far cheaper than the sparse bookkeeping it would replace, and this model is
one-dimensional by construction.

`σ_inner` and `σ_outer` are the **radial stresses** applied at the two ends, compression
negative. They enter as boundary work `-σ r δu` at the inner face, whose outward normal
points inward, and `+σ r δu` at the outer one.

Convergence is measured on the **Newton increment, scaled per field**, which is Bil's own
criterion — its `Objective Variations` block names exactly these two scales, `u_1 = 1e-3`
and `p_l = 1e5` for this deck.

Neither obvious alternative works, and both were tried. A single norm of the residual tests
whichever block carries the larger units: the mechanical one is a force per radian of order
σ r ≈ 10⁸ while the hydraulic one is a mass rate of order 10⁻⁷, so `‖R‖ < 10⁻⁹` is not
strict, it is a mechanical test with the hydraulic block along for the ride. Scaling that
norm by the applied traction inverts the failure rather than removing it: after the cavity
pressure is released the hydraulic residual is fifteen orders of magnitude below the
mechanical scale, every step converges at the first iteration, and the pressure field simply
stops evolving — a smooth, plausible, entirely frozen answer.
"""
function poroplast_step!(
        m::PoroplastModel, nodes, states, dofs, Δt;
        σ_inner, σ_outer, p_inner, p_outer,
        maxiter = 40, tol = 1.0e-8, u_scale = 1.0e-3, p_scale = 1.0e5
    )
    nn = length(nodes)
    ndof = 2nn
    r_in, r_out = first(nodes), last(nodes)

    ## Dirichlet on the pressure at both ends; the displacement is traction-driven, and the
    ## hoop term `u/r` leaves no rigid-body mode to pin.
    fixed = (2, ndof)
    values = (p_inner, p_outer)

    dofs = copy(dofs)
    for (k, i) in pairs(fixed)
        dofs[i] = values[k]
    end

    new_states = deepcopy(states)
    local measure = Inf
    for iter in 1:maxiter
        R = zeros(eltype(dofs), ndof)
        K = zeros(eltype(dofs), ndof, ndof)

        for e in 1:(nn - 1)
            idx = (2e - 1, 2e, 2e + 1, 2e + 2)
            de = [dofs[i] for i in idx]
            re, se = poroplast_element_residual(
                m, de, de, nodes[e], nodes[e + 1], states[e], Δt
            )
            new_states[e] = se
            Ke = ForwardDiff.jacobian(
                d -> first(
                    poroplast_element_residual(
                        m, d, de, nodes[e], nodes[e + 1], states[e], Δt
                    )
                ),
                de,
            )
            for (a, ia) in pairs(idx)
                R[ia] += re[a]
                for (b, ib) in pairs(idx)
                    K[ia, ib] += Ke[a, b]
                end
            end
        end

        ## Applied tractions.
        R[1] -= -σ_inner * r_in
        R[ndof - 1] -= σ_outer * r_out

        ## Dirichlet rows.
        for (k, i) in pairs(fixed)
            R[i] = dofs[i] - values[k]
            K[i, :] .= 0
            K[i, i] = 1
        end

        δ = K \ R
        dofs -= δ

        ## Each field against its own scale, so neither can hide behind the other.
        measure = max(
            maximum(abs, @view δ[1:2:end]) / u_scale,
            maximum(abs, @view δ[2:2:end]) / p_scale,
        )
        measure <= tol && return dofs, new_states
    end
    error(
        "poroplast_step!: Newton did not converge — scaled increment $measure after " *
            "$maxiter iterations (tolerance $tol, scales u = $u_scale, p = $p_scale)"
    )
end
