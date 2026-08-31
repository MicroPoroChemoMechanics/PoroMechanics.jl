# # Poroplasticity — a borehole in a plastic ground, and a law confirmed twice
#
# `base/Poroplast` is a cavity of radius 3 m in a saturated ground held at −11.5 MPa with
# 4.7 MPa of pore pressure. Its support and its internal pressure are released together over
# 1.5×10⁶ s — about eighteen days — and the ground is then left to drain for ten years. One
# dimension, axial symmetry, a Drucker-Prager skeleton under a Biot coupling.
#
# It is the second case this package was compared with Bil on, and its interest is not the
# agreement. It is that **the explanation found on `Richards-2d` predicted this one before
# it was measured**. One observation is an anecdote; the same first-order lag, in a model
# sharing no equations with the first, is a property of the code.
#
# ## The agreement
#
# Against Bil with its own time step refined to `Dtmax = 1e6` s:
#
# | date | pressure | displacement |
# |---:|---:|---:|
# | 1.5×10⁶ s — end of the release | 1.40e-4 | 2.01e-4 |
# | 5.0×10⁷ s | 9.34e-5 | 3.93e-4 |
# | 3.0×10⁸ s | 7.01e-4 | 1.40e-3 |
#
# Against the deck's own `Dtmax = 1e8`, on the same sampling:
#
# | date | pressure | displacement |
# |---:|---:|---:|
# | 1.5×10⁶ s | 1.40e-4 | 2.01e-4 |
# | 5.0×10⁷ s | **2.81e-3** | **2.85e-3** |
# | 3.0×10⁸ s | **4.47e-3** | 1.12e-3 |
#
# Read the two tables together and the shape is unmistakable. **The release phase does not
# move at all** — 1.40e-4 and 2.01e-4 in both, to every digit — because Bil's step
# controller was already resolving it. Only the long drainage changes, and there by a factor
# of thirty on the pressure. That is the signature seen on `Richards-2d`: refining Bil's own
# step collapses the gap where its step was coarse, and leaves untouched what it already
# resolved.
#
# ## Why that was predictable
#
# `Poroplast.cpp` computes its mass flux like this:
#
# ```c
# /* Transfer coefficient */
# double k_h = val_n.Permeability_liquid;
# ...
# for(i = 0 ; i < 3 ; i++) w_l[i] = - k_h*gpl[i];
# ```
#
# `val_n` is the **previous** step. The same explicit lag as `Richards.cpp`, in a model that
# shares none of its equations — and it costs the same first order in time. Two independent
# models, one code, one habit.
#
# That is worth more than either measurement on its own. After `Richards-2d` the lag was an
# explanation; after `Poroplast` it is a prior. The next Bil model this package is compared
# with should be assumed to lag its transfer coefficients until shown otherwise, and its
# reference should be refined before any gap is attributed here.
#
# ## Two errors this case found, both silent
#
# Neither would have shown up as a crash, a warning, or an implausible-looking picture.
#
# ### An initial stress is an *effective* stress
#
# The deck quotes ``\sigma_0 = -11.5`` MPa with ``p_0 = 4.7`` MPa. The skeleton does not
# start there: it starts at
#
# ```math
# \boldsymbol{\sigma}'_0 = \boldsymbol{\sigma}_0 + \beta\,p_{l0}\,\mathbf{I} = -7.74 \text{ MPa}
# ```
#
# Hand the total stress to the skeleton instead and the initial state is out of equilibrium
# by ``\beta p_0`` — 3.76 MPa here — so the very first step produces a smooth, plausible and
# entirely spurious wave of displacement and pressure. `poroplast_initial_states` therefore
# takes the *total* stress a deck quotes and converts it, because that is the step nobody
# notices getting wrong.
#
# The conversion uses ``\beta`` and not ``b``. Bil converts with the coefficient it calls
# `beta` — `sig += beta*pl` before its return mapping and `-=` after — while `b` drives its
# incremental update and the elastic part of its porosity. The two are equal in this deck,
# both 0.8, so the choice is invisible here. It is written down because the deck that
# separates them is the one that would expose a wrong guess.
#
# The check that caught it takes one line to state: **at the initial state, nothing may
# move**. Imposing the deck's own stresses and pressures and stepping once must return
# ``u = 0`` and ``p_l`` unchanged, exactly. It does, to machine precision — and it did not,
# before.
#
# ### A single residual norm cannot judge a two-field problem
#
# The mechanical block of this residual is a force per radian, of order ``\sigma r \approx
# 10^8``. The hydraulic block is a mass rate, of order ``10^{-7}``. Fifteen orders of
# magnitude apart, in the same vector.
#
# So `‖R‖ < 1e-9` is not a strict test. It is a test of the mechanical block, with the
# hydraulic one along for the ride. And scaling that norm by the applied traction — the
# obvious repair — inverts the failure rather than removing it: once the cavity pressure is
# released, the hydraulic residual sits far below the mechanical scale, every step converges
# on its first iteration, and **the pressure field stops evolving entirely**. The answer
# stayed smooth, monotone and completely wrong.
#
# What works is what Bil does: measure the Newton increment against a scale *per field*. Its
# `Objective Variations` block names exactly those two scales, `u_1 = 1e-3` and `p_l = 1e5`,
# and it names them because the same problem exists there.
#
# ## What is left
#
# The residual 1e-4 to 1e-3 is not explained further here, and should not be assumed to be
# the last of the time discretisation. Two candidates remain, both first-order and both
# Bil's:
#
# * its stress update is **incremental** from ``\sigma_n`` — `sig = sig_n + C:deps -
#   b dp` then return mapping — while [`DruckerPrager`](@ref) predicts from the plastic
#   strain, which is the path-independent choice and the same distinction as
#   [`ExplicitPredictor`](bbm_bil.md) on the Barcelona Basic Model;
# * its tangent is built by finite differences with a perturbation set by the deck.
#
# Neither has been isolated. Saying which of them carries the residual would need the same
# two-sided study that settled `Richards-2d`, and it has not been done.

using PoroMechanics
using Tensors
using Printf

include(joinpath(@__DIR__, "..", "test", "bil", "cases.jl"))

# ## Reproducing the comparison
#
# Both halves need Bil: the source tree for the deck and its mesh, the binary to refine the
# time step. The tables above were measured on 2026-08-30 with Bil 2.11.

if bil_root() === nothing || bil_executable() === nothing
    @info "Bil not available — see the tables above for the measured values."
else
    reference = poroplast_bil()
    ours = poroplast_ours()

    n = length(ours) ÷ (2 * length(POROPLAST_PROBES))
    rel(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), eps())

    println("Poroplast against Bil refined to Dtmax = $(POROPLAST_BIL_DT) s")
    @printf("  %-12s %14s %14s\n", "date [s]", "pressure", "displacement")
    for (k, probe) in pairs(POROPLAST_PROBES)
        block = (2k - 2) * n
        p_range = (block + 1):(block + n)
        u_range = (block + n + 1):(block + 2n)
        @printf(
            "  %-12.3e %14.4e %14.4e\n",
            POROPLAST_DATES[probe + 1],
            rel(ours[p_range], reference.values[p_range]),
            rel(ours[u_range], reference.values[u_range]),
        )
    end
end
