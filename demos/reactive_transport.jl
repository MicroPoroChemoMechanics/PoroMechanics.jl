# # Reactive transport: models, inventories and chemistry
#
# PoroMechanics supplies two reusable transport formulations. Choose according to
# the unknowns and the physics required:
#
# | Model | Unknowns | Stored quantities | Local closure |
# |:---|:---|:---|:---|
# | `NernstPlanck` | Ion concentrations and electric potential | `phi * c` | Electroneutrality |
# | `EquilibratedTransport` | Total component inventories | The totals themselves | Certified chemical equilibrium |
#
# The second model currently uses a Fickian flux per component and **omits migration**.
# Combining these two formulations requires a consistent species-to-component flux
# mapping; wrapping one in the other does not supply that missing physics.
#
# ## A short ionic transport calculation
#
# Consider NaCl entering a saturated column. Concentrations are in mol/m³ of pore
# solution; the stored inventories are in mol/m³ of medium. The potential is
# dimensionless, `Psi = F * electric_potential / (R * temperature)`.
#
# Without a tortuosity law, `D` is the effective diffusivity per bulk area. With a
# law such as `OhJang`, it is a free-water diffusivity multiplied by `D_eff / D_free`.
# There is no additional porosity multiplier in the flux.

using PoroMechanics
using VoronoiFVM
using ExtendableGrids

grid = simplexgrid(range(0.0, 0.01; length = 21))
model = NernstPlanck(;
    phi = 0.3,
    D = (0.3 * 2.032e-9, 0.3 * 1.334e-9), z = (-1, 1),
    dirichlet = (((1, 100.0),), ((1, 100.0),), ((1, 0.0),)),
)
sys = fvm_system(model, grid; reaction = true)
initial = unknowns(sys; inival = 1.0)
initial[ipot(model), :] .= 0.0
initial[1:2, 1] .= 100.0
control = VoronoiFVM.SolverControl(;
    Δt = 1.0, Δt_min = 1.0e-4, Δt_max = 100.0, Δu_opt = 1.0,
)
solution = solve(sys; inival = initial, times = [0.0, 1.0e4], control)
final = solution(1.0e4)

# `reaction = true` assembles the algebraic electroneutrality equation. The last
# row has zero storage and zero flux; omitting the reaction leaves it undetermined.
# The left boundary fixes a potential reference. The right boundary has zero ionic
# flux, which makes the conserved current zero throughout this one-dimensional case.
# A prescribed voltage difference would describe a different electrical problem.
# Initial and boundary concentrations must satisfy `sum(z .* c) == q_background`.
# The default is zero; a nonzero value represents the opposite of an untransported
# background charge.

charge_error = maximum(abs, net_charge(model, final))
current, current_scale = edge_current(model, final, 0.01 / 20)
relative_current = maximum(abs, current) / maximum(current_scale)
(charge_error, relative_current)

# Both diagnostics should be small. The unequal ion mobilities produce a diffusion
# potential while the ions enter together. The tests additionally compare with an
# analytical binary-electrolyte reduction and check integrated mass conservation.
#
# ## Transporting inventories through local equilibrium
#
# This route needs the examples environment, including the patched ChemistryLab
# dependency prepared by `julia scripts/prepare_chemistrylab.jl`. Load:
#
# ```julia
# using PoroMechanics, ChemistryLab, DynamicQuantities, OptimaSolver
# ```
#
# ChemistryLab and DynamicQuantities activate the PoroMechanics extension;
# OptimaSolver activates ChemistryLab's certified solver. Chemical equilibrium,
# phase selection and implicit equilibrium sensitivities remain in ChemistryLab.
#
# Construct a `ComponentSet` from the names of the transported chemical primaries,
# their charge numbers `z`, their diffusivities `D`, and a named tuple `fixed` of
# all remaining primary totals. The complete basis is checked at construction;
# the charge row defaults to zero. Unlike species amounts, a component total can
# be signed, depending on the chosen chemical basis.
#
# ```julia
# model = equilibrated_transport(components, chemical_system;
#     initial_state = chemical_seed, phi = 0.121, dirichlet = boundary_totals)
# sys = fvm_system(model, grid)
# ```
#
# Here each row stores a total in mol/m³ of medium, including aqueous and solid
# contributions. Dirichlet values use these same units. The chemical seed supplies
# a nonnegative starting composition for one m³ of medium, temperature and pressure;
# the evolving totals are passed separately to ChemistryLab as its conservation
# vector. The flux uses dissolved component concentrations in mol/m³ of solution.
# Multiplying this storage by porosity again would count porosity twice.
#
# `equilibrium_state(model, totals)` returns a state and its certificate.
# `speciate(model, totals)` returns aqueous component concentrations and a success
# flag. A failed certificate raises `LocalEquilibriumError` when computing a flux;
# solver exceptions also propagate. Configure the transient solver to retry failed
# steps with smaller time increments, and check that it reaches the requested end.
#
# The runnable cement recipe is
# `examples/chloride_ingress/repro_equilibrated_transport.jl`. It prints the
# certificate, component balance error and concentration sensitivities without
# redefining a transport model. The corresponding chemistry tests also run a
# closed-column transient and verify total conservation.
#
# ## What stays in the examples
#
# Chemical species selection, cement composition, mesh, initial state, loading and
# solver controls belong to a case. The double-layer and AFm prototypes also remain
# there for now: their chemical laws must move to ChemistryLab before becoming a
# shared transport API. They already reuse the package's `NernstPlanck` callbacks.
# Older includes of `nernst_planck.jl` and `equilibrated_transport.jl` remain as
# compatibility entry points. New code can import the package directly; supply
# `phi`, `D` and `z` explicitly when constructing `NernstPlanck`.
