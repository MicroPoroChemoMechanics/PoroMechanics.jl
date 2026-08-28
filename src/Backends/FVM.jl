"""
Finite volume backend — the glue between a model and `VoronoiFVM.jl`.

`VoronoiFVM.System` takes four-argument callbacks `(f, u, node, data)`, while a model here
carries its own type and is selected by dispatch, so its callbacks take five arguments
`(f, u, node, model, data)`. Every example used to bridge that gap with the same three
closures, copied verbatim:

```julia
_storage!(f, u, node, data) = PoroMechanics.storage!(f, u, node, m, data)
_flux!(f, u, edge, data) = PoroMechanics.flux!(f, u, edge, m, data)
_bcondition!(f, u, bnode, data) = PoroMechanics.bcondition!(f, u, bnode, m, data)
```

[`fvm_system`](@ref) does it once.
"""

"""
    fvm_system(model, grid; species = 1:nspecies(model), reaction = false, kwargs...)

Build a `VoronoiFVM.System` for `model` on `grid`, wiring [`storage!`](@ref),
[`flux!`](@ref) and [`bcondition!`](@ref) to the model by dispatch.

`reaction = true` also wires [`reaction!`](@ref) — for models carrying volumetric source
terms or algebraic constraints such as electroneutrality. It is off by default so that
models without reactions do not pay for differentiating a no-op.

Any further keyword argument is forwarded untouched to `VoronoiFVM.System`.

```julia
sys = fvm_system(model, grid)
tsol = solve(sys; inival, times, control)
```
"""
function fvm_system(
        model::AbstractPoroModel, grid;
        species = 1:nspecies(model),
        reaction::Bool = false,
        kwargs...
    )
    callbacks = (
        storage = (f, u, node, data) -> storage!(f, u, node, model, data),
        flux = (f, u, edge, data) -> flux!(f, u, edge, model, data),
        bcondition = (f, u, bnode, data) -> bcondition!(f, u, bnode, model, data),
    )
    if reaction
        callbacks = merge(
            callbacks,
            (reaction = (f, u, node, data) -> reaction!(f, u, node, model, data),),
        )
    end
    return VoronoiFVM.System(grid; callbacks..., species = collect(species), kwargs...)
end
