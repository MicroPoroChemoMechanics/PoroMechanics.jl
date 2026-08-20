# PoroMechanics.jl

**PoroMechanics.jl** simulates coupled phenomena in porous media: flow, solute and
reactive transport, cement chemistry, and poromechanics — on finite volume and finite
element backends.

## Covered domains

- Flow in porous media (Richards, Darcy)
- Solute transport (Fick)
- Non-isothermal drying (M6)
- Poromechanics (BBM, M7)
- Structural mechanics
- Cement chemistry

## Backends

| Model type | Library |
|---|---|
| Diffusion / transport (M1, M2, M6, Richards…) | [VoronoiFVM.jl](https://github.com/j-fu/VoronoiFVM.jl) |
| Coupled mechanics (M7, BBM, M10…) | [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl) |

## Installation

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url = "https://github.com/MicroPoroChemoMechanics/MPCM-Registry"))
Pkg.add("PoroMechanics")
```

## Quick start

```julia
using PoroMechanics

# A physics model is a subtype of AbstractPoroModel:
struct MyDiffusionModel <: AbstractPoroModel
    D::Float64   # diffusion coefficient
end

function PoroMechanics.storage!(f, u, node, m::MyDiffusionModel, data)
    f[1] = u[1]        # ∂c/∂t
end

function PoroMechanics.flux!(f, u, edge, m::MyDiffusionModel, data)
    f[1] = -m.D * (u[1,1] - u[1,2])   # -D ∇c · n
end
```

## Validated examples

| Example | Physics | Backend | Unknown(s) |
|---|---|---|---|
| [Darcy 1D](examples/darcy_column.md) | Linear saturated flow | VoronoiFVM | $p$ |
| [Richards 1D — M1](examples/M1_Richards.md) | Unsaturated imbibition (Van Genuchten) | VoronoiFVM | $p_l$ |
| [Non-isothermal Drying — M6](examples/M6_drying.md) | Thermo-hydraulic coupling, 2 materials, radioactive waste barrier | VoronoiFVM | $p_l$, $p_a$, $T$ |
| [Biot 2D — Ternay dam (M7)](examples/M7_Biot.md) | Saturated poroelasticity, 2-material FEM | Ferrite | $\mathbf{u}$, $p_l$ |

## Documentation

```@contents
Pages = ["examples/darcy_column.md", "examples/M1_Richards.md", "examples/M6_drying.md", "examples/M7_Biot.md"]
```

## API Reference

```@autodocs
Modules = [PoroMechanics]
```
