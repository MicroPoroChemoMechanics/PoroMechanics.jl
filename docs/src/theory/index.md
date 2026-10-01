# Theory: a first course in porous media

A wet sponge becomes thinner when squeezed, and water leaves its pores. A saturated
soil beneath a new building behaves in a related way: loading changes both its shape
and its pore pressure, and drainage allows further settlement. **Poromechanics studies
this interaction between deformation and fluid movement.**

These pages explain the equations used in PoroMechanics.jl for a reader encountering
the subject for the first time. You need elementary calculus, force balance, and
Hooke's law for a spring. Vectors and tensors are introduced as they become necessary.
The goal is to help you choose a model, understand its parameters, and interpret a
computed result.

## Reading path

| Read | Question it answers | Then try |
|:---|:---|:---|
| [1. Porous media and notation](basics.md) | What are the skeleton, porosity, stress, and pressure? | Identify the phases in your material. |
| [2. Balance laws](balances.md) | What must be conserved, and what needs a material law? | Write the water balance for a column. |
| [3. Transport](transport.md) | Why do water and dissolved substances move? | [Darcy column](../examples/darcy_column.md) and [Fickian diffusion](../examples/fickian_diffusion.md) |
| [4. Saturated poroelasticity](poroelasticity.md) | How do pressure and deformation affect each other? | [Biot consolidation](../examples/biot_consolidation.md) |
| [5. Consolidation](consolidation.md) | Why does settlement take time? | [Terzaghi](../validation/terzaghi.md), then [Mandel](../validation/mandel.md) |
| [6. Unsaturated media](unsaturated.md) | What changes when air occupies some pores? | [Richards infiltration](../examples/richards_1d.md) |
| [7. Drying and temperature](drying.md) | How do evaporation and heat transport interact? | [Non-isothermal drying](../examples/nonisothermal_drying.md) |
| [8. From equations to a solver](numerics.md) | How does a balance become a computer calculation? | [Writing a model](../demos/writing_a_model.md) |

For a first pass through saturated poromechanics, read chapters 1–5. Chapters 6–7
introduce additional physics; chapter 8 connects the equations to the implementation.
Each chapter states its assumptions and points to the corresponding API or worked case.

## Scope and sources

The main mechanical setting is **small deformation, slow loading, and linear elastic
behavior**. Small deformation lets us work on a fixed reference geometry; slow loading
lets us neglect inertia. The transport-only models make additional assumptions, such as
an undeforming skeleton or a prescribed storage coefficient. A law available in the
constitutive API does not by itself imply that every coupled problem using it has a solver.

The progression follows Patrick Dangla's *Introduction à la mécanique des milieux poreux*
[dangla_notes](@cite): chapters 1–4 introduce the continuum description and thermodynamics,
chapters 5–6 develop saturated poroelasticity, and chapter 7 treats partial saturation.
These pages use new explanations and figures, and adapt the notation to the software.
The course is undated; chapter and equation references here refer to the 110-page version. More specialized laws retain their own references in
the [bibliography](../references.md).

The present introduction does not derive plasticity, homogenization, or reactive chemistry.
For their current implementation and validation scope, see
[material models](../api/materials.md), the
[homogenization benchmark](../validation/mfh_poroelastic.md), and
[reactive transport](../demos/reactive_transport.md). In particular, a successful equilibrium
calculation does not establish the accuracy of a complete reactive transport history.
