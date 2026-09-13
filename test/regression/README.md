# Regression references

References record numerical behavior; they do not establish physical accuracy.
Regenerate a reference only for an intentional change, and document the reason.

## Certified chloride chemistry

The chloride reference was renewed when `run_4.jl` switched its transient Gibbs
equilibria and Friedel sensitivities to ChemistryLab's certified API. Failed
certification now aborts the segment. The reference's relative L2 tolerance remains
**1e-6**.

The old interior-point path returned 24 uncertified transient states in the small
regression case. These states conserved elements but failed first-order optimality.
Changing only OptimaSolver from 0.5.1 to 0.5.5 changed the certified initial amounts
by less than 7e-12 mol, yet the unconverged transient solves amplified that difference
into a failed regression. Reusing the exact 0.5.1 initial state under 0.5.5 recovered
the entire old signature bit for bit. Changing BLAS from four threads to one did
not affect the 0.5.1 result on the tested Mac.

Controlled comparisons used Julia 1.12.7, patched ChemistryLab 0.15.2, 12 cells,
two saved profiles over 0.1 year, and identical remaining dependencies:

| Transient chemistry | Relative L2 difference between OptimaSolver 0.5.1 and 0.5.5 | Certified states per run |
| :--- | ---: | ---: |
| Historical interior-point solver | Approximately 1.17e-4 | 0/24 |
| Certified solver | Approximately 5.75e-11 | 24/24 |

At node 8 of the first saved profile, adsorbed calcium changes from approximately
20.311 to 19.483 mol/m³ of medium. This is an intentional consequence of enforcing
equilibrium, not a tolerance adjustment. The comparison below shows both saved
profiles; the two certified dependency versions overlap at this scale.

![Adsorbed calcium before and after certifying transient chemistry](../../docs/src/assets/chloride_certified_regression.png)

To reproduce the reference and checks from the repository root:

```sh
julia +1.12 scripts/prepare_chemistrylab.jl
julia +1.12 --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia +1.12 --project=examples test/regression/generate.jl chloride_ingress
julia +1.12 --project -e 'using Pkg; Pkg.test(test_args=["regression", "chemistry"])'
```

The initial interior-point *guess* can still emit `MaxIters`; the initialization
adapter certifies the state it actually accepts. The separate legacy-optimizer
test remains an expected failure. Local equilibrium certification does not validate
the accuracy or global conservation of the full SNIA splitting scheme, nor does
this Mac comparison replace Linux/Windows CI checks.
