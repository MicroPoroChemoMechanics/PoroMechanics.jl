# ChemistryLab sensitivity patch

The examples and tests use ChemistryLab **0.15.2** with the local patch in this
folder. This is an upstream correction, maintained in a separate ChemistryLab
checkout under the ignored `local/` directory, not an alternative chemistry solver.

On a fresh checkout, run:

```sh
julia +1.12 scripts/prepare_chemistrylab.jl
julia +1.12 --project=examples -e 'using Pkg; Pkg.instantiate()'
julia +1.12 --project -e 'using Pkg; Pkg.test(test_args=["chemistry"])'
```

The preparation script checks out the exact `v0.15.2` commit, checks the patch before
applying it, and recognizes an already applied patch. The `[sources]` entries in
`examples/Project.toml` and `test/Project.toml` select this checkout. Julia's installed
package cache is not modified. Git and network access are needed for the initial clone.

The patch keeps trace aqueous species in the sensitivity problem, pins absent pure
phases, differentiates log activities below the primal optimizer's regularization
floor, and scales the KKT equations before solving. It rejects responses that fail
stationarity or differentiated conservation. Certificates inspect primal values.
Sensitivities apply on a fixed phase branch; phase appearance/disappearance can make
the equilibrium map nondifferentiable.

Upstream regression tests are included in the patch and can be run with:

```sh
julia +1.12 --project=examples local/ChemistryLab-0.15.2/test/trace_sensitivity.jl
```

Once an upstream release contains these fixes, revalidate the OPC and trace tests,
then remove the patch, preparation step and source overrides together.
