# 4. Linear saturated poroelasticity

## Two ways to change the fluid content

A saturated specimen can accept more water because its pores expand, or because the
water already present is compressed. Biot's linear theory combines those effects with
elastic deformation [biot1941](@cite). We assume complete saturation, small deformation,
isothermal conditions, an isotropic elastic skeleton, and constant material coefficients.
The variables below are increments from an equilibrated reference state.

The constitutive equations are

```math
\boldsymbol\sigma=2G\boldsymbol\varepsilon
+\lambda\varepsilon_v\mathbf I-bp\mathbf I,
\qquad
\zeta=b\varepsilon_v+\frac{p}{M}.
```

The constants $G$ and $\lambda$ are drained elastic constants. The dimensionless Biot coefficient
$b$ measures coupling to pressure. The Biot modulus $M$ describes resistance to
fluid-content change at fixed strain. At fixed strain, increasing pressure produces
$\Delta\zeta=\Delta p/M$; at fixed pressure, expansion produces
$\Delta\zeta=b\Delta\varepsilon_v$.

## Effective stress

Rearrange the first equation:

```math
\boldsymbol\sigma'=\boldsymbol\sigma+bp\mathbf I
=2G\boldsymbol\varepsilon+\lambda\varepsilon_v\mathbf I.
```

This **effective stress** governs the elastic strain. Under a fixed compressive total
stress, increasing pore pressure makes effective stress less compressive, allowing the
skeleton to expand. Reducing pressure has the opposite effect. The coefficient $b$
need not equal one; the expression with $b=1$ is a special case.

## Drained elastic constants

A drained test allows enough exchange with a pressure-controlled reservoir to keep
excess pore pressure approximately zero. The measured Young's modulus $E$ and
Poisson's ratio $\nu$ then determine

```math
G=\frac{E}{2(1+\nu)}, \qquad
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}, \qquad
K_d=\lambda+\frac{2G}{3}=\frac{E}{3(1-2\nu)}.
```

The coefficient $K_d$ is the drained bulk modulus: under uniform isotropic stress,
$\sigma_m=K_d\varepsilon_v-bp$, where
$\sigma_m=\operatorname{tr}\boldsymbol\sigma/3$ is positive in tension.
The elastic stability conditions are $G>0$ and $K_d>0$; positive $M$ gives
positive storage at fixed strain.

## Undrained loading

In a homogeneous sealed test without internal redistribution, fluid content cannot change:
$\zeta=0$. The storage law gives $p=-bM\varepsilon_v$. Substituting into the
mean-stress relation yields

```math
\sigma_m=(K_d+b^2M)\varepsilon_v=K_u\varepsilon_v, \qquad
K_u=K_d+b^2M.
```

Thus the undrained bulk modulus $K_u$ exceeds the drained one: compressing trapped
water adds resistance. For an isotropic compression increment $\sigma_m=-P$,

```math
p=BP, \qquad B=\frac{bM}{K_d+b^2M}.
```

Here $B$ is Skempton's coefficient for this isotropic test. A laterally confined test
uses a different mechanical modulus, as the next chapter shows. Also, a sealed
boundary constrains total mass; it does not prohibit redistribution inside a
heterogeneous specimen.

## Translating Dangla's notation into the API

Dangla distinguishes pore-volume change from fluid-content change. In his chapter 5,

```math
\delta\phi=b\varepsilon_v+N_{\mathrm D}p, \qquad
\zeta=\delta\phi+\frac{\phi_0}{K_f}p,
\qquad \frac1M=N_{\mathrm D}+\frac{\phi_0}{K_f}.
```

Here $K_f$ is the fluid bulk modulus and $N_{\mathrm D}$ denotes the coefficient called
$N$ in the course (equations 5.13 and 5.23). In [`BiotPoroelastic`](@ref), **`N` is
already the total storage coefficient $1/M$**, in Pa⁻¹. Copying $N_{\mathrm D}$
into that field would omit fluid compressibility.

For a homogeneous isotropic solid constituent with bulk modulus $K_s$, the additional
relations in Dangla §5.5 give

```math
b=1-\frac{K_d}{K_s}, \qquad
N_{\mathrm D}=\frac{b-\phi_0}{K_s}, \qquad
N_{\mathrm{code}}=\frac{b-\phi_0}{K_s}+\frac{\phi_0}{K_f}.
```

These relations require the stated constituent assumptions; they are not universal
identities for every composite skeleton [dangla_notes](@cite).

| Theory | API | Meaning |
|:---|:---|:---|
| $E,\nu$ | `E`, `nu` | Drained elastic parameters |
| $b$ | `b` | Biot coefficient |
| $1/M$ | `N` | Storage at fixed strain, Pa⁻¹ |
| $M$ | `biot_modulus(material)` | Biot modulus, Pa |
| $K_d, K_u$ | `bulk_modulus`, `undrained_bulk_modulus` | Drained and undrained bulk moduli |
| $B$ | `skempton(material)` | Isotropic undrained pressure response |

For example, $K_d=5$ GPa, $K_s=25$ GPa, $K_f=2$ GPa, and $\phi_0=0.2$
give $b=0.8$, $N_{\mathrm D}=2.4\times10^{-11}$ Pa⁻¹ and
$N_{\mathrm{code}}=1.24\times10^{-10}$ Pa⁻¹, hence $M\simeq8.06$ GPa.
With $\nu=0.25$, $E=3K_d(1-2\nu)=7.5$ GPa:

```julia
material = BiotPoroelastic(;
    E = 7.5e9, nu = 0.25, b = 0.8, N = 1.24e-10,
    k = 1.0e-16, mu_l = 1.0e-3,
)
```

Permeability and viscosity determine the rate of drainage; they do not enter the
instantaneous elastic relations above. This material model does not include plastic
strain, damage, or large deformation. Continue with [consolidation](consolidation.md).
