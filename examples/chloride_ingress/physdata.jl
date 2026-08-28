# physdata.jl — invariant physico-chemical data (cemdata18 + literature)
#
# NOTE — this belongs in ChemistryLab.jl. These are database values, not transport
# parameters: equilibrium constants, free-water diffusivities and molar volumes. They sit
# here only until ChemistryLab.jl exposes them.
#
# Constants that depend neither on the material nor on the test:
#   DLMConstants       : surface equilibrium constants on C-S-H (Tran & Soive 2018)
#   IonicDiffusivities : free-water diffusion coefficients (Atkins 1987, Oelkers 1988)
#   MolarVolumes       : molar volumes of the solid phases (cemdata18, Lothenbach 2019)
#
# Usage:
#   include("physdata.jl")    # from tran2018.jl, m100_ternary.jl, etc.

# ════════════════════════════════════════════════════════════════════════════════
# Invariant physico-chemical data
# ════════════════════════════════════════════════════════════════════════════════

"""
Double Layer Model constants (Tran & Soive 2018, Cem. Concr. Res. 110, 70–85),
extended to Mg²⁺ for seawater. All values are equilibrium constants taken from
the literature — they depend neither on the material nor on the test.
"""
Base.@kwdef struct DLMConstants
    Ka1::Float64      = 2.0e-10    # [mol/m³]  ≡SiOH → ≡SiO⁻ + H⁺
    K_Ca::Float64     = 2.0        # [m³/mol]  ≡SiO⁻ + Ca²⁺ → ≡SiOCa⁺
    K_Mg::Float64     = 0.10       # [m³/mol]  ≡SiO⁻ + Mg²⁺ → ≡SiOMg⁺  (seawater)
    K_OHCl::Float64   = 4.47e-4    # [m³/mol]  ≡SiOH + Cl⁻  → ≡SiOHCl⁻
    K_Na_Tob::Float64 = 1.106e-3   # [m³/mol]  ≡SiO⁻ + Na⁺  → ≡SiONa  (x_CaS = 0.83)
    K_Na_Jen::Float64 = 9.0e-5     # [m³/mol]  ≡SiO⁻ + Na⁺  → ≡SiONa  (x_CaS = 1.67)
    Gamma_max::Float64 = 1.3e-6    # [mol/m²_CSH] density of silanol sites
    a_s::Float64      = 85000.0    # [m²_CSH/mol_CSH] BET specific surface area
    eps_r::Float64    = 78.5       # [-] relative permittivity of water
    Kw_SI::Float64    = 6.76e-9    # [mol²/m⁶] ionic product of water at 20 °C
end

"""
Ionic diffusion coefficients in free water [m²/s].
Sources: Atkins (1987) for the common ions, Oelkers & Helgeson (1988) for Si,
Li & Gregory (1974) for Al(OH)₄⁻.
"""
Base.@kwdef struct IonicDiffusivities
    D_Cl::Float64  = 2.032e-9   # Cl⁻
    D_Na::Float64  = 1.334e-9   # Na⁺
    D_K::Float64   = 1.957e-9   # K⁺
    D_Ca::Float64  = 0.792e-9   # Ca²⁺
    D_Mg::Float64  = 0.706e-9   # Mg²⁺
    D_SO4::Float64 = 1.065e-9   # SO₄²⁻
    D_Si::Float64  = 1.000e-9   # Si total (H₄SiO₄, Oelkers & Helgeson 1988)
    D_Al::Float64  = 0.541e-9   # total Al (Al(OH)₄⁻ dominant at pH > 12, Li & Gregory 1974)
end

"""
Molar volumes of the solid phases [m³/mol] from cemdata18
(Lothenbach et al. 2019, Table 1) and from the literature.
"""
Base.@kwdef struct MolarVolumes
    Vm_CH::Float64     = 33.06e-6    # Portlandite Ca(OH)₂
    Vm_ett::Float64    = 707.0e-6    # Ettringite
    Vm_ms::Float64     = 309.0e-6    # Monosulfoaluminate
    Vm_fs::Float64     = 271.0e-6    # Friedel's salt C₄AClH₁₀
    Vm_brc::Float64    = 24.63e-6    # Brucite Mg(OH)₂
    Vm_TobD::Float64   = 27.7e-6     # CSHQ-TobD  C₀.₆₇SH₀.₅
    Vm_TobH::Float64   = 59.0e-6     # CSHQ-TobH  C₀.₆₇SH₁.₅
    Vm_JenH::Float64   = 107.7e-6    # CSHQ-JenH  C₁.₅SH₂.₅
    Vm_JenD::Float64   = 57.6e-6     # CSHQ-JenD  C₁.₅SH₀.₈₃
    Vm_MSH_08::Float64 = 94.885e-6   # M075SH  Mg₁.₅Si₂O₅.₅(H₂O)₂.₅ (Mg/Si=0.75)
    Vm_MSH_13::Float64 = 74.32e-6    # M15SH   Mg₁.₅SiO₃.₅(H₂O)₂.₅  (Mg/Si=1.50)
    Vm_LDH_M4::Float64 = 219.1e-6    # M4A-OH-LDH Mg₄Al₂(OH)₁₄(H₂O)₃ (Mg/Al=2)
    Vm_LDH_M6::Float64 = 305.44e-6   # M6A-OH-LDH Mg₆Al₂(OH)₁₈(H₂O)₃ (Mg/Al=3)
    Vm_LDH_M8::Float64 = 392.36e-6   # M8A-OH-LDH Mg₈Al₂(OH)₂₂(H₂O)₃ (Mg/Al=4)
    Vm_Gyp::Float64    = 74.69e-6    # secondary gypsum CaSO₄·2H₂O (cemdata18)
    # C-A-S-H (CNASH_ss / CSH3T, cemdata18 / Myers et al. 2014)
    # TODO: check against dict_sp["CSH3T-TobH"].standard_volume on the first run
    Vm_CASH_TobH::Float64 = 59.4e-6    # CSH3T-TobH  Ca₀.₈₃Al₀.₁Si₀.₉H… (Myers 2014)
    Vm_CASH_T5C::Float64  = 62.0e-6    # CSH3T-T5C   Ca₁.₂₅Al₀.₂₅Si₀.₇₅H…
    Vm_CASH_T2C::Float64  = 68.4e-6    # CSH3T-T2C   Ca₁.₁₇Al₀.₃₃Si₀.₆₇H…
end
