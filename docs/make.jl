using Documenter
using PoroMechanics

makedocs(;
    modules  = [PoroMechanics],
    sitename = "PoroMechanics.jl",
    authors  = "Jean-François Barthélémy and Anthony Soive",
    format   = Documenter.HTML(;
        prettyurls       = get(ENV, "CI", "false") == "true",
        canonical        = "https://MicroPoroChemoMechanics.github.io/PoroMechanics.jl",
        edit_link        = "main",
        assets           = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => [
            "1D Diffusion (M1)"             => "examples/M1_diffusion.md",
            "Darcy 1D"                      => "examples/darcy_column.md",
            "Richards 1D (M1)"              => "examples/M1_Richards.md",
            "Non-isothermal Drying (M6)"    => "examples/M6_drying.md",
            "Biot 2D — Ternay (M7)"         => "examples/M7_Biot.md",
        ],
    ],
    warnonly = true,
)

deploydocs(;
    repo         = "github.com/MicroPoroChemoMechanics/PoroMechanics.jl",
    devbranch    = "main",
    push_preview = true,
)
