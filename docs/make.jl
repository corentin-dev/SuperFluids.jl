using Documenter
using Literate
using SuperFluids
using MPI

MPI.Init()

# DocMeta.setdocmeta!(SuperFluids, :DocTestSetup, :(using SuperFluids); recursive=true)

examples = [
    joinpath("examples/BEC_2D/BEC_2D_quadratic/BEC_2D.jl"),
    joinpath("examples/QT_TG_2D/QT_TG_2D.jl"),
    joinpath("examples/QT_TG_3D/QT_TG_3D.jl"),
]

for example in examples
    Literate.markdown(example, "docs/src/generated"; flavor = Literate.DocumenterFlavor())
end

makedocs(
    authors = "Corentin Lothode <corentin.lothode@univ-rouen.fr> and contributors.",
    repo = "https://plmlab.math.cnrs.fr/lmrs/num/SuperFluids.jl",
    sitename = "SuperFluids.jl",
    format = Documenter.HTML(),
    doctest = false,
    strict = false,
    clean = true,
    modules = [SuperFluids],
    pages = [
        "Home" => "index.md",
        "QuickStart" => "quickstart.md",
        "Physic models" => [
            "Gross-Pitaevskii" => [
                "grosspitaevskii/grosspitaevskii.md",
                "grosspitaevskii/init.md",
                "grosspitaevskii/potential.md",
                "grosspitaevskii/nummodel.md",
            ]
        ],
        "Library" => [
            "grid.md",
            "field.md",
            "plan.md",
            "gradientfield.md",
        ],
        "Examples" => [
           "generated/BEC_2D.md",
           "generated/QT_TG_2D.md",
           "generated/QT_TG_3D.md",
        ],
        "Plots" => [
            "plots/plots.md",
        ],
        "API" => "api.md",
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
