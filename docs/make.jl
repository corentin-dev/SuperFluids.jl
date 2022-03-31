using Documenter
using Literate
using SuperFluids
using MPI

MPI.Init()

# DocMeta.setdocmeta!(SuperFluids, :DocTestSetup, :(using SuperFluids); recursive=true)

examples = [
    joinpath("examples/QT_TG_2D/QT_TG_2D.jl"),
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
        "Equations" => [
            "grosspitaevskii.md",
        ],
        "Library" => [
            "grid.md",
            "field.md",
            "init.md",
            "plan.md",
            "gradientfield.md",
        ],
        "Examples" => [
           "generated/QT_TG_2D.md",
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
