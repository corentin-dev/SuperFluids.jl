using Documenter
using Literate
using SuperFluids
using MPI

MPI.Init()

# DocMeta.setdocmeta!(SuperFluids, :DocTestSetup, :(using SuperFluids); recursive=true)
run_example = true

if run_example
    examples = [joinpath("examples/BEC_2D/BEC_2D.jl"),
                joinpath("examples/QT_TG_2D/QT_TG_2D.jl"),
                joinpath("examples/QT_TG_3D/QT_TG_3D.jl"),
                joinpath("examples/VR/VR_2D.jl")]
    examples_md = ["generated/BEC_2D.md",
                   "generated/VR_2D.md",
                   "generated/QT_TG_2D.md",
                   "generated/QT_TG_3D.md"]

    for example in examples
        Literate.markdown(example, "docs/src/generated"; flavor=Literate.DocumenterFlavor(),
                          credit=false)
    end
else
   examples_md = []
end

pages = ["Home" => "index.md",
        "QuickStart" => "quickstart.md",
        "Physic models" => ["Gross-Pitaevskii" => ["grosspitaevskii/grosspitaevskii.md",
                                                "grosspitaevskii/init.md",
                                                "grosspitaevskii/potential.md",
                                                "grosspitaevskii/nummodel.md"]],
        "Library" => ["grid.md",
                    "field.md",
                    "plan.md",
                    "gradientfield.md"],
        "Plots" => ["plots/plots.md"],
        "API" => "api.md"]

if run_example
    insert!(pages, 5, "Examples" => examples_md)
end

makedocs(; authors="Corentin Lothode <corentin.lothode@univ-rouen.fr> and contributors.",
         repo="https://plmlab.math.cnrs.fr/lmrs/num/SuperFluids.jl",
         sitename="SuperFluids.jl",
         format=Documenter.HTML(),
         doctest=false,
         strict=false,
         clean=true,
         modules=[SuperFluids],
         pages=pages)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
