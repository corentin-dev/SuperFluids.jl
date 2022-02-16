using Documenter
using SuperFluids

DocMeta.setdocmeta!(SuperFluids, :DocTestSetup, :(using SuperFluids); recursive=true)

makedocs(
    authors = "Corentin Lothode <corentin.lothode@univ-rouen.fr> and contributors.",
    repo = "https://plmlab.math.cnrs.fr/lmrs/num/SuperFluids.jl",
    sitename = "SuperFluids.jl",
    format = Documenter.HTML(),
    doctest = false,
    modules = [SuperFluids]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
