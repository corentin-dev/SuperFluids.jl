using Documenter
using SuperFluids


DocMeta.setdocmeta!(
     SuperFluids, :DocTestSetup,
     quote
        using SuperFluids
     end;
     recursive=true,
    )

makedocs(
    sitename = "SuperFluids",
    format = Documenter.HTML(),
    modules = [SuperFluids]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
