module Plots

using ..SuperFluids
import ..SuperFluids: AbstractField, AbstractField2D, AbstractField3D

export updatePlot!, createPlot!, Plot

using Requires: @require

struct PlotGL end
struct PlotPNG end
struct PlotWGL end

"Abstract supertype for numerical models."
abstract type AbstractPlot{F} end
"Abstract supertype for numerical models."
abstract type AbstractPlot2D{F} <: AbstractPlot{F} end
"Abstract supertype for numerical models."
abstract type AbstractPlot3D{F} <: AbstractPlot{F} end

function __init__()
    @require GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a" @eval include("plot-makie.jl")
end

end
