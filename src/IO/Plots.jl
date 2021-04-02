module Plots

using GLMakie
using SuperFluids

export updatePlot!, createPlot!

"Abstract supertype for numerical models."
abstract type AbstractPlot{F} end
"Abstract supertype for numerical models."
abstract type AbstractPlot2D{F} <: AbstractPlot{F} end
"Abstract supertype for numerical models."
abstract type AbstractPlot3D{F} <: AbstractPlot{F} end

"""
    Plot2D{F<:AbstractField2D} <: AbstractPlot2D{F}

Type representing a 2D plot on a 2D grid.
"""
struct Plot2D{F<:AbstractField2D} <: AbstractPlot2D{F}
   f :: Observable
   modϕ :: Observable
   modϕsurf :: Observable
   reϕ :: Observable
   imϕ :: Observable
end

function Plot(f::AbstractField2D)
   nf = Node(f)
   modϕ = @lift(real($nf.ϕ.*conj.($nf.ϕ)))
   modϕsurf = @lift(0.25*$nf.ϕ.*conj.($nf.ϕ)*max($nf.g.Lx,$nf.g.Ly)/maximum(real($nf.ϕ)))
   realϕ = @lift(real($nf.ϕ))
   imagϕ = @lift(imag($nf.ϕ))
   return Plot2D{typeof(f)}(nf,modϕ,modϕsurf,realϕ,imagϕ)
end

function createPlot!(p::AbstractPlot2D)
   # references
   f = p.f.val
   x = f.g.x
   y = f.g.y
   # plot
   fig = Figure(resolution = (600, 600))
   display(fig)
   ax1 = fig[1, 1] = Axis(fig, title = "Module ϕ")
   ax2 = fig[1, 2] = LScene(fig, scenekw = (camera = cam3d!, raw = false), title = "Module ϕ 3D")
   ax3 = fig[2, 1] = Axis(fig, title = "Real ϕ")
   ax4 = fig[2, 2] = Axis(fig, title = "Imag ϕ")
   heatmap!(ax1,x,y,p.modϕ)
   surface!(ax2,x,y,p.modϕsurf)
   heatmap!(ax3,x,y,p.reϕ)
   heatmap!(ax4,x,y,p.imϕ)
   return fig, ax1, ax2, ax3, ax4
end

function updatePlot!(p::AbstractPlot2D)
   p.f[] = p.f.val
end

end
