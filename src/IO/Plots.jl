module Plots

using GLMakie
using SuperFluids

export updatePlot!, createPlot!, Plot

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
   # layout
   fig = Figure(resolution = (600, 600))
   display(fig)
   ax1 = fig[1, 1] = Axis(fig, title = "Module ϕ")
   ax2 = fig[1, 2] = LScene(fig, scenekw = (camera = cam3d!, raw = false), title = "Module ϕ 3D")
   ax3 = fig[2, 1] = Axis(fig, title = "Real ϕ")
   ax4 = fig[2, 2] = Axis(fig, title = "Imag ϕ")
   # create plots
   heatmap!(ax1,x,y,p.modϕ)
   surface!(ax2,x,y,p.modϕsurf)
   update_cam!(ax2.scene)
   heatmap!(ax3,x,y,p.reϕ)
   heatmap!(ax4,x,y,p.imϕ)
   return fig, ax1, ax2, ax3, ax4
end

function updatePlot!(p::AbstractPlot2D)
   p.f[] = p.f.val
end

struct Plot3D{F<:AbstractField3D} <: AbstractPlot3D{F}
   f :: Observable
   idx :: Observable
   idy :: Observable
   idz :: Observable
   isoϕ :: Real
   modϕx :: Observable
   modϕy :: Observable
   modϕz :: Observable
   modϕ :: Observable
end

function Plot(f::AbstractField3D)
   # references
   x, y, z = f.g.x, f.g.y, f.g.z
   # create nodes
   nf = Node(f)
   idx = Node(floor(Int,length(x)/2+1))
   idy = Node(floor(Int,length(y)/2+1))
   idz = Node(floor(Int,length(z)/2+1))
   isoϕ = 0.4
   modϕx = @lift(real($nf.ϕ[$idx,1:end,1:end].*conj.($nf.ϕ[$idx,1:end,1:end])))
   modϕy = @lift(real($nf.ϕ[1:end,$idy,1:end].*conj.($nf.ϕ[1:end,$idy,1:end])))
   modϕz = @lift(real($nf.ϕ[1:end,1:end,$idz].*conj.($nf.ϕ[1:end,1:end,$idz])))
   modϕ = @lift(real($nf.ϕ.*conj.($nf.ϕ)))
   return Plot3D{typeof(f)}(nf,idx, idy, idz, isoϕ, modϕx, modϕy, modϕz, modϕ)
end

function createPlot!(p::AbstractPlot3D)
   # references
   f = p.f.val
   x = f.g.x
   y = f.g.y
   z = f.g.z
   # layout
   fig = Figure(resolution = (600, 600))
   display(fig)
   ax1 = fig[1, 1] = Axis(fig, title = "slice ϕ x")
   ax2 = fig[1, 2] = LScene(fig, scenekw = (camera = cam3d!, raw = false), title = "iso value ϕ")
   ax3 = fig[2, 1] = Axis(fig, title = "slice ϕ y")
   ax4 = fig[2, 2] = Axis(fig, title = "slice ϕ z")
   sl_isoϕ = Slider(fig[3, 1:2], range = 0:0.01:1, startvalue = p.isoϕ[])
   # create plots
   heatmap!(ax1,p.modϕx)
   volume!(ax2,p.modϕ, algorithm = :iso, isorange = sl_isoϕ.value, isovalue = sl_isoϕ.value)
   update_cam!(ax2.scene)
   heatmap!(ax3,p.modϕy)
   heatmap!(ax4,p.modϕz)
   return fig, ax1, ax2, ax3, ax4
end

function updatePlot!(p::AbstractPlot3D)
   p.f[] = p.f.val
end

end
