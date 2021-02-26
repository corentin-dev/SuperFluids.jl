using AbstractFFTs
using FFTW
using LinearAlgebra: mul!, ldiv!

"""
    AbstractNumModel

Abstract supertype for field initialization classes.
"""
abstract type AbstractNumModel{F,P} end
abstract type AbstractPlan{F} end

struct Plan2D{F} <: AbstractPlan{F}
   plan :: AbstractFFTs.Plan
   plan_x :: AbstractFFTs.Plan
   plan_y :: AbstractFFTs.Plan
end

struct Plan3D{F} <: AbstractPlan{F}
   plan :: AbstractFFTs.Plan
   plan_x :: AbstractFFTs.Plan
   plan_y :: AbstractFFTs.Plan
   plan_z :: AbstractFFTs.Plan
end

function Plan(f::F) where {F<:AbstractField2D}
   return Plan2D{F}(plan_fft(f.ϕ),
                    plan_fft(f.ϕ,1),
                    plan_fft(f.ϕ,2))
end

function Plan(f::F) where {F<:AbstractField3D}
   return Plan3D{F}(plan_fft(f.ϕ),
                    plan_fft(f.ϕ,1),
                    plan_fft(f.ϕ,2),
                    plan_fft(f.ϕ,3))
end

struct NumModelADI2{F,P} <: AbstractNumModel{F,P}
   f :: F
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   coeffΔ :: Real
   β :: Real
   Ω :: Real
   plan :: AbstractPlan{F}
   ϕ_hat :: Array
   potential :: P
end

function NumModel(f::F, p::P, conf::ConfParse) where F where P
   nummodel = nothing
   nmodel = parse(Int64,retrieve(conf, "solver", "model"))
   plan = Plan(f)
   ϕ_hat = similar(f.ϕ)
   Δt =  parse(Float64,retrieve(conf, "time", "deltat"))
   niter = parse(Float64,retrieve(conf, "time", "itermax"))
   restart = ( parse(Int64,retrieve(conf, "time", "restart")) == 1 )
   freqbckp = parse(Int64,retrieve(conf, "io", "freqbckp"))
   coeffΔ = parse(Float64,retrieve(conf, "model", "delta"))
   β = parse(Float64,retrieve(conf, "model", "beta"))
   Ω = parse(Float64,retrieve(conf, "model", "omega"))
   if nmodel == 2
      nummodel = NumModelADI2{typeof(f),typeof(p)}(f,
                       Δt, niter, freqbckp, coeffΔ, β, Ω, plan, ϕ_hat, p
                      )
   elseif nmodel == 42
      nummodel = NumModelADI2{typeof(f),typeof(p)}(f,
                       Δt, niter, freqbckp, coeffΔ, β, Ω, plan, ϕ_hat, p
                      )
      # _ => "something else"
   end
   return nummodel
end

function timeStep!(n::NumModelADI2)
   solveLapRot!(n,n.Δt*0.5)
   solveNL!(n,n.Δt)
   solveLapRot!(n,n.Δt*0.5)
end

function solveLapRot!(n::NumModelADI2{F}, Δtl) where {F<:AbstractField2D}
   # references
   ϕ, ϕhat_x, ϕhat_y = n.f.ϕ, n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω = n.coeffΔ, n.Ω
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy = n.f.g.ξx, n.f.g.ξy
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   # perform FFT
   mul!(ϕhat_x, plan_x, ϕ)
   # compute the laplacian and rotation in the Fourier space (x)
   @. ϕhat_x = exp(im*(coeffΔ*ξx^2-Ω*y*ξx)*Δtl) * ϕhat_x
   # backward FFT
   ldiv!(ϕ, plan_x, ϕhat_x)
   # perform FFT
   mul!(ϕhat_y, plan_y, ϕ)
   # compute the laplacian and rotation in the Fourier space (y)
   @. ϕhat_y = exp(im*(coeffΔ*ξy^2+Ω*x*ξy)*Δtl) * ϕhat_y
   # backward FFT
   ldiv!(ϕ, plan_y, ϕhat_y)
   return nothing
end

function solveLapRot!(n::NumModelADI2{F}, Δtl) where {F<:AbstractField3D}
   # references
   ϕ, ϕhat_x, ϕhat_y, ϕhat_z = n.f.ϕ, n.ϕ_hat, n.ϕ_hat, n.ϕ_hat
   coeffΔ, Ω = n.coeffΔ, n.Ω
   x, y, z = n.f.g.x, n.f.g.y, n.f.g.z
   ξx, ξy, ξz = n.f.g.ξx, n.f.g.ξy, n.f.g.ξz
   plan_x, plan_y, plan_z = n.plan.plan_x, n.plan.plan_y, n.plan.plan_z
   # perform FFT
   mul!(ϕhat_x, plan_x, ϕ)
   # compute the laplacian and rotation in the Fourier space (x)
   @. ϕhat_x = exp(im*(coeffΔ*ξx^2-Ω*y*ξx)*Δtl) * ϕhat_x
   # backward FFT
   ldiv!(ϕ, plan_x, ϕhat_x)
   # perform FFT
   mul!(ϕhat_y, plan_y, ϕ)
   # compute the laplacian and rotation in the Fourier space (y)
   @. ϕhat_y = exp(im*(coeffΔ*ξy^2+Ω*x*ξy)*Δtl) * ϕhat_y
   # backward FFT
   ldiv!(ϕ, plan_y, ϕhat_y)
   # perform FFT
   mul!(ϕhat_z, plan_z, ϕ)
   # compute the laplacian and rotation in the Fourier space (y)
   @. ϕhat_z = exp(im*(coeffΔ*ξz^2)*Δtl) * ϕhat_z
   # backward FFT
   ldiv!(ϕ, plan_z, ϕhat_z)
   return nothing
end

function solveNL!(n::NumModelADI2, Δtl)
   # references
   ϕ = n.f.ϕ
   β = n.β
   V = n.potential.V
   # computation
   @. ϕ = exp(-1im * ( V + β*ϕ*conj(ϕ) ) * Δtl) * ϕ
end

function solve!(n::AbstractNumModel)
   for it = 1:n.niter
      println("iteration $(it)")
      timeStep!(n)
   end
end

Base.show(io::IO, n::NumModelADI2{F}) where {F<:AbstractField3D} = print(io,
         "Splitting Order 2",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
