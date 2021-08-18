"""
    AbstractInit

Abstract supertype for field initialization classes.
"""
abstract type AbstractInit{F} end

export InitNone, InitThomasFermi, InitGauss, InitExternalVelocity

include("thomas-fermi.jl")
include("gauss.jl")
include("external-velocity.jl")

function Init(f::F, ninit::Integer,conf::AbstractConfig) where {F <: AbstractField2D}
   init = nothing
   if ninit == 0
      init = InitNone{typeof(f)}
   elseif ninit == 1
      init = InitThomasFermi2D{typeof(f)}(f,
                       retrieve(conf, "model", "beta", Float64),
                       retrieve(conf, "potential", "gamma_x", Float64),
                       retrieve(conf, "potential", "gamma_y", Float64))
   elseif ninit == 2
      init = InitGauss2D{typeof(f)}(f,
                       retrieve(conf, "potential", "gamma_x", Float64),
                       retrieve(conf, "potential", "gamma_y", Float64),
                       retrieve(conf, "model", "omega", Float64))
   elseif ninit == 6
      coeffΔ = retrieve(conf, "model", "delta", Float64)
      β = retrieve(conf, "model", "beta", Float64)
      ξ = √( -coeffΔ / β)
      init = InitExternalVelocity{typeof(f)}(f,ξ,coeffΔ,"taylorgreen")
   end
   return init
end

function Init(f::F, ninit::Integer,conf::AbstractConfig) where {F <: AbstractField3D}
   init = nothing
   if ninit == 0
      init = InitNone{typeof(f)}
   elseif ninit == 1
      init = InitThomasFermi3D{typeof(f)}(f,
                       retrieve(conf, "model", "beta", Float64),
                       retrieve(conf, "potential", "gamma_x", Float64),
                       retrieve(conf, "potential", "gamma_y", Float64),
                       retrieve(conf, "potential", "gamma_z", Float64))
   elseif ninit == 2
      init = InitGauss3D{typeof(f)}(f,
                       retrieve(conf, "potential", "gamma_x", Float64),
                       retrieve(conf, "potential", "gamma_y", Float64),
                       retrieve(conf, "potential", "gamma_z", Float64),
                       retrieve(conf, "model", "omega", Float64))
   elseif ninit == 6
      coeffΔ = retrieve(conf, "model", "delta", Float64)
      β = retrieve(conf, "model", "beta", Float64)
      ξ = √( -coeffΔ / β)
      init = InitExternalVelocity{typeof(f)}(f,ξ,coeffΔ,"taylorgreen")
   end
   return init
end

struct InitNone{F} <: AbstractInit{F} end

Base.show(io::IO, init::InitNone) = print(io, "No Init")
