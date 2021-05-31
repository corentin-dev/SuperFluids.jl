"Abstract supertype for SuperFluid configuration."
abstract type AbstractConfig end

"Config type containing a dictionary."
mutable struct Config <: AbstractConfig
   conf :: Dict
end

"Constructor using a config file."
function Config(fileName::String)
   conf = ConfParse(fileName)
   parse_conf!(conf)
   return Config(conf._data)
end

"Allows to retrieve a value based on a category and a key."
function retrieve(c::AbstractConfig,category::String,key::String,T::Type)
   vString = try
      parse(T,c.conf[category][key][1])
   catch e
      nothing
   end
end

"Allows to retrieve a value based on a category and a key, providing a default value."
function retrieve(c::AbstractConfig,category::String,key::String,T::Type,default)
   v = retrieve(c,category,key)
   if v == nothing
      return default
   else
      return v
   end
end
