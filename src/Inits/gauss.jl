function InitGauss(f :: AbstractField2D; Ω :: Real = 0)
   return (x,y) -> (1-Ω) * (1/sqrt(π)*exp(-0.5*(x^2+y^2))) + Ω * (1*(x+im*y)/sqrt(π)*exp(-0.5*(x^2+y^2)))
end

function InitGauss(f :: AbstractField3D; Ω :: Real = 0)
   return (x,y,z) -> (1-Ω) * exp(-0.5*(x^2+y^2+γz*z^2))*(γz^0.25/π^0.75) + Ω * (x+im*y+0*z)*exp(-0.5*(x^2+y^2+γz*z^2))*(γz^0.25/π^0.75)
end
