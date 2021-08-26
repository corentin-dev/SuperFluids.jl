function InitThomasFermi(f :: AbstractField2D, β :: Real; γx :: Real = 1, γy :: Real = 1)
   ρ0 = √( 4* β * √(γx * γy) / π )
   function TF(x,y)
      if x^2 + y^2 > ρ0
         return 0
      else
         return ρ0 .- (x^2 + y^2 )
      end
   end
   return TF
end

function InitThomasFermi(f :: AbstractField3D, β :: Real; γx :: Real = 1, γy :: Real = 1, γz :: Real = 1)
   ρ0 = ( 30 * β * √(γx * γy * γz) / ( 8 * π ) )^(2/5)
   function TF(x,y)
      if x^2 + y^2 + z^2 > ρ0
         return 0
      else
         return ρ0 .- (x^2 + y^2 + z^2)
      end
   end
   return TF
end
