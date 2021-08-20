function InitThomasFermi(f :: AbstractField2D, β :: Real; γx :: Real = 1, γy :: Real = 1)
   return (x,y) -> √(
                     (im*
                      √(β*γx*γy/π) -
                      0.5*((γx*x)^2+(γy*y)^2)
                     )/β
                    )
end

function InitThomasFermi(f :: AbstractField3D, β :: Real; γx :: Real = 1, γy :: Real = 1, γz :: Real = 1)
   return (x,y,z) -> √(im*
                       (
                        0.5 * (15. * β * γx * γy * γz / 4. / π)^(2. / 5.) - # μ
                        0.5 * ( γx*x^2 + γy*y^2 + γz*z^2 )  # V
                       )/β
                      )
end
