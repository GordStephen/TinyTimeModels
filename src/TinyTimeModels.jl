module TinyTimeModels

using Optim

export fit

include("kalman.jl")
include("parameterestimation.jl")

end # module
