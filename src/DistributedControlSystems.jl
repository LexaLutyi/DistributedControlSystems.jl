module DistributedControlSystems

using LinearAlgebra
using ControlSystems
using BlockArrays

include("general_linear.jl")

export StateSpaceDistributed
export fixed_modes, modes
export distributed_controllers

end
