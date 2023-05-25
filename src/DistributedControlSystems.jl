module DistributedControlSystems

using LinearAlgebra
using ControlSystems
using BlockArrays
using JuMP
using SCS
using RobustAndOptimalControl

include("general_linear.jl")

export StateSpaceDistributed
export fixed_modes, modes
export distributed_controllers


include("siso.jl")
export control_form
export null_form
export null_split
export relative_degree

include("lmi.jl")
export lmi_h_inf
export lmi_static_h_inf

end
