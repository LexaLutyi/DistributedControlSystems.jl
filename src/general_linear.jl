"""
    StateSpaceDistributed

Distributed control system stored as state space.  
ẋ = A x + ∑ Bᵢ uᵢ  
yᵢ = Cᵢ x + ∑ Dᵢⱼ uⱼ  

Block matrices are used to store data  
B = [B₁ B₂ … Bₙ]  
C = [C₁; C₂; …; Cₙ]
"""
struct StateSpaceDistributed{BM_A, BM_B, BM_C, BM_D}
    A::BM_A
    B::BM_B
    C::BM_C
    D::BM_D
end


"""
B = hcat(Bs...), C = vcat(Cs...)
"""
function StateSpaceDistributed(A, Bs, Cs)
    m, n = size(A)
    @assert m == n "Matrix A should be square"

    input_sizes = size.(Bs, 2)
    output_sizes = size.(Cs, 1)
    len_b = length(Bs)
    len_c = length(Cs)
    @assert len_b == len_c "length(Bs) != length(Cs)"
    @assert all(size.(Bs, 1) .== m) "All B matrix should have size(A, 1) rows"
    @assert all(size.(Cs, 2) .== n) "All C matrix should have size(A, 2) columns"    
    
    B = BlockMatrix(hcat(Bs...), [size(A, 1)], input_sizes)
    C = BlockMatrix(vcat(Cs...), output_sizes, [size(A, 2)])
    D = similar(A, size(C, 1), size(B, 2)) .= 0
    D = BlockMatrix(D, output_sizes, input_sizes)

    StateSpaceDistributed(A, B, C, D)
end


"""
    robust_intersect(a, b)
return intersection of a and b
"""
function robust_intersect(a, b)
    ix = findall(x -> (x ∈ b), a)
    a[ix]
end


"""
    fixed_modes(A, B, C; digits = 10)
return centralized fixed_modes (CFM) of system (A, B, C)

digits have the same meaning as for round
"""
function fixed_modes(A, B, C; digits = 10)
    m = size(B, 2)
    n = size(C, 1)
    mapreduce(robust_intersect, 1:size(A, 1)) do _
        K = randn(m, n)
        e = eigvals(A + B * K * C) .|> ComplexF64
        round.(e; digits)
    end
end


"""
    modes(ssd::StateSpaceDistributed; digits = 10)

return local centralized fixed modes (CFM) for each agent

digits have the same meaning as for round
"""
function modes(ssd::StateSpaceDistributed; digits = 10)
    A = ssd.A
    map(blockaxes(ssd.B, 2), blockaxes(ssd.C, 1)) do bj, ci
        B = ssd.B[bj]
        C = ssd.C[ci]
        fixed_modes(A, B, C; digits)
    end
end

# A * D, where D is a block diagonal matrix
# D is stored as vector of matrices
function Base.:*(a::AbstractBlockMatrix, block_diag::AbstractVector{<:AbstractArray})
    mapreduce(vcat, blockaxes(a, 1)) do bi
        mapreduce(hcat, blockaxes(a, 2), block_diag) do bj, d
            a[bi, bj] * d
        end
    end
end

# D * A, where D is a block diagonal matrix
# D is stored as vector of matrices
function Base.:*(block_diag::AbstractVector{<:AbstractArray}, a::AbstractBlockMatrix)
    mapreduce(vcat, blockaxes(a, 1), block_diag) do bi, d
        mapreduce(hcat, blockaxes(a, 2)) do bj
            d * a[bi, bj]
        end
    end
end


"""
    fixed_modes(ssd::StateSpaceDistributed; digits = 10)

return decentralized fixed modes (DFM)
"""
function fixed_modes(ssd::StateSpaceDistributed; digits = 10)
    mapreduce(robust_intersect, 1:size(ssd.A, 1)) do i
        A = ssd.A
        B = ssd.B
        C = ssd.C
        D = ssd.D

        Ks = map(blocksizes(B)[2], blocksizes(C)[1]) do m, n
            K = randn(m, n)
        end
        Acl = A + B * Ks * inv(I - D * Ks) * C
        e = eigvals(Acl) .|> ComplexF64
        round.(e; digits)
    end
end


"""
    robust_controller(sys::StateSpace)

return standard controller with full-rank observer  
feedback matrices are computed via kalman and lqr approaches
"""
function robust_controller(sys::StateSpace)
    reduced_system = sys |> minreal |> ss
    R = kalman(reduced_system, I, I)
    Q = lqr(reduced_system, I, I)
    observer_controller(reduced_system, Q, R)
end


"""
    local_controller_and_closed_sys(ssd, k, Ak)

perform k-th step of decentralized controller synthesis 

Ak is a closed loop matrix obtained on previous step
"""
function local_controller_and_closed_sys(ssd, k, Ak;
    alg=robust_controller
)
    B = ssd.B[Block(k)]
    m = size(Ak, 1) - size(B, 1)
    Bk = [
        B
        zeros(m, size(B, 2))
    ]
    C = ssd.C[Block(k)]
    n = size(Ak, 2) - size(C, 2)
    Ck = [C zeros(size(C, 1), n)]
    
    sys = ss(Ak, Bk, Ck, 0)
    local_controller = alg(sys)
    closed_sys = feedback(sys, local_controller)
    local_controller, closed_sys
end


"""
    controllers, Acl = distributed_controllers(ssd)

return vector of local controllers and final closed loop matrix A
"""
function distributed_controllers(ssd; alg=robust_controller)
    n = blocksize(ssd.B, 2)
    controller, sys = local_controller_and_closed_sys(ssd, 1, ssd.A; alg)
    Ak = sys.A
    controllers = [controller]
    
    for k in 2:n
        controller, sys = local_controller_and_closed_sys(ssd, k, Ak; alg)
        Ak = sys.A
        push!(controllers, controller)
    end
    controllers, Ak
end