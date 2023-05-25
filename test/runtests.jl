using DistributedControlSystems
using Test
using LinearAlgebra
using ControlSystems
using RobustAndOptimalControl

@testset "StateSpaceDistributed" begin
    A = [
        0 1
        0 1
    ]
    Bs = [
        [1, 2],
        [2, 3]
    ]
    Cs = [
        [1 2],
        [2 3]
    ]
    ssd = StateSpaceDistributed(A, Bs, Cs)

    @test ssd.A == A
    @test all(ssd.B .== hcat(Bs...))
    @test all(ssd.C .== vcat(Cs...))
    @test all(ssd.D .== 0)
    @test size(ssd.D) == (size(ssd.B, 2), size(ssd.C, 1))

    Ds = [
        ones(Int, 1, 1), ones(Int, 1, 1),
        ones(Int, 1, 1), ones(Int, 1, 1)
    ]
    Ds = reshape(Ds, 2, 2)
    ssd = StateSpaceDistributed(A, Bs, Cs, Ds)
end


@testset "modes" begin
    A = [
        1 0
        0 2
    ]
    B = [
        1
        0
    ]
    C = [1 0]

    @test fixed_modes(A, B, C) ≈ [2]

    ssd = StateSpaceDistributed(A, [B], [C])
    @test fixed_modes(ssd) ≈ [2]

    # Decentralized Control of Large-Scale Systems.pdf, page 27
    A = [
        -1 0 -3
        0 0.1 0
        0 0 -3
    ]
    B1 = [1., 0, 1]
    B2 = [0., 1, 1]
    Bs = [B1, B2]
    C1 = [0. 1 0]
    C2 = [-1.1 0 0.1]
    Cs = [C1, C2]
    ssd = StateSpaceDistributed(A, Bs, Cs)
    @test fixed_modes(ssd) ≈ [0.1]

    A = [
        0 1 0 0
        0 0 0 0
        0 0 0 1
        0 0 0 0
    ]
    Bs = [[0, 1, 0, 0], [0, 0, 0, 1]]
    Cs = [[1 0 0 0], [0 0 1 0]]
    ssd = StateSpaceDistributed(A, Bs, Cs)
    @test fixed_modes(ssd) == []
end


@testset "distributed_controllers" begin
    T = [
        1 2 3
        -1 2 -1
        3 2 1
    ]
    A = T * Float64[
        0 1 0
        1 -1 1
        0 0 0
    ] * inv(T)
    B1 = T * Float64[
        2 0
        0 1
        0 0
    ]
    B2 = T * Float64[0, 0, 1]
    Bs = [B1, B2]
    C1 = [1. 0 0] * inv(T)
    C2 = [0. 0 1] * inv(T)
    Cs = [C1, C2]
    ssd = StateSpaceDistributed(A, Bs, Cs)

    ssd2 = StateSpaceDistributed(
        ssd.A,
        reverse(ssd.B, dims=1),
        reverse(ssd.C, dims=2),
        reverse(ssd.D, dims=(1, 2)),
    )

    controllers, A_closed_loop = distributed_controllers(ssd)
    controllers2, A_closed_loop2 = distributed_controllers(ssd2)

    @test length(controllers) == 2
    @test length(controllers2) == 2
    @test size(controllers[1].A) == (2, 2)
    @test size(controllers[2].A) == (1, 1)
    @test size(controllers2[1].A) == (1, 1)
    @test size(controllers2[2].A) == (2, 2)
    @test all(e -> e < 0, A_closed_loop |> eigvals |> real)
    @test all(e -> e < 0, A_closed_loop2 |> eigvals |> real)
end


@testset "lmi siso" begin
    a = [1;;]
    b = [1;;]
    f, gamma = lmi_h_inf(a, b)
    @test f[1] > 0

    a = [
        0 1
        0 0
    ]
    b = [
        0; 1;;
    ]
    f, gamma = lmi_h_inf(a, b)
    @test all(f .> 0)

    a = diagm(10, 10, 1 => ones(9))
    b = zeros(10, 1)
    b[end] = 1
    f, gamma = lmi_h_inf(a, b)
    @test isstable(ss(a - b * f))
end

@testset "lmi mimo" begin
    a = diagm(6, 6, 1 => ones(5))
    a[3, 4] = 0
    b = [
        0 0
        0 0
        1 0
        0 0
        0 0
        0 1
    ]
    f, gamma = lmi_h_inf(a, b)
    @test isstable(ss(a - b * f))

    a = [
        0 1 0 0
        0 0 0 0
        0 0 0 1
        0 0 0 0
    ]
    b = [
        0 0
        0 1
        0 0
        1 0
    ]
    T = [
        1 0 -1 0
        0 1 0 -1
    ]
    # z = T x => A = TAT⁺
    A = T * a * pinv(T)
    B = T * b
    f, gamma = lmi_h_inf(A, B)
    @test isstable(ss(A - B * f))
end


@testset "lmi static controller" begin
    a = [
        0 1
        1 -1
    ]
    b = [
        1
        1
    ]
    c = [1 1]
    system = ss(a, b, c, 0)
    ess = ExtendedStateSpace(system)
    f, gamma = lmi_static_h_inf(ess)
    closed = feedback(ess, ss(f))
    @test isstable(closed)
    @test isapprox(hinfnorm2(closed)[1], gamma, atol=1e-2)
    @test size(f) == (1, 1)
end