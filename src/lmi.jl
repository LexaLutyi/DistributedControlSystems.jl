function lmi_Phi(A, B, C, W, N)
    NT = permutedims(N)
    W * A' + A * W - B * N * C - C' * NT * B'
end


function lmi_static_h_inf(A, Bw, Bu, Cz, Cy, Dwz, Duz; 
    σ = 0.01, 
    row_block_lens = [size(Bu, 2)], 
    col_block_lens = [size(Cy, 1)]
)
    n0, n = size(A)
    n1, m = size(Bu)
    r, n2 = size(Cy)
    n3, q = size(Bw)
    p, n4 = size(Cz)
    p1, q1 = size(Dwz)
    p2, m1 = size(Duz)

    @assert n == n0 "A must be square, ($n, $n0)"
    @assert n == n1 "A and Bu must have the same number of rows, ($n, $n1)"
    @assert n == n2 "A and Cy must have the same number of columns, ($n, $n2)"
    @assert n == n3 "A and Bw must have the same number of rows, ($n, $n3)"
    @assert n == n4 "A and Cz must have the same number of columns, ($n, $n4)"

    @assert p == p1 "Dwz and Cz must have the same number of rows, ($p1, $p)"
    @assert q == q1 "Dwz and Bw must have the same number of columns, ($q1, $q)"
    @assert p == p2 "Duz and Cz must have the same number of rows, ($p2, $p)"
    @assert m == m1 "Duz and Bu must have the same number of columns, ($m1, $m)"

    n_blocks = length(row_block_lens)
    n_blocks1 = length(col_block_lens)
    @assert n_blocks == n_blocks1 "length(row_block_lens) != length(col_block_lens)"
    @assert sum(row_block_lens) == m "sum(row_block_lens) != m"
    @assert sum(col_block_lens) == r "sum(col_block_lens) != r"

    model = Model(SCS.Optimizer)
    set_silent(model)
    @variable(model, W[1:n, 1:n])
    # @variable(model, N[1:m, 1:r])
    # @variable(model, M[1:r,1:r])
    @variable(model, γ)

    n_blocks = length(row_block_lens)
    row_block_end = cumsum([0; row_block_lens])
    col_block_end = cumsum([0; col_block_lens])
    # for i in 1:n_blocks
    #     row_start = row_block_end[i] + 1
    #     row_end = row_block_end[i + 1]
    #     col_start = col_block_end[i] + 1
    #     col_end = col_block_end[i + 1]
    #     row_ix = map(ix -> row_start <= ix <= row_end, 1:m)
    #     col_ix = map(ix -> col_start <= ix <= col_end, 1:r)
    #     @constraint(model, N[row_ix, .!col_ix] .== 0)
    #     @constraint(model, M[col_ix, .!col_ix] .== 0)
    # end

    N = zeros(AffExpr, m, r)
    M = zeros(AffExpr, r, r)
    foreach(1:n_blocks, row_block_lens, col_block_lens) do i, m, n
        Nb = @variable(model, [1:m, 1:n])
        Mb = @variable(model, [1:n, 1:n])
        row_start = row_block_end[i] + 1
        row_end = row_block_end[i + 1]
        col_start = col_block_end[i] + 1
        col_end = col_block_end[i + 1]
        N[row_start:row_end, col_start:col_end] = Nb
        M[col_start:col_end, col_start:col_end] = Mb
    end

    NT = permutedims(N)
    lmi_13 = W * Cz' - Cy' * NT * Duz'
    lmi_31 = permutedims(lmi_13)
    lmi_main = lmi_Phi(A, Bu, Cy, W, N)
    lmi = [
        lmi_main Bw lmi_13
        Bw' -γ * I(q) Dwz'
        lmi_31 Dwz -I(p)
    ]
    @constraint(model, W - σ * I ∈ PSDCone())
    @constraint(model, M * Cy .== Cy * W)
    @constraint(model, γ >= 0)
    @constraint(model, -lmi ∈ PSDCone())

    @objective(model, Min, γ)

    optimize!(model)
    w = value.(W)
    m = value.(M)
    n = value.(N)
    gamma = value.(γ)
    F = n * inv(m)

    F, gamma
end

lmi_static_h_inf(e::ExtendedStateSpace; 
        σ = 0.01, 
        row_block_lens = [size(e.B2, 2)], 
        col_block_lens = [size(e.C2, 1)]
    ) = lmi_static_h_inf(
        e.A, e.B1, e.B2, e.C1, e.C2, e.D11, e.D12; 
        σ, row_block_lens, col_block_lens
    )

function lmi_h_inf(A, B, C = I, Bw = B)
    n, m = size(B)
    C = C == I ? I(n) : C
    r, n = size(C)

    sys = ss(A, B, C, 0)
    # z = z1 + z2 = C x + D12 u
    C1  = [
        C           # output signal
        zeros(m, n) # control signal
    ]
    C2 = I(n) # controller use full state
    D12 = zeros(r + m, m)
    D12[CartesianIndex.(r+1:r+m, 1:m)] .= 1 # D12 u = u
    ess = ExtendedStateSpace(sys; C1, C2, B1 = Bw, D12)
    lmi_static_h_inf(ess)
end