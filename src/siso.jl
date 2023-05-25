function control_form(siso)
    A, B, C = siso.A, siso.B, siso.C
    n = size(A, 1)
    tf_siso = tf(siso)
    α = tf_siso.matrix[1].den.coeffs
    
    P = zeros(n, n)
    P[:, n] = B
    for i in n - 1:-1:1
        p = P[:, i + 1]
        P[:, i] = A * p + α[i + 1] * B
    end

    similarity_transform(siso, P), P
end


function relative_degree(siso)
    A, B, C = siso.A, siso.B, siso.C
    n = size(A, 1)
    for r in 1:n
        u_to_dy = C * A^(r - 1) * B
        if (!isapprox(u_to_dy[1], 0, atol=1e-10))
            return r, u_to_dy[1]
        end
    end
    
end


function null_form(siso)
    siso, T = control_form(siso)
    A, B, C = siso.A, siso.B, siso.C
    n = size(A, 1)
    r, R = relative_degree(siso)
    P = diagm(0 => ones(n))
    
    C = @. C / R
    
    for i in 1:n - r
        P[n - r + 1:end, :] += diagm(r, n, i - 1 => fill(C[i], r))
    end
    P = inv(P)
    siso = similarity_transform(siso, P)
    siso, P, r
end


function null_split(siso)
    null_siso, _, r = null_form(siso)
    A, B, C = null_siso.A, null_siso.B, null_siso.C
    n = size(A, 1)
    
    T = diagm(r, n, n - r => fill(1, r))
    pT = pinv(T)
    # N = diagm(n - r, n, 0 => fill(1, n - r))
    # pN = pinv(N)
    
    yA = T * A * pT
    yB = T * B
    yC = C * pT
    y_siso = ss(yA, yB, yC, 0)
    if r != n
        nA = A[1:n - r, 1:n - r]
        nB = zeros(n - r, 1)
        nB[end] = 1 / yC[1]
        nC = A[n:n, 1:n - r]
        
        n_siso = ss(nA, nB, nC, 0)
    else
        n_siso = ss(0)
    end
    
    y_siso, n_siso
end