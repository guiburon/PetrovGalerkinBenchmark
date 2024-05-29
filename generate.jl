using LinearAlgebra
using JLD2

@doc """
`condrand(m::Integer, κ::Real; abs_sv_dom::Real=1.0, sym::Bool=true, def=:posdef)`

Generate a square `Matrix{<:Real}` of size `m` with condition number `κ`. By default the matrix is symmetric `sym=true` and positive definite `def=:posdef`. A symmetric matrix can also be negative definite `def=:negdef`. The absolute value of the dominant singular value `abs_sv_dom` defaults to 1.0 but can be specified. 
"""
function condrand(m::Integer, κ::Real; abs_sv_dom::Real=1.0, sym::Bool=true, def=:posdef)
  (m > 0) || throw(ArgumentError("matrix size must be > 0"))
  (def ∈ (:negdef, :posdef)) || throw(ArgumentError("def can only be :negdef or :posdef"))
  (abs_sv_dom > 0.0) || throw(ArgumentError("absolute value of the dominant singular value must be > 0"))
  if !sym && def != :posdef   # literature exists but not implemented here
    throw(ArgumentError("definiteness is not implemented for non-symmetric matrices"))
  end

  j = m - 1
  l = κ^(1 / j) # k-1 root of condition number

  sv = l .^ (0:-1:-j)
  sv *= abs_sv_dom

  if def == :negdef
    sv *= -1.0
  end

  S = Diagonal(sv)

  if sym
    V, = qr(rand(m, m)) # qr is only Float64
    return V' * S * V
  else
    U, = qr(rand(m, m))
    V, = qr(rand(m, m))
    return U * S * V
  end
end

n1 = 20
n2 = 50
n = n1 + n2

# eqs used to build the matrix structure
eq1 = 1:n1
eq2 = n1+1:n

A = zeros(n, n)
A[eq1, eq1] = condrand(n1, 100)
A[eq2, eq2] = condrand(n2, 1000)
A[eq1, eq2] = rand(n1, n2)

B = zeros(n, n)
B[eq1, eq1] = condrand(n1, 1000)
B[eq2, eq2] = condrand(n2, 1000)
B[eq2, eq1] = rand() * A[eq1, eq2]'

A = A - 2 * real(eigvals(A, B)[1]) * B

save("matrices.jld2",
  "A", A,
  "B", B,
  "eq1", eq1,
  "eq2", eq2
)

