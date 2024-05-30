using LinearAlgebra
using DataFrames
using JLD2

d = load("matrices.jld2")

A = d["A"]
B = d["B"]
eq1 = d["eq1"]
eq2 = d["eq2"]

# s0 = 4.6e6
s0 = parse(Float64, ARGS[1])
A = A - s0 * B

n1 = length(eq1)
n2 = length(eq2)
n = n1 + n2

Ψ0 = factorize(A[eq2, eq2]) \ -A[eq2, eq1]
ΨL0 = factorize(A[eq2, eq2])' \ -A[eq1, eq2]'

T0 = zeros(Float64, n, n1)
T0[eq1, :] += 1.0 * I
TL0 = copy(T0)
T00 = copy(T0)
T0[eq2, :] = Ψ0
TL0[eq2, :] = ΨL0

doOrtho = parse(Bool, ARGS[3])
if doOrtho
  using Substructuring
  CGS!(T0)
  CGS!(T0)
  CGS!(TL0)
  CGS!(TL0)
end

Tpr0 = A * T0
Tpe0 = factorize(A)' \ T0

# "unsym Guyan" / alternative realization
Ar = TL0' * A * T0
Br = TL0' * B * T0

# Galerkin
Ag = T0' * A * T0
Bg = T0' * B * T0

# left inverse
A0 = T00' * A * T0
B0 = T00' * B * T0

# Petrov-Galerkin
# min residual
Apr = Tpr0' * A * T0
Bpr = Tpr0' * B * T0
# min residual
Ape = Tpe0' * A * T0
Bpe = Tpe0' * B * T0

# modal
ev = hcat(eigvals(Ar, Br), eigvals(Ag, Bg), eigvals(A0, B0), eigvals(Apr, Bpr), eigvals(Ape, Bpe))

# harmonic
b = rand(Float64, n)
br = TL0' * b
bg = T0' * b
b0 = T00' * b
bpr = Tpr0' * b
bpe = Tpe0' * b

# s = 1e6
s = parse(Float64, ARGS[2])

x = factorize(A - s * B) \ b
xr = factorize(Ar - s * Br) \ br
xg = factorize(Ag - s * Bg) \ bg
x0 = factorize(A0 - s * B0) \ b0
xpr = factorize(Apr - s * Bpr) \ bpr
xpe = factorize(Ape - s * Bpe) \ bpe

sol = hcat(xr, xg, x0, xpr, xpe)

# error
err = hcat(
  T0' * T0 * xr - T0' * x,
  T0' * T0 * xg - T0' * x,
  T0' * T0 * x0 - T0' * x,
  T0' * T0 * xpr - T0' * x,
  T0' * T0 * xpe - T0' * x,
)

err_rel_norm = [norm(err_i) / norm(T0' * x) for err_i in eachcol(err)]

# residual
res = hcat(
  TL0' * (A - s * B) * T0 * xr - TL0' * b,
  T0' * (A - s * B) * T0 * xg - T0' * b,
  T00' * (A - s * B) * T0 * x0 - T00' * b,
  Tpr0' * (A - s * B) * T0 * xpr - Tpr0' * b,
  Tpe0' * (A - s * B) * T0 * xpe - Tpe0' * b,
)

res_rel_norm = [norm(res_i) for res_i in eachcol(res)]
res_rel_norm[1] /= norm(TL0' * b)
res_rel_norm[2] /= norm(T0' * b)
res_rel_norm[3] /= norm(T00' * b)
res_rel_norm[4] /= norm(Tpr0' * b)
res_rel_norm[5] /= norm(Tpe0' * b)

summary = DataFrame(
  "method" => ["alt realization", "Galerkin", "truncature", "Petrov min res", "Petrov min err"],
  "rel err" => err_rel_norm,
  "rel res" => res_rel_norm,
)


# expanded
# error
errxp = hcat(
  T0 * xr - x,
  T0 * xg - x,
  T0 * x0 - x,
  T0 * xpr - x,
  T0 * xpe - x,
)

errxp_rel_norm = [norm(err_i) / norm(x) for err_i in eachcol(errxp)]

# residual
resxp = hcat(
  (A - s * B) * T0 * xr - b,
  (A - s * B) * T0 * xg - b,
  (A - s * B) * T0 * x0 - b,
  (A - s * B) * T0 * xpr - b,
  (A - s * B) * T0 * xpe - b,
)

resxp_rel_norm = [norm(res_i) / norm(b) for res_i in eachcol(resxp)]

# local residuals
resxp1 = resxp[eq1, :]
resxp2 = resxp[eq2, :]

resxp1_rel_norm = [norm(res_i) / norm(b[eq1]) for res_i in eachcol(resxp1)]
resxp2_rel_norm = [norm(res_i) / norm(b[eq2]) for res_i in eachcol(resxp2)]

summaryxp = DataFrame(
  "method" => ["alt realization", "Galerkin", "truncature", "Petrov min res", "Petrov min err"],
  "rel err" => errxp_rel_norm,
  "rel res" => resxp_rel_norm,
  "master rel res" => resxp1_rel_norm
)


println("====================================================")
println("================== reduced spaces ==================")
println("====================================================")
display(summary)
println("=====================================================")
println("================== original spaces ==================")
println("=====================================================")
display(summaryxp)

