"""
    Utils

Miscellaneous functions for internal use.
"""
module Utils

using LinearAlgebra: Symmetric, I, cholesky, issuccess

using SpecialFunctions: erfcinv

"""
    norminvcdf(x)

Compute the inverse cumulative distribution function of the standard normal.

This function is often denoted ``\\Phi^{-1}(\\cdot)``.

See
<https://en.wikipedia.org/wiki/Error_function#Cumulative_distribution_function>
for more information.
"""
norminvcdf(x) = -sqrt(2) * erfcinv(2 * x)

"""
    log2prod(x)

Compute `log2(prod(x))` for a vector `x` efficiently without under/overflow.
"""
function log2prod(x)
    mantissa = 1.0
    power = 0
    for i in eachindex(x)
        power += exponent(x[i])
        mantissa *= significand(x[i])
        # prevent under/overflow by periodic normalization
        if (i >> 9) << 9 == i # mod(i, 2^9) == 0
            power += exponent(mantissa)
            mantissa = significand(mantissa)
        end
    end
    return power + log2(mantissa)
end

"""
    chol(M; noise=sqrt(eps(eltype(M))), skip=10.0, report_noise=false)

Cholesky factorization of the matrix `M`, resetting on failure.
"""
function chol(M; noise=sqrt(eps(eltype(M))), skip=10.0, report_noise=false)
    starting_noise = noise
    factor = cholesky(Symmetric(M); check=false)
    while !issuccess(factor)
        factor = cholesky(Symmetric(M + noise * I); check=false)
        noise *= skip
    end
    noise = noise > starting_noise ? noise / skip : 0.0
    return report_noise ? (factor, noise) : factor
end

end
