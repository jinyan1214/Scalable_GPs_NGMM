

function loss_loo(θ)
    k = kernelc(exp.(θ[1:end-1]))
    K_full = kernelmatrix(k, X_samp', X_samp')

    iₛ = collect(1:n_samp)

    ŷ = [let iₜ = filter(i -> i ≠ iₚ, iₛ)  
            Kₜₜ = @views K_full[iₜ, iₜ]
            kₚₜ = @views K_full[[iₚ], iₜ]
            f(Kₜₜ, kₚₜ, y_samp[iₜ])[1]
        end for iₚ in iₛ]

    # println("loss error: ",norm(y_samp - ŷ))
    # println("loss regularization: ",norm(ŷ) * (sum(exp.(θ[3:4])) + 0.0001 * exp(θ[1])))
    return norm(y_samp - ŷ) + norm(ŷ) * (sum(exp.(θ[3:4])) + 0.0001 * exp(θ[1]))
end