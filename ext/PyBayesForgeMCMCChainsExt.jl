module PyBayesForgeMCMCChainsExt

using PyBayesForge
using MCMCChains

"""
    Chains(m::PyBayesForge.BFObject)

Posterior draws of a fitted BayesForge model as an `MCMCChains.Chains`
(iterations × parameters × chains), for use with MCMCChains / ArviZ.jl tools.
"""
function MCMCChains.Chains(m::PyBayesForge.BFObject)
    t = bf_draws(m)
    params = [k for k in keys(t) if !(k in (:chain, :iteration, :draw))]
    nchains, niter = maximum(t.chain), maximum(t.iteration)
    vals = Array{Float64}(undef, niter, length(params), nchains)
    for (j, p) in enumerate(params), c in 1:nchains
        vals[:, j, c] = getproperty(t, p)[t.chain .== c]
    end
    return Chains(vals, params)
end

end
