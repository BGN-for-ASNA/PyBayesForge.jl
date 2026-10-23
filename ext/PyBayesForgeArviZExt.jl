module PyBayesForgeArviZExt

using PyBayesForge
using PythonCall
using ArviZ

_julia(x) = pyconvert(Array{Float64}, pyimport("numpy").asarray(x, dtype = "float64"))
# (chains, draws, sizes...) from Python -> (draws, chains, sizes...) for ArviZ
_draws_first(a) = permutedims(a, (2, 1, 3:ndims(a)...))

"""
    convert_to_inference_data(m::PyBayesForge.BFObject) -> InferenceData

A fitted BayesForge model as an ArviZ.jl `InferenceData`, with groups
`posterior`, `log_likelihood` (pointwise, per observed site, from NumPyro) and
`observed_data`, so that `summarize`, `loo`, `waic` and the ArviZ plots work.
"""
function ArviZ.convert_to_inference_data(m::PyBayesForge.BFObject)
    bf = m.py
    samples = bf.sampler.get_samples(group_by_chain = true)
    post = Dict{Symbol,Any}()
    for k in samples.keys()
        post[Symbol(pyconvert(String, k))] = _draws_first(_julia(samples[k]))
    end
    data = bf.data_on_model
    ll = pyimport("numpyro.infer").log_likelihood(bf.model, samples; batch_ndims = 2, data...)
    loglik = Dict{Symbol,Any}()
    for k in ll.keys()
        loglik[Symbol(pyconvert(String, k))] = _draws_first(_julia(ll[k]))
    end
    observed = Dict{Symbol,Any}()
    for (site, key) in bf._observed_sites(data).items()
        pyis(key, pybuiltins.None) && continue
        observed[Symbol(pyconvert(String, site))] = _julia(data[key])
    end
    return from_namedtuple((; post...);
                           log_likelihood = (; loglik...),
                           observed_data = (; observed...),
                           library = "BayesForge")
end

end
