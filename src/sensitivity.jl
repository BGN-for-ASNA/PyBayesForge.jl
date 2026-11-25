# -------------------------------------------------------
# Prior sensitivity: m.diag.prior_sensitivity
# -------------------------------------------------------
#
# When the first argument is a Julia function, the fits must run in Julia
# workers (a Python process pool cannot pickle a Julia function), so the name
# `prior_sensitivity` is answered in Julia for diag objects (Base.getproperty,
# src/PyBayesForge.jl) and the fits go through bf_run_simulations
# (src/simulate.jl). Everything else -- the scenario list, the summary table
# and the figures -- reuses BayesForge.Diagnostic.sensitivity, so a Julia model
# and a Python one produce the same table and the same plots.

_sens_py() = pyimport("BayesForge.Diagnostic.sensitivity")

_jstr(s) = s isa Union{AbstractString,Symbol} ? String(s) : pyconvert(String, s)
_pynone() = pybuiltins.None
_pynone(::Nothing) = pybuiltins.None
_pynone(x) = x

# Is x a BayesForge diag object (m.diag)? Both classes are accepted:
# Main.main builds instances of Diag2.diagWIP, Diag.diag is the legacy one.
function _isdiag(p::Py)
    for (mod, name) in (("BayesForge.Diagnostic.Diag2", "diagWIP"),
                        ("BayesForge.Diagnostic.Diag", "diag"))
        cls = try
            getproperty(pyimport(mod), name)
        catch
            nothing
        end
        if cls !== nothing && pyisinstance(p, cls)
            return true
        end
    end
    return false
end
_isdiag(x::BFObject) = _isdiag(x.py)
_isdiag(x) = false

# Python creates `m.diag` inside `m.fit`, but prior sensitivity does not need a
# posterior from `m`: it fits its own models. So `m.diag` on an unfitted model
# allocates the real diag class without running __init__ (which wants a fitted
# sampler); the only method that works on it is prior_sensitivity, which this
# package answers in Julia anyway.
function _new_diag()
    cls = getproperty(pyimport("BayesForge.Diagnostic.Diag2"), :diagWIP)
    return cls.__new__(cls)
end

# -------------------------------------------------------
# Julia <-> Python conversions for this entry point
# -------------------------------------------------------

_pyprior_value(v::AbstractDict) = _pydict_pairs(v)
_pyprior_value(v::Py) = v
_pyprior_value(v) = _pyval(v)

# (key, value) pairs of a Julia mapping or of a Python dict, in the order the
# caller wrote them (a Julia Dict has no order of its own).
_pairs_any(d::Py) = [(k, d[k]) for k in d.keys()]
_pairs_any(d::BFObject) = _pairs_any(d.py)
_pairs_any(d) = [p for p in pairs(d)]

# An ordered Python dict from any Julia or Python mapping (nothing -> empty).
function _pydict_pairs(d)
    out = pydict()
    d === nothing && return out
    for (k, v) in _pairs_any(d)
        out[_jstr(k)] = _pyprior_value(v)
    end
    return out
end

# {arg: value} -> Dict{Symbol, Any} of model keyword arguments. The worker
# receives them as JSON: a Python dict cannot cross a process boundary.
function _priors_kwargs(p)
    p isa AbstractString && (p = pyimport("json").loads(p))
    out = Dict{Symbol,Any}()
    p === nothing && return out
    for (k, v) in _pairs_any(p)
        out[Symbol(_jstr(k))] = _tojl(v)
    end
    return out
end

# fit_kwargs / priors: any Julia or Python mapping -> Julia with Symbol keys.
function _sym_kwargs(d)
    out = Dict{Symbol,Any}()
    d === nothing && return out
    for (k, v) in _pairs_any(d)
        out[Symbol(_jstr(k))] = v
    end
    return out
end

# obs values must be plain Julia arrays to reach the workers; numpy arrays
# (float32 included) come back as numpy arrays inside the worker.
function _obs_julia(obs)
    out = Dict{String,Any}()
    obs === nothing && return out
    for (k, v) in _pairs_any(obs)
        out[_jstr(k)] = v isa BFObject ? pyconvert(Array, v.py) :
                        v isa PyWrappers ? pyconvert(Array, v) : v
    end
    return out
end

# Dict{param, draws} -> Python dict of numpy arrays, as prior_sensitivity expects.
function _post_py(p::AbstractDict)
    out = pydict()
    for (k, v) in pairs(p)
        out[_jstr(k)] = v
    end
    return out
end
_post_py(p) = _topy(p)

_posts_py(ps::AbstractVector) = pylist([_post_py(p) for p in ps])
_posts_py(ps) = _topy(ps)

_pystr_list(v::AbstractVector) = pylist([x for x in v])
_pystr_list(v) = _topy(v)

# Positional arguments of BayesForge.Diagnostic.sensitivity.prior_sensitivity,
# after model_or_posteriors, so that `m.diag.prior_sensitivity(model, labels,
# priors, ...)` means the same thing in Julia as in Python.
const _PRIOR_SENS_ARGS = [:labels, :priors, :obs, :mode, :baseline, :joint, :fit_kwargs,
                          :workers, :seed, :param_names, :hdi_prob, :title,
                          :x_label, :y_label, :return_results, :plot]

# -------------------------------------------------------
# The fit, run on the Julia workers
# -------------------------------------------------------

"""
    _BoundModel(f, m, names, priors)

A model function `f(m, <data...>; <priors...>)` bound to one bf instance and to
one set of prior hyper-parameters, exposing only the data argument `names` to
Python: BayesForge matches the observed data to the model arguments by name, so
the callable Python sees is `lambda weight, height: fn(weight, height)` -- the
same trick as `InspectableFunction` (src/PyBayesForge.jl).
"""
struct _BoundModel <: Function
    f::Function
    m::BFObject
    names::Vector{String}
    priors::Dict{Symbol,Any}
end

function (b::_BoundModel)(args...; kwargs...)
    new_args = map(a -> a isa PyWrappers ? BFObject(a) : a, args)
    return b.f(b.m, new_args...; b.priors...)
end

function PythonCall.Py(b::_BoundModel)
    isempty(b.names) &&
        error("the model takes no data arguments besides `m`; prior sensitivity " *
              "needs a model of the form model(m, x, y; prior_sd = 1.0).")
    args_str = join(b.names, ", ")
    factory = pybuiltins.eval("lambda fn: lambda $args_str: fn($args_str)",
                              pybuiltins.dict(), pybuiltins.dict())
    wrapper(args...; kwargs...) = b(args...; kwargs...)
    return factory(wrapper)
end

# Data argument names of `model(m, weight, height; ...)`, i.e. every positional
# argument after `m`.
_model_data_names(f) = String.(Base.method_argnames(first(methods(f)))[3:end])

"""
    _prior_sens_fit(; label, varied, priors, seed, model, obs, fit_kwargs, kw...)

One prior-sensitivity fit, run by `bf_run_simulations` on a Julia worker; the
module-level home is what lets the workers see it. Julia mirror of
`BayesForge.Diagnostic.sensitivity._prior_sens_sim`: a fresh bf instance with
its own seed, the model bound to one prior setting, and the posterior as plain
Julia data (Python objects cannot cross a process boundary).
"""
function _prior_sens_fit(; label, varied, priors, seed, model, obs, fit_kwargs, kw...)
    mm = importBF(rand_seed = seed, print_devices_found = false)
    bound = _BoundModel(model, mm, _model_data_names(model), _priors_kwargs(priors))
    np = pyimport("numpy")
    obs_py = pydict()
    for (k, v) in _pairs_any(obs)
        obs_py[_jstr(k)] = np.asarray(v)
    end
    fk = _sym_kwargs(fit_kwargs)
    # invokelatest: `model` arrives by serialisation, newer than this world.
    Base.invokelatest(() -> mm.fit(bound; obs = obs_py, progress_bar = false, fk...))
    post = Dict{String,Vector{Float64}}()
    posts = mm.posteriors
    for k in posts.keys()
        name = pyconvert(String, k)
        post[name] = vec(Array{Float64}(pyconvert(Array, posts[name])))
    end
    return (posteriors = post,)
end

# -------------------------------------------------------
# The entry point
# -------------------------------------------------------

_mean(v) = sum(v) / length(v)

# Population sd, i.e. numpy's np.std (ddof = 0).
function _std(v, mu = _mean(v))
    n = length(v)
    n < 1 && return NaN
    return sqrt(sum((x - mu)^2 for x in v) / n)
end

_hdi_jl(v, prob) = begin
    bounds = _sens_py()._hdi(v, prob)
    # Python indexing: [0] and [1] are the two bounds.
    (pyconvert(Float64, bounds[0]), pyconvert(Float64, bounds[1]))
end

"""
    bf_prior_sensitivity(diag, model_or_posts, args...; kwargs...)

Julia side of `m.diag.prior_sensitivity`. A Julia model function is fitted once
per prior scenario in Julia workers; anything else (a vector of posteriors you
already have) is handed to Python's `prior_sensitivity` unchanged.
"""
function bf_prior_sensitivity(diag, model_or_posts, args...; kwargs...)
    model_or_posts isa Function ||
        return _prior_sens_python(diag, model_or_posts, args...; kwargs...)
    # Positional arguments are named as in Python's signature.
    kw = Dict{Symbol,Any}(kwargs)
    for (i, a) in enumerate(args)
        i > length(_PRIOR_SENS_ARGS) &&
            error("prior_sensitivity takes at most $(length(_PRIOR_SENS_ARGS) + 1) positional arguments.")
        haskey(kw, _PRIOR_SENS_ARGS[i]) &&
            error("prior_sensitivity: argument $(_PRIOR_SENS_ARGS[i]) given both positionally and by name.")
        kw[_PRIOR_SENS_ARGS[i]] = a
    end
    g(name, default) = get(kw, name, default)
    priors = g(:priors, nothing)
    joint = g(:joint, nothing)
    isempty(_sym_kwargs(priors)) && isempty(_sym_kwargs(joint)) &&
        error("prior_sensitivity: pass `priors = Dict(arg => [values])` (or `joint = ...`).")

    scen = _prior_scenarios_jl(priors, g(:mode, "one_at_a_time"),
                               g(:baseline, nothing), joint)
    extra = filter(p -> !(p.first in _PRIOR_SENS_ARGS), collect(pairs(kw)))
    rest = isempty(extra) ? NamedTuple() : NamedTuple{Tuple(first.(extra))}(Tuple(last.(extra)))
    res = bf_run_simulations(_prior_sens_fit, scen;
                             fn_kwargs = (model = model_or_posts,
                                          obs = _obs_julia(g(:obs, nothing)),
                                          fit_kwargs = _sym_kwargs(g(:fit_kwargs, nothing))),
                             workers = g(:workers, nothing),
                             seed = g(:seed, 0),
                             rest...)

    bad = [r for r in res if !ismissing(r.error)]
    if !isempty(bad)
        msgs = join(("[$(r.label)] " * replace(String(r.error), "\n" => " ") for r in bad), "\n")
        error("prior_sensitivity: $(length(bad)) fit(s) failed:\n$msgs")
    end

    posts = [r.posteriors for r in res]
    lbls = [String(r.label) for r in res]
    varied = [String(r.varied) for r in res]
    base = lbls[1] == "baseline" ? posts[1] : nothing
    param_names = g(:param_names, nothing)
    pn = param_names === nothing ? String[] : _jstr.(collect(param_names))
    hdi_prob = g(:hdi_prob, 0.89)

    # The summary table: same columns as Python's prior_sensitivity.
    basestats = base === nothing ? Dict{String,Tuple{Float64,Float64}}() :
                Dict(k => (_mean(v), _std(v)) for (k, v) in base)
    rows = pylist()
    for (lab, v, post) in zip(lbls, varied, posts)
        for k in sort!(collect(keys(post)))
            isempty(pn) || k in pn || continue
            smp = post[k]
            mu, sd = _mean(smp), _std(smp)
            lo, hi = _hdi_jl(smp, hdi_prob)
            row = pydict()
            row["label"] = lab
            row["varied"] = v
            row["param"] = k
            row["mean"] = mu
            row["sd"] = sd
            row["hdi_low"] = lo
            row["hdi_high"] = hi
            if haskey(basestats, k)
                bmu, bsd = basestats[k]
                row["shift_sd"] = (mu - bmu) / (bsd == 0 ? NaN : bsd)
            end
            rows.append(row)
        end
    end
    table = pyimport("pandas").DataFrame(rows)

    posts_py = pylist([_post_py(p) for p in posts])
    lbls_py = pylist(lbls)
    pn_py = isempty(pn) ? _pynone() : pylist(pn)
    plot = _jstr(g(:plot, "by_prior"))
    mode = _jstr(g(:mode, "one_at_a_time"))
    if plot == "by_prior" && mode == "one_at_a_time" && base !== nothing
        fig = _sens_py().prior_sensitivity_by_prior(posts_py, lbls_py, pylist(varied),
                                                    pn_py, hdi_prob)
    elseif plot == "shift" && base !== nothing
        fig = _sens_py().prior_sensitivity_shift(table)
    else
        fig = _sens_py().prior_sensitivity_plot(posts_py, lbls_py;
                                                param_names = pn_py, hdi_prob = hdi_prob,
                                                title = _jstr(g(:title, "Prior Sensitivity Analysis")),
                                                x_label = _pynone(g(:x_label, nothing)),
                                                y_label = _pynone(g(:y_label, nothing)))
    end
    fig = BFObject(fig)
    return_results = g(:return_results, false)
    return_results = return_results isa Bool ? return_results : pyconvert(Bool, return_results)
    return return_results ? (fig, BFObject(table)) : fig
end

# BayesForge.Diagnostic.sensitivity._prior_scenarios -> Julia scenario rows for
# bf_run_simulations. The priors dict becomes a JSON string: Python objects
# cannot be sent to a worker.
function _prior_scenarios_jl(priors, mode, baseline, joint)
    scen = _sens_py()._prior_scenarios(_pydict_pairs(priors), _jstr(mode),
                                       _pydict_pairs(baseline), _pydict_pairs(joint))
    json = pyimport("json")
    out = NamedTuple[]
    for d in scen
        push!(out, (label = pyconvert(String, d["label"]),
                    varied = pyconvert(String, d["varied"]),
                    priors = pyconvert(String, json.dumps(d["priors"]))))
    end
    return out
end

# Posteriors you already fitted: Python's own path, unchanged.
function _prior_sens_python(diag, model_or_posts, args...; kwargs...)
    f = pyhasattr(diag.py, "prior_sensitivity") ? getproperty(diag.py, :prior_sensitivity) :
        getproperty(_sens_py(), :prior_sensitivity)
    args = [a isa AbstractVector ? _pystr_list(a) : _topy(a) for a in args]
    kw = Dict{Symbol,Any}(k => (k === :labels && v isa AbstractVector ? _pystr_list(v) : _topy(v))
                          for (k, v) in kwargs)
    val = f(_posts_py(model_or_posts), args...; kw...)
    return val isa Py ? BFObject(val) : val
end
