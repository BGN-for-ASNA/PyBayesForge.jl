# -------------------------------------------------------
# Parallel simulations: m.grid / m.run_simulations
# -------------------------------------------------------
# The Julia side of BayesForge.Parallel.simulate. Core splitting and seed
# derivation stay in Python (_plan), so Python, R and Julia budget cores and
# seed simulations identically; Julia only runs the workers (Distributed) and
# builds the result table. Python's own run_simulations cannot be used: its
# workers are Python processes, which cannot run a Julia function.

_simulate_py() = pyimport("BayesForge.Parallel.simulate")

# Definitions the workers need, recorded by @sim and replayed in each worker's
# Main before it runs anything. Distributed sends a function defined in Main by
# name only (anonymous ones included), so a worker that never saw the
# definition fails with UndefVarError.
const _SIM_DEFS = Pair{Any,Any}[]

function _sim_def_name(ex)
    ex isa Expr && ex.head === :macrocall && return nothing
    ex isa Expr && ex.head in (:function, :(=)) || return nothing
    call = ex.args[1]
    while call isa Expr && call.head === :where
        call = call.args[1]
    end
    return call isa Expr && call.head === :call ? call.args[1] : call
end

function _register_sim_def!(ex)
    name = _sim_def_name(ex)
    i = name === nothing ? nothing : findfirst(p -> first(p) == name, _SIM_DEFS)
    i === nothing ? push!(_SIM_DEFS, name => ex) : (_SIM_DEFS[i] = name => ex)
    return nothing
end

"""
    @sim function one_sim(; N, seed) ... end

Define a function here and on every worker of `m.run_simulations`. Workers
are new Julia processes created by the runner; they replay every `@sim`
definition (in order, the latest one per name) before running. Mark the
simulation function, any helper it calls, and any `using` it needs beyond
PyBayesForge and PythonCall (`@sim using Statistics`).

```julia
@sim helper(x) = 2x
@sim function one_sim(; N, seed)
    (v = helper(N),)
end
res = m.run_simulations(one_sim, m.grid(N = [10, 20]))
```
"""
macro sim(ex)
    quote
        $(esc(ex))
        $_register_sim_def!($(QuoteNode(ex)))
    end
end

_isbf(x::BFObject) = pyisinstance(x.py, pyimport("BayesForge").bf)

# Python values -> Julia (int, float, str, bool, None); anything else stays Py.
_tojl(v) = v
_tojl(v::BFObject) = _tojl(v.py)
_tojl(v::Py) = pyis(v, pybuiltins.None) ? missing : pyconvert(Any, v)

# A Python dict -> ordered Vector of Symbol => value pairs (dicts keep insertion order).
_pairs(d::Py) = [Symbol(pyconvert(String, k)) => _tojl(d[k]) for k in d]

# Ordered pairs -> NamedTuple, so a vector of them is a Tables.jl row table.
_nt(ps) = NamedTuple{Tuple(first.(ps))}(Tuple(last.(ps)))

_pyval(v::AbstractVector) = pylist(v)
_pyval(v::AbstractRange) = pylist(collect(v))
_pyval(v) = v

"""
    m.grid(; reps = 1, params...)

Scenarios from the cartesian product of `params`, each repeated `reps` times.
Returns a vector of named tuples (a Tables.jl row table, so `DataFrame(g)`
works) with the parameters plus `sim` (0-based, as in Python) and `rep`.

```julia
scenarios = m.grid(N = [100, 500], b_true = [-0.8, 0.0, 0.8], reps = 4)  # 24
```
"""
function bf_grid(; reps = 1, params...)
    g = _simulate_py().grid(; reps = reps, (k => _pyval(v) for (k, v) in params)...)
    return [_nt(_pairs(d)) for d in g]
end

# Scenarios as Python's _normalise_scenarios accepts them: an int, or a list of dicts.
_scenarios_to_py(n::Integer) = n
function _scenarios_to_py(sc)
    rows = Tables.istable(sc) && !(sc isa AbstractVector{<:AbstractDict}) ? Tables.rowtable(sc) : sc
    out = pylist()
    for r in rows
        d = pydict()   # filled in order: a Julia Dict would scramble the columns
        for (k, v) in pairs(r)
            d[String(k)] = v
        end
        out.append(d)
    end
    return out
end

# Keyword arguments fn accepts: Python passes every scenario key, Julia would
# throw "unsupported keyword argument" on the extras (sim, rep, ...).
function _accepted(fn, kw::NamedTuple)
    ms = methods(fn)
    length(ms) == 1 || return kw
    names = Base.kwarg_decl(first(ms))
    any(n -> endswith(String(n), "..."), names) && return kw
    keep = Tuple(k for k in keys(kw) if k in names)
    return NamedTuple{keep}(kw)
end

# Mirror of Python _normalise_result: whatever fn returned becomes rows of
# Symbol => value pairs. Python objects cannot cross a process boundary, so
# values are converted to Julia here, in the worker.
_row(r::NamedTuple) = [k => _tojl(v) for (k, v) in pairs(r)]
_row(r::AbstractDict) = [Symbol(k) => _tojl(v) for (k, v) in r]

function _sim_rows(ret)
    if ret isa BFObject && _isbf(ret)
        error("the simulation function returned the `m` object. A fitted model " *
              "cannot cross a process boundary. Return summaries instead, e.g. " *
              "(b_hat = pyconvert(Float64, est.loc[\"b\", \"mean\"]),).")
    end
    ret === nothing && return [Pair{Symbol,Any}[]]
    (ret isa NamedTuple || ret isa AbstractDict) && return [_row(ret)]
    if ret isa AbstractVector
        isempty(ret) && return [Pair{Symbol,Any}[]]
        all(r -> r isa NamedTuple || r isa AbstractDict, ret) && return [_row(r) for r in ret]
    end
    Tables.istable(ret) && return [_row(r) for r in Tables.rowtable(ret)]
    return [[:value => _tojl(ret)]]
end

# One scenario, start to rows. Runs identically in the master (serial) and in
# a worker, so the serial path is a genuine debugging equivalent.
function _sim_task(fn, fn_kwargs::NamedTuple, scenario::NamedTuple)
    kw = _accepted(fn, merge(fn_kwargs, scenario))
    t0 = time()
    rows = [Pair{Symbol,Any}[]]   # keep the scenario row on failure, like Python's [{}]
    err = missing
    try
        # invokelatest: fn arrives by serialisation, newer than this task's world.
        rows = _sim_rows(Base.invokelatest(fn; kw...))
    catch e
        err = sprint(showerror, e)
        if occursin("not defined in `Main`", err)
            err *= "\n(Functions defined in Main do not reach the workers: mark one_sim and " *
                   "the helpers it calls with @sim, or load them with " *
                   "run_simulations(...; setup = :(include(\"file.jl\"))).)"
        end
    end
    elapsed = time() - t0
    meta = Pair{Symbol,Any}[pairs(scenario)...,
                            :worker => getpid(), :elapsed_s => round(elapsed; digits = 3),
                            :error => err]
    # Meta first, result second: a result column of the same name wins, as
    # Python _run_one does.
    out = map(rows) do row
        names = Set(first.(row))
        vcat(filter(p -> !(first(p) in names), meta), row)
    end
    return (rows = out, scenario = scenario, error = err, elapsed = elapsed)
end

# One progress line per finished simulation. fn's own println output is
# forwarded live by Distributed, prefixed "From worker k:".
function _sim_report(r, node, done, total)
    sc = r.scenario
    shown = [k => v for (k, v) in pairs(sc) if !(k in (:sim, :rep, :seed, :num_chains))]
    values = join(("$(k)=$(v isa AbstractFloat ? round(v; sigdigits = 4) : v)"
                   for (k, v) in first(shown, 4)), ", ")
    status = ismissing(r.error) ? "ok" : "ERROR: " * first(replace(r.error, '\n' => ' '), 80)
    w = ndigits(total)
    println("[", lpad(done, w), "/", total, "] worker ", node, " | sim ", sc.sim, " | ",
            values, " | ", round(r.elapsed; digits = 1), "s | ", status)
end

# Rows with differing keys -> vector of named tuples over the union of keys
# (first-seen order), missing where a row lacks one.
function _rbind_fill(rows)
    isempty(rows) && return NamedTuple[]
    names = Symbol[]
    for row in rows, (k, _) in row
        k in names || push!(names, k)
    end
    return [begin
                d = Dict(row)
                NamedTuple{Tuple(names)}(Tuple(get(d, k, missing) for k in names))
            end for row in rows]
end

"""
    m.run_simulations(fn, scenarios; workers = nothing, num_chains = nothing,
                      cores_per_worker = nothing, fn_kwargs = (;), seed = 0,
                      on_error = :record, quiet = true, progress = true,
                      backend = :process, setup = nothing)

Run independent simulations in parallel Julia worker processes and return a
vector of named tuples (a Tables.jl row table: `DataFrame(res)`), one row per
result, carrying the scenario columns plus `sim`, `rep`, `seed`, `worker`,
`elapsed_s` and `error`.

`fn` is called with the scenario as keyword arguments, plus `fn_kwargs`; keys
it does not declare are dropped unless it takes `kw...`. Workers are new Julia
processes: define `fn`, and every helper it calls, with [`@sim`](@ref) so the
workers get the definitions, or load them with `setup`, an expression
evaluated in `Main` on every worker (e.g. `setup = :(include("sim.jl"))`).
`fn`'s own `println` output is forwarded live, prefixed `From worker k:`.

The core budget, the seeds and the worker environment come from Python
(`BayesForge.Parallel.simulate._plan`), so the same scenarios get the same
seeds in Python, R and Julia. `num_chains` reserves one core per chain in each
worker and is passed to `fn`. `on_error = :record` keeps going and stores the
message in `error`; `:raise` throws. `backend = :serial` runs in this process.
"""
function bf_run_simulations(fn, scenarios; workers = nothing, num_chains = nothing,
                            cores_per_worker = nothing, fn_kwargs = (;), seed = 0,
                            on_error = :record, quiet = true, progress = true,
                            backend = :process, setup = nothing)
    on_error = Symbol(on_error)
    backend = Symbol(backend)
    on_error in (:record, :raise) || throw(ArgumentError("on_error must be :record or :raise"))
    backend in (:process, :serial) || throw(ArgumentError("backend must be :process or :serial"))
    fn_kwargs = NamedTuple(fn_kwargs)

    plan = _simulate_py()._plan(_scenarios_to_py(scenarios); workers = workers,
                                num_chains = num_chains, cores_per_worker = cores_per_worker,
                                seed = seed, quiet = quiet)
    for w in plan["warnings"]
        @warn pyconvert(String, w)
    end
    scen = [_nt(_pairs(d)) for d in plan["scenarios"]]
    nw = pyconvert(Int, plan["workers"])
    n = length(scen)
    results = Vector{Any}(undef, n)

    if backend === :serial || nw == 1
        progress && println("run_simulations: $n simulations in this session")
        for j in 1:n
            r = _sim_task(fn, fn_kwargs, scen[j])
            results[j] = r
            progress && _sim_report(r, "main", j, n)
            if on_error === :raise && !ismissing(r.error)
                error("simulation $(r.scenario.sim) failed: $(r.error)")
            end
        end
    else
        env = Dict(pyconvert(String, k) => pyconvert(String, v) for (k, v) in plan["env"].items())
        # run_simulations budgets CPU cores; without this every worker's JAX
        # probes for CUDA at import and prints a fallback warning. A
        # JAX_PLATFORMS already set by the user wins.
        env["JAX_PLATFORMS"] = get(ENV, "JAX_PLATFORMS", "cpu")
        block_size = pyconvert(Int, plan["block_size"])
        cpw = pyconvert(Int, plan["cores_per_worker"])
        # The environment is fixed at process start, before Python and XLA
        # come up in the worker: XLA reads its device count only once.
        pids = Distributed.addprocs(nw; env = collect(env),
                                    exeflags = ["--project=$(Base.active_project())", "--threads=1"])
        try
            # PythonCall through PyBayesForge: the user's project may not list it,
            # but one_sim usually calls pyconvert / pydict.
            Distributed.remotecall_eval(Main, pids, :(using PyBayesForge, PyBayesForge.PythonCall))
            # One disjoint CPU block per worker, set before the JAX backend comes up.
            @sync for (i, p) in enumerate(pids)
                @async Distributed.remotecall_wait(_pin_worker, p, i - 1, block_size)
            end
            setup === nothing || Distributed.remotecall_eval(Main, pids, setup)
            for (_, ex) in _SIM_DEFS
                Distributed.remotecall_eval(Main, pids, ex)
            end
            progress && println("run_simulations: $n simulations on $nw workers ($cpw cores each)")
            # Each worker pulls the next scenario as soon as it is free, so
            # results are reported as they arrive. Tasks are cooperative and
            # the counter is read and bumped with no yield in between.
            next = 0
            done = 0
            # Labelled by Distributed id, as in the "From worker k:" lines.
            @sync for p in pids
                @async while true
                    next += 1
                    j = next
                    j > n && break
                    r = Distributed.remotecall_fetch(_sim_task, p, fn, fn_kwargs, scen[j])
                    results[j] = r
                    done += 1
                    progress && _sim_report(r, p, done, n)
                end
            end
        finally
            Distributed.rmprocs(pids)
        end
    end

    rows = reduce(vcat, (r.rows for r in results))
    res = _rbind_fill(rows)
    sort!(res; by = r -> r.sim)
    bad = [r for r in res if !ismissing(r.error)]
    if !isempty(bad)
        first_bad = bad[argmin([r.sim for r in bad])]
        on_error === :raise && error("simulation $(first_bad.sim) failed: $(first_bad.error)")
        @warn "run_simulations: $(length(bad)) row(s) carry an error -- first (sim $(first_bad.sim)): " *
              first(first_bad.error, 200)
    end
    return res
end

_pin_worker(block, block_size) =
    (_simulate_py()._pin_worker(block, block_size); nothing)
