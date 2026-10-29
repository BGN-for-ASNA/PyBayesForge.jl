using Test
using PyBayesForge
using PythonCall
using DataFrames
using MCMCChains
using ArviZ
using Statistics

m_has(m, k) = m.posteriors.__contains__(k)

# run_simulations: @sim definitions must be at top level, where the workers
# replay them.
@sim sim_double(x) = 2x
@sim function sim_twice(; N, seed)
    println("hello ", N)
    (twice = sim_double(N), s = seed)
end
@sim function sim_boom(; N)
    N == 2 && error("boom")
    (ok = true,)
end
@sim sim_any(; kw...) = (has_sim = haskey(kw, :sim), nc = get(kw, :num_chains, missing))

@testset "PyBayesForge.jl" begin

    @testset "Module exports" begin
        # Test that main functions and macros are exported
        @test isdefined(PyBayesForge, :importBF)
        @test isdefined(PyBayesForge, :jnp)
        @test isdefined(PyBayesForge, :jax)
        @test isdefined(PyBayesForge, Symbol("@BF"))
        @test isdefined(PyBayesForge, Symbol("@pyplot"))
    end

    @testset "Python interop basics" begin
        # Test that pybuiltins, pydict, pylist are accessible
        @test isdefined(PyBayesForge, :pybuiltins)
        @test isdefined(PyBayesForge, :pydict)
        @test isdefined(PyBayesForge, :pylist)
    end

    @testset "BF Initialization" begin
        # Test basic initialization
        # Note: This may take time on first run as CondaPkg sets up the environment
        println("Initializing BF (may take time on first run)...")
        m = importBF(print_devices_found=false)
        @test !isnothing(m)
        @test pyhasattr(m, "dist")
        println("✓ BF initialized successfully")
    end

    @testset "JAX/NumPy availability" begin
        # Test that jax and jnp are properly initialized
        @test !isnothing(jnp)
        @test !isnothing(jax)

        # Test basic jnp operations
        arr = jnp.array([1, 2, 3])
        @test !isnothing(arr)
        @test pyconvert(Int, arr.shape[0]) == 3
    end

    @testset "InspectableFunction wrapper" begin
        # Test the @BF macro creates proper wrapper
        @BF function test_model(x, y)
            return x + y
        end

        @test typeof(test_model) == PyBayesForge.InspectableFunction
        @test test_model(2, 3) == 5
    end

    @testset "Fit a model end to end" begin
        m = importBF(platform = "cpu", rand_seed = 1, print_devices_found = false)
        x = collect(range(-1, 1; length = 40))
        y = 0.5 .+ 2.0 .* x .+ 0.1 .* cos.(7 .* x)
        m.data_on_model = pydict(x = jnp.array(x), y = jnp.array(y))

        @BF function linreg(x, y)
            a = m.dist.normal(0, 5, name = "a")
            b = m.dist.normal(0, 5, name = "b")
            s = m.dist.exponential(1, name = "s")
            m.dist.normal(a + b * x, s, obs = y)
        end

        m.fit(linreg, num_samples = 200, num_warmup = 200, num_chains = 1, progress_bar = false)
        b_hat = pyconvert(Float64, m.posteriors["b"].mean().item())
        @test isapprox(b_hat, 2.0; atol = 0.2)
        @test pyconvert(Int, m.get_seed()) == 1
    end

    @testset "@BF infers parameter names from assignments" begin
        m = importBF(platform = "cpu", rand_seed = 3, print_devices_found = false)
        x = collect(range(-1, 1; length = 30))
        m.data_on_model = pydict(x = jnp.array(x), y = jnp.array(0.5 .+ 2.0 .* x))
        @BF function named(x, y)
            a = m.dist.normal(0, 2)                    # name inferred: "a"
            b = m.dist.normal(0, 2, name = "slope")    # explicit name kept
            s = m.dist.exponential(1)                  # name inferred: "s"
            m.dist.normal(a + b * x, s, obs = y)
        end
        m.fit(named, num_samples = 50, num_warmup = 50, num_chains = 1, progress_bar = false)
        @test sort(pyconvert(Vector{String}, pybuiltins.list(m.posteriors.keys()))) == ["a", "s", "slope"]
    end

    @testset "Julia data frames in, Julia tables out" begin
        m = importBF(platform = "cpu", rand_seed = 2, print_devices_found = false)
        x = collect(range(-1, 1; length = 30))
        df = DataFrame(x = x, y = 0.5 .+ 2.0 .* x .+ 0.1 .* cos.(7 .* x))
        m.data(df)                         # a Julia DataFrame passed directly
        @test pyconvert(Int, m.df.shape[0]) == 30
        m.data_to_model(["x", "y"])

        @BF function lin(x, y)
            a = m.dist.normal(0, 2, name = "a", shape = (2,))
            s = m.dist.exponential(1, name = "s")
            m.dist.normal(a[0] + a[1] * x, s, obs = y)   # 0-based inside the model
        end
        m.fit(lin, num_samples = 50, num_warmup = 50, num_chains = 2, progress_bar = false)

        t = bf_draws(m)
        @test collect(keys(t)) == [:chain, :iteration, :draw, Symbol("a[1]"), Symbol("a[2]"), :s]
        @test t.chain == repeat(1:2, inner = 50)
        np = pyimport("numpy")
        a = pyconvert(Array{Float64}, np.asarray(m.py.sampler.get_samples(group_by_chain = true)["a"]))
        @test t[Symbol("a[2]")][50 + 7] == a[2, 7, 2]     # chain 2, iteration 7, element 2
        @test isapprox(sum(t[Symbol("a[2]")]) / 100, 2.0; atol = 0.2)
        @test size(DataFrame(t)) == (100, 6)

        ch = Chains(m)
        @test size(ch) == (50, 3, 2)
    end

    @testset "enum = true and fit(sampler = ...) reach Python" begin
        rng_y = [-2.1, -1.8, -2.4, 2.2, 1.9, 2.5, -2.0, 2.1, -1.7, 2.3,
                 -2.2, 1.8, -1.9, 2.0, -2.3, 2.4, -2.0, 1.7, -2.5, 2.2]
        function fit_mixture(mark; kw...)
            m = importBF(platform = "cpu", rand_seed = 123, print_devices_found = false)
            m.data_on_model = pydict(y = jnp.array(rng_y))
            @BF function mix(y)
                w = m.dist.dirichlet(jnp.ones(2), name = "w")
                sigma = m.dist.exponential(1.0, name = "sigma")
                mu = nothing
                pywith(Py(m.dist.plate("components", 2))) do _
                    mu = m.dist.normal(0.0, 5.0, name = "mu")
                end
                pywith(Py(m.dist.plate("N", length(rng_y)))) do _
                    z = mark == :enum  ? m.dist.categorical(probs = w, name = "z", enum = true) :
                        mark == :infer ? m.dist.categorical(probs = w, name = "z",
                                                            infer = pydict(enumerate = "parallel")) :
                                         m.dist.categorical(probs = w, name = "z")
                    m.dist.normal(jnp.take(mu, z), sigma, obs = y)
                end
            end
            m.fit(mix; num_samples = 100, num_warmup = 100, num_chains = 1, progress_bar = false, kw...)
            m
        end
        np = pyimport("numpy")
        draws(m, k) = pyconvert(Array{Float64}, np.asarray(m.posteriors[k]))
        has_kw(f, k) = pyconvert(Bool, pyimport("inspect").signature(f).parameters.__contains__(k))
        bfpy = pyimport("BayesForge")
        if has_kw(bfpy.Distributions.np_dists.UnifiedDist.categorical, "enum")
            a, b = fit_mixture(:enum), fit_mixture(:infer)
            @test !pyconvert(Bool, m_has(a, "z"))
            for k in ("w", "mu", "sigma")
                @test draws(a, k) == draws(b, k)          # bit-identical
            end
        else
            @info "installed BayesForge has no enum=; enum test skipped"
        end
        if has_kw(bfpy.bf.fit, "sampler")
            g = fit_mixture(:none; sampler = "discrete_gibbs")
            @test pyconvert(String, g.sampler_name) == "discrete_gibbs"
            @test pyconvert(Bool, m_has(g, "z"))          # sampled, not summed out
        else
            @info "installed BayesForge has no fit(sampler=); sampler test skipped"
        end
    end

    @testset "@BF names observed sites after the obs variable" begin
        ex = PyBayesForge._add_inferred_names!(:(begin
            x = m.dist.normal(0, 5)
            t2 = m.dist.track("t2", x * 2)
            m.dist.normal(x, 1, obs = y)
            m.dist.normal(x, 1, obs = y .- 1)
            m.dist.normal(x, 1, obs = y, name = "kept")
        end))
        names = String[]
        walk(e) = e isa Expr && (e.head === :kw && e.args[1] === :name ?
                                 push!(names, e.args[2]) : foreach(walk, e.args))
        walk(ex)
        @test names == ["x", "y", "obs", "kept"]

        m = importBF(platform = "cpu", rand_seed = 4, print_devices_found = false)
        t = collect(range(-1, 1; length = 25))
        m.data_on_model = pydict(t = jnp.array(t), y = jnp.array(1 .+ 2 .* t))
        @BF function latent_x(t, y)
            x = m.dist.normal(0, 5)              # a latent called x
            s = m.dist.exponential(1)
            m.dist.normal(x + 0 * t, s, obs = y) # used to be named "x" too
        end
        m.fit(latent_x, num_samples = 50, num_warmup = 50, num_chains = 1, progress_bar = false)
        @test sort(pyconvert(Vector{String}, pybuiltins.list(m.posteriors.keys()))) == ["s", "x"]
        @test pyconvert(Vector{String}, pybuiltins.list(m._observed_sites(m.data_on_model).keys())) == ["y"]
    end

    @testset "tidy draws: long format with labels" begin
        m = importBF(platform = "cpu", rand_seed = 6, print_devices_found = false)
        g = repeat([0, 1, 2], inner = 10)
        y = [1.0, 2.0, 3.0][g .+ 1] .+ 0.1 .* sin.(1:30)
        m.data_on_model = pydict(g = jnp.array(g), y = jnp.array(y))
        @BF function grp(g, y)
            alpha = m.dist.normal(0, 5, shape = (3,))
            s = m.dist.exponential(1)
            m.dist.normal(alpha[g], s, obs = y)
        end
        m.fit(grp, num_samples = 100, num_warmup = 100, num_chains = 2, progress_bar = false)
        levels = ["south", "north", "east"]
        wide = bf_draws(m)
        long = bf_draws(m; format = :long, labels = Dict("alpha" => levels))
        @test collect(keys(long)) == [:chain, :iteration, :draw, :variable, :index, :value]
        @test length(long.value) == length(wide.draw) * (length(keys(wide)) - 3)
        for (k, lev) in enumerate(levels)
            sel = (long.variable .== "alpha") .& isequal.(long.index, lev)
            @test long.value[sel] == wide[Symbol("alpha[$k]")]
        end
        @test all(ismissing, long.index[long.variable .== "s"])
        @test_throws ArgumentError bf_draws(m; format = :long, labels = Dict("beta" => ["a"]))
        @test_throws ArgumentError bf_draws(m; format = :long, labels = Dict("alpha" => ["a"]))
    end

    @testset "ArviZ.jl: convert_to_inference_data(m)" begin
        m = importBF(platform = "cpu", rand_seed = 8, print_devices_found = false)
        x = collect(range(-1, 1; length = 40))
        y = 0.5 .+ 2.0 .* x .+ 0.2 .* sin.(5 .* x)
        m.data_on_model = pydict(x = jnp.array(x), y = jnp.array(y))
        @BF function lr(x, y)
            a = m.dist.normal(0, 2)
            b = m.dist.normal(0, 2)
            s = m.dist.exponential(1)
            m.dist.normal(a + b * x, s, obs = y)
        end
        m.fit(lr, num_samples = 200, num_warmup = 200, num_chains = 2, progress_bar = false)
        idata = convert_to_inference_data(m)
        @test :posterior in propertynames(idata)
        @test :log_likelihood in propertynames(idata)
        @test :observed_data in propertynames(idata)
        np = pyimport("numpy")
        b_py = pyconvert(Array{Float64}, np.asarray(m.py.sampler.get_samples(group_by_chain = true)["b"]))  # chains x draws
        b_jl = Array(idata.posterior[:b])                                                                   # draws x chains
        @test b_jl == permutedims(b_py, (2, 1))                       # same draws, ArviZ layout
        @test size(Array(idata.log_likelihood[:y])) == (200, 2, 40)    # draws x chains x observations
        @test Array(idata.observed_data[:y]) == y
        @test isapprox(mean(b_jl), mean(b_py); rtol = 1e-12)
        elpd_jl = loo(idata).estimates.elpd
        elpd_py = pyconvert(Float64, m.py.diag.loo().elpd)
        @test abs(elpd_jl - elpd_py) < 0.1                             # two PSIS-LOO implementations
    end

    @testset "m.grid and m.run_simulations" begin
        m = importBF(print_devices_found = false)
        sim = pyimport("BayesForge.Parallel.simulate")

        g = m.grid(N = [100, 500], b_true = [-0.8, 0.0, 0.8], reps = 4)
        @test length(g) == 24
        @test collect(keys(g[1])) == [:N, :b_true, :sim, :rep]
        @test [r.sim for r in g] == 0:23          # 0-based, as in Python
        @test nrow(DataFrame(g)) == 24             # a Tables.jl row table

        # serial and 2 worker processes agree; seeds come from Python
        rs = m.run_simulations(sim_twice, m.grid(N = 1:4); backend = :serial, progress = false)
        rp = m.run_simulations(sim_twice, m.grid(N = 1:4); workers = 2, progress = false)
        key(r) = (r.sim, r.N, r.seed, r.twice)
        @test key.(rs) == key.(rp)
        @test length(unique(r.worker for r in rp)) == 2
        @test !any(r.worker == getpid() for r in rp)
        @test rs[4].seed == pyconvert(Int, sim._derive_seed(0, 3))

        # errors recorded per row, or raised
        rb = @test_logs (:warn, r"first \(sim 1\): boom") match_mode = :any m.run_simulations(
            sim_boom, m.grid(N = 1:3); workers = 2, progress = false)
        @test [ismissing(r.error) for r in rb] == [true, false, true]
        @test_throws ErrorException m.run_simulations(sim_boom, m.grid(N = 1:3);
                                                      backend = :serial, on_error = :raise,
                                                      progress = false)

        # kw... receives every key; num_chains is injected
        ra = m.run_simulations(sim_any, m.grid(N = 1:2); num_chains = 1, workers = 2,
                               progress = false)
        @test all(r.has_sim for r in ra)
        @test all(r.nc == 1 for r in ra)

        # progress lines
        out = mktemp() do path, io
            redirect_stdout(io) do
                m.run_simulations(sim_twice, m.grid(N = 1:2); backend = :serial)
            end
            close(io)
            read(path, String)
        end
        @test occursin("worker main | sim 0", out)
        @test occursin("hello 1", out)

        # returning the model is refused
        r = m.run_simulations((; N) -> m, m.grid(N = 1); backend = :serial, progress = false)
        @test occursin("cannot cross a process boundary", r[1].error)

        # every other property is still Python's
        @test m.fit isa PyBayesForge.BFObject
    end
end
