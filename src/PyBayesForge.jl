module PyBayesForge

using PythonCall
using CondaPkg
using Tables

# Check if utils.jl exists in your folder, otherwise comment this out
if isfile(joinpath(@__DIR__, "utils.jl"))
    include("utils.jl")
    using .Utils
    export create_env, delete_env, install_package, update_package
end

export importBF, bf_draws, @BF, @pyplot, jnp, jax, pybuiltins, pydict, pylist

# -------------------------------------------------------
# BFObject Wrapper
# -------------------------------------------------------

const PyWrappers = Union{Py,PythonCall.PyArray,PythonCall.PyList,PythonCall.PyDict,PythonCall.PyIterable,PythonCall.PySet}

"""
    BFObject(py::Py)

A wrapper around a Python object (`Py`) to provide a more Julia-friendly interface
without committing type piracy.
"""
struct BFObject
    py::Py
end

# Outer constructor for other Python wrappers
BFObject(x::PyWrappers) = BFObject(Py(x))

# Allow easy conversion back to Py
PythonCall.Py(x::BFObject) = x.py

# Delegate show to the underlying Py object
Base.show(io::IO, x::BFObject) = show(io, x.py)
Base.show(io::IO, mime::MIME, x::BFObject) = show(io, mime, x.py)
Base.show(io::IO, mime::MIME"text/plain", x::BFObject) = show(io, mime, x.py)
Base.show(io::IO, mime::MIME"text/html", x::BFObject) = show(io, mime, x.py)
Base.show(io::IO, mime::MIME"image/png", x::BFObject) = show(io, mime, x.py)
Base.show(io::IO, mime::MIME"image/jpeg", x::BFObject) = show(io, mime, x.py)
Base.show(io::IO, mime::MIME"image/svg+xml", x::BFObject) = show(io, mime, x.py)
Base.show(io::IO, mime::MIME"application/json", x::BFObject) = show(io, mime, x.py)

# Allow pyconvert to work on BFObject
PythonCall.pyconvert(T::Type, x::BFObject) = pyconvert(T, x.py)

# -------------------------------------------------------
# Julia -> Python argument conversion
# -------------------------------------------------------

# Julia tables (DataFrame or any Tables.jl source) become pandas DataFrames on
# the Python side, so `m.data(df)` / `m.df = df` take a Julia data frame
# directly. Arrays, dicts, named tuples and scalars keep PythonCall's default
# conversion (a dict of vectors is also a Tables.jl table, but means data).
_is_table(a) = !(a isa Union{AbstractArray,AbstractDict,NamedTuple,AbstractString,Number,Symbol,Nothing,Py}) &&
               Tables.istable(a)
_topy(a::BFObject) = a.py
_topy(a) = _is_table(a) ? pytable(a, :pandas) : a

# -------------------------------------------------------
# BFObject Behavior
# -------------------------------------------------------

# 1. Property Access
function Base.getproperty(x::BFObject, s::Symbol)
    if s === :py
        return getfield(x, :py)
    end

    # Delegate to Python
    val = getproperty(x.py, s)

    # Wrap result if it's a Py object
    if val isa Py
        return BFObject(val)
    else
        return val
    end
end

function Base.setproperty!(x::BFObject, s::Symbol, v)
    if s === :py
        return setfield!(x, :py, v)
    end

    setproperty!(x.py, s, _topy(v))
end

# 2. Indexing
function Base.getindex(x::BFObject, idx...)
    # Convert Colon to slice(nothing)
    new_idx = map(i -> i == Colon() ? pybuiltins.slice(nothing) : i, idx)

    val = getindex(x.py, new_idx...)

    if val isa Py
        return BFObject(val)
    else
        return val
    end
end

function Base.setindex!(x::BFObject, v, idx...)
    new_idx = map(i -> i == Colon() ? pybuiltins.slice(nothing) : i, idx)
    val = v isa BFObject ? v.py : v
    setindex!(x.py, val, new_idx...)
end

# 3. Functor (Calling the object)
function (obj::BFObject)(args...; kwargs...)
    # Unwrap BFObjects, convert Julia tables to pandas
    new_args = map(_topy, args)
    new_kwargs = Dict{Symbol,Any}(k => _topy(v) for (k, v) in kwargs)

    val = obj.py(new_args...; new_kwargs...)

    if val isa Py
        return BFObject(val)
    else
        return val
    end
end

# 4. Arithmetic
for op in (:+, :-, :*, :/, ://, :%, :^)
    @eval begin
        Base.$op(a::BFObject, b::BFObject) = BFObject($op(a.py, b.py))
        Base.$op(a::BFObject, b::Number) = BFObject($op(a.py, b))
        Base.$op(a::Number, b::BFObject) = BFObject($op(a, b.py))
        # Handle Py objects too if mixed
        Base.$op(a::BFObject, b::Py) = BFObject($op(a.py, b))
        Base.$op(a::Py, b::BFObject) = BFObject($op(a, b.py))
    end
end

# Comparison Operators
for op in (:(==), :(!=), :(<), :(<=), :(>), :(>=), :isless)
    @eval begin
        Base.$op(a::BFObject, b::BFObject) = BFObject($op(a.py, b.py))
        Base.$op(a::BFObject, b::Number) = BFObject($op(a.py, b))
        Base.$op(a::Number, b::BFObject) = BFObject($op(a, b.py))
        Base.$op(a::BFObject, b::Py) = BFObject($op(a.py, b))
        Base.$op(a::Py, b::BFObject) = BFObject($op(a, b.py))
    end
end

# 5. Iteration & Length
function Base.iterate(x::BFObject, state...)
    res = iterate(x.py, state...)
    if res === nothing
        return nothing
    else
        val, new_state = res
        return (val isa Py ? BFObject(val) : val, new_state)
    end
end

Base.length(x::BFObject) = length(x.py)


"""
    importBF(; platform="cpu", cores=nothing, rand_seed=true, deallocate=false, print_devices_found=true, backend="numpyro")

Initializes and returns a BayesForge session object (`bf` from the `BayesForge`
Python package), wrapped for Julia. On first use CondaPkg installs the Python
dependencies declared in `CondaPkg.toml`.

# Example
```julia
using PyBayesForge
m = importBF(platform = "cpu", rand_seed = 42)
```
"""
function importBF(;
    platform="cpu",
    cores=nothing,
    rand_seed=true,
    deallocate=false,
    print_devices_found=true,
    backend="numpyro"
)
    # Import BayesForge module
    bf_module = pyimport("BayesForge")

    # Access the bf class from the module
    BF_class = bf_module.bf

    # Instantiate and return the BF object
    return BFObject(BF_class(
        platform=platform,
        cores=cores,
        rand_seed=rand_seed,
        deallocate=deallocate,
        print_devices_found=print_devices_found,
        backend=backend
    ))
end

"""
    bf_draws(m)

Posterior draws of a fitted model as a Tables.jl column table (a `NamedTuple`
of vectors), one row per draw: `chain`, `iteration`, `draw`, then one column
per scalar parameter element. A vector parameter `a` gives `Symbol("a[1]")`,
`Symbol("a[2]")`, ...; a matrix parameter `S` gives `Symbol("S[1,1]")`, ...

Column names are 1-based (Julia convention), whereas indexing *inside model
code* is passed to Python and is 0-based: `a[0]` in the model is `a[1]` here.

```julia
using DataFrames
df = DataFrame(bf_draws(m))
```

With MCMCChains loaded, `Chains(m)` gives an `MCMCChains.Chains` object.
"""
function bf_draws(m::BFObject)
    np = pyimport("numpy")
    samples = m.py.sampler.get_samples(group_by_chain = true)
    names = Symbol[]
    columns = Vector{Float64}[]
    nchains = niter = 0
    for key in samples.keys()
        name = pyconvert(String, key)
        arr = pyconvert(Array{Float64}, np.asarray(samples[key], dtype = "float64"))
        nchains, niter = size(arr, 1), size(arr, 2)
        # rows ordered by chain, then iteration
        rows(slice) = vec(permutedims(slice))
        if ndims(arr) == 2
            push!(names, Symbol(name))
            push!(columns, rows(arr))
        else
            trailing = size(arr)[3:end]
            flat = reshape(arr, nchains, niter, prod(trailing))
            for (k, idx) in enumerate(CartesianIndices(trailing))
                push!(names, Symbol(name, "[", join(Tuple(idx), ","), "]"))
                push!(columns, rows(flat[:, :, k]))
            end
        end
    end
    ids = (chain = repeat(1:nchains, inner = niter),
           iteration = repeat(1:niter, outer = nchains),
           draw = collect(1:nchains*niter))
    return merge(ids, NamedTuple{Tuple(names)}(Tuple(columns)))
end

# -------------------------------------------------------
# Function Wrapper & Macro
# -------------------------------------------------------

# 1. The Wrapper Struct
struct InspectableFunction <: Function
    f::Function
end

# 2. Make it callable in Julia (so it behaves like a normal function)
function (obj::InspectableFunction)(args...; kwargs...)
    # Wrap args in BFObject if they are Py objects
    new_args = map(a -> a isa PyWrappers ? BFObject(a) : a, args)

    # Handle kwargs
    new_kwargs = Dict{Symbol,Any}()
    for (k, v) in kwargs
        new_kwargs[k] = v isa PyWrappers ? BFObject(v) : v
    end

    return obj.f(new_args...; new_kwargs...)
end

# 3. Teach PythonCall how to convert it
#    Called automatically when the function is passed to Python (e.g. m.fit(model)).
#    BayesForge matches model arguments to data by NAME, so the Python callable
#    must expose the Julia argument names: build `lambda <names>: fn(<names>)`.
function PythonCall.Py(obj::InspectableFunction)
    meth = first(methods(obj.f))
    # Positional argument names (skip #self#)
    arg_names = [String(a) for a in Base.method_argnames(meth)[2:end]]
    args_str = join(arg_names, ", ")

    # Explicit globals/locals dicts avoid "SystemError: frame does not exist"
    factory = pybuiltins.eval("lambda fn: lambda $args_str: fn($args_str)",
                              pybuiltins.dict(), pybuiltins.dict())

    # Python passes Py objects; wrap them so the model body can use the
    # BFObject interface (m.dist.normal(...), arithmetic, indexing).
    wrapper(args...) = obj(args...)
    return factory(wrapper)
end

# Parameter names: `a = m.dist.normal(0, 1)` gets `name = "a"`, as in Python,
# where BayesForge infers the name from the assigned variable. Python cannot
# see Julia variable names, so the macro adds them when it expands.
_is_dist_call(f) = f isa Expr && f.head === :. && length(f.args) == 2 &&
                   f.args[1] isa Expr && f.args[1].head === :. &&
                   f.args[1].args[end] == QuoteNode(:dist)

function _has_name_kw(call::Expr)
    for a in call.args[2:end]
        a isa Expr || continue
        a.head === :kw && a.args[1] === :name && return true
        a.head === :parameters && any(p -> p isa Expr && p.head === :kw && p.args[1] === :name, a.args) && return true
    end
    return false
end

function _kw_value(call::Expr, key::Symbol)
    for a in call.args[2:end]
        a isa Expr || continue
        a.head === :kw && a.args[1] === key && return a.args[2]
        if a.head === :parameters
            for p in a.args
                p isa Expr && p.head === :kw && p.args[1] === key && return p.args[2]
            end
        end
    end
    return nothing
end

function _add_inferred_names!(ex)
    ex isa Expr || return ex
    if ex.head === :(=) && ex.args[1] isa Symbol && ex.args[2] isa Expr &&
       ex.args[2].head === :call && _is_dist_call(ex.args[2].args[1]) && !_has_name_kw(ex.args[2])
        push!(ex.args[2].args, Expr(:kw, :name, String(ex.args[1])))
    end
    # An observed site is named after the variable passed as obs (obs = y ->
    # "y"), "obs" for an expression; before, it defaulted to "x" in Python and
    # collided with a parameter called x. (Assignments were named just above.)
    if ex.head === :call && _is_dist_call(ex.args[1]) && !_has_name_kw(ex)
        obs = _kw_value(ex, :obs)
        obs === nothing || push!(ex.args, Expr(:kw, :name, obs isa Symbol ? String(obs) : "obs"))
    end
    foreach(_add_inferred_names!, ex.args)
    return ex
end

# 4. The Macro for Python Models
macro BF(ex)
    # Check validity
    if !isa(ex, Expr) || (ex.head != :function && ex.head != :(=))
        error("@BF must be used on a function definition")
    end

    call_expr = ex.args[1]
    if !isa(call_expr, Expr) || call_expr.head != :call
        error("Invalid function signature")
    end

    # 1. Capture the user's name (e.g., :model) from the ORIGINAL expression
    user_func_name = call_expr.args[1]

    # 2. Create a unique hidden name (e.g., ##model#291)
    internal_name = gensym(user_func_name)

    # 3. DEEPCOPY the expression before modifying it.
    #    Using `copy()` is shallow and corrupts the input AST, causing the crash.
    internal_def = deepcopy(ex)
    _add_inferred_names!(internal_def.args[2])

    # 4. Rename the function in the copy
    internal_def.args[1].args[1] = internal_name

    quote
        # Define the actual logic via the hidden name (escaped to exist in user scope)
        $(esc(internal_def))

        # Assign the Wrapper to the User's name
        # InspectableFunction is NOT escaped, so it refers to PyBayesForge.InspectableFunction
        $(esc(user_func_name)) = InspectableFunction($(esc(internal_name)))
    end
end

# -------------------------------------------------------
# Plotting Macro (@pyplot)
# -------------------------------------------------------
"""
    @pyplot expression

Executes a Python plotting command, intercepts plt.show()/plt.close(),
captures the figure, and displays it in Julia.
Uses a native Python lambda for the hook to avoid __signature__ errors.
"""
macro pyplot(ex)
    return quote
        local plt = PythonCall.pyimport("matplotlib.pyplot")
        local pybuiltins = PythonCall.pybuiltins

        # Save original functions
        local real_show = plt.show
        local real_close = plt.close

        # Define a Python No-Op lambda (Safe for introspection)
        # We pass empty dicts to avoid "SystemError: frame does not exist"
        local g = pybuiltins.dict()
        local l = pybuiltins.dict()
        local noop = pybuiltins.eval("lambda *args, **kwargs: None", g, l)

        # Inject No-Ops to prevent the figure from being closed/hidden
        plt.show = noop
        plt.close = noop

        local fig = nothing
        try
            # Run the user's plotting code
            $(esc(ex))

            # Grab the current figure (it should still be alive)
            fig = plt.gcf()

            # Check if it has content
            if length(fig.axes) > 0
                display(fig)
            else
                println("⚠️ No plot was generated (Figure is empty).")
            end
        catch e
            rethrow(e)
        finally
            # Restore original functions
            plt.show = real_show
            plt.close = real_close

            # Now we can safely close the figure to free memory
            if fig !== nothing
                real_close(fig)
            end
        end
    end
end

# Global Constants for jax.numpy and Python builtins
# 1. Define global constants
const jnp = BFObject(PythonCall.pynew())
const jax = BFObject(PythonCall.pynew())
const pybuiltins = PythonCall.pybuiltins
const pydict = PythonCall.pydict
const pylist = PythonCall.pylist

# 2. Initialize JAX modules when the module loads
function __init__()
    PythonCall.pycopy!(jnp.py, pyimport("jax.numpy"))
    PythonCall.pycopy!(jax.py, pyimport("jax"))
end


end

