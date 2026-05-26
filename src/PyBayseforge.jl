module PyBayseforge

using PythonCall
using CondaPkg

# Check if utils.jl exists in your folder, otherwise comment this out
if isfile(joinpath(@__DIR__, "utils.jl"))
    include("utils.jl")
    using .Utils
    export create_env, delete_env, install_package, update_package
end

export importBF, @BF, @pyplot, jnp, jax, pybuiltins, pydict, pylist

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

    # Unwrap value if it's a BFObject
    val = v isa BFObject ? v.py : v

    setproperty!(x.py, s, val)
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
    # Unwrap args
    new_args = map(a -> a isa BFObject ? a.py : a, args)
    new_kwargs = Dict{Symbol,Any}()
    for (k, v) in kwargs
        new_kwargs[k] = v isa BFObject ? v.py : v
    end

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

Initializes and returns the Python `BI` object from the `BayesForge` package.
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
#    This is called automatically when you pass the object to Python
function PythonCall.Py(obj::InspectableFunction)
    meth = first(methods(obj.f))

    # Introspection: Get arg names (skip #self#)
    # Note: This only reliably works for POSITIONAL arguments.
    arg_names = [String(a) for a in Base.method_argnames(meth)[2:end]]
    args_str = join(arg_names, ", ")

    # Create the lambda factory code
    # We wrap the arguments in BFObject before passing them to the Julia function
    lambda_code = "lambda fn: lambda $args_str: fn(*[BFObject(a) for a in ($args_str,)])"

    # FIX: Provide explicit globals/locals dictionaries to avoid SystemError
    # We use pybuiltins.dict() to create empty Python dicts
    globals_dict = pybuiltins.dict()
    locals_dict = pybuiltins.dict()

    # Inject BFObject into the globals so the lambda can use it?
    # No, the lambda runs in Python, it calls `fn` (which is the Julia function).
    # `fn` expects Julia objects.
    # Wait, `fn` is `obj.f`.
    # When Python calls `fn`, PythonCall converts Python args to Julia args.
    # By default, PythonCall converts Python objects to `Py` (or `PyArray` etc).
    # We want them to be `BFObject`.
    # So we need to intercept the call to `fn`.

    # Let's wrap `obj.f` in a Julia closure that wraps args.
    wrapped_f = (args...; kwargs...) -> begin
        new_args = map(a -> a isa PyWrappers ? BFObject(a) : a, args)
        # Also handle kwargs if needed, but PythonCall passes kwargs as...
        obj.f(new_args...; kwargs...)
    end

    # But `PythonCall.Py(obj::InspectableFunction)` returns a Python object factory.
    # The factory takes `obj.f`.
    # If we pass `wrapped_f` to the factory, it should work.

    # Re-evaluating the lambda code:
    lambda_code = "lambda fn: lambda $args_str: fn($args_str)"

    globals_dict = pybuiltins.dict()
    locals_dict = pybuiltins.dict()
    factory = pybuiltins.eval(lambda_code, globals_dict, locals_dict)

    # Wrap the function to convert Py args to BFObject
    function wrapper(args...; kwargs...)
        new_args = map(a -> a isa PyWrappers ? BFObject(a) : a, args)
        # kwargs keys are Symbols, values are... Py?
        # PythonCall converts kwargs values.
        new_kwargs = kwargs # Simplified for now
        obj.f(new_args...; new_kwargs...)
    end

    return factory(wrapper)
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

    # 4. Rename the function in the copy
    internal_def.args[1].args[1] = internal_name

    quote
        # Define the actual logic via the hidden name (escaped to exist in user scope)
        $(esc(internal_def))

        # Assign the Wrapper to the User's name
        # InspectableFunction is NOT escaped, so it refers to PyBayseforge.InspectableFunction
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

