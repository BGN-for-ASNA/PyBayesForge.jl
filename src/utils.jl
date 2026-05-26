module Utils

using CondaPkg

export create_env, delete_env, install_package, update_package

"""
    create_env()

Triggers CondaPkg to resolve and install the environment defined in CondaPkg.toml.
"""
function create_env()
    CondaPkg.resolve()
end

"""
    delete_env()

Removes the .CondaPkg directory, effectively deleting the environment.
"""
function delete_env()
    env_dir = joinpath(dirname(@__DIR__), ".CondaPkg")
    if isdir(env_dir)
        rm(env_dir, recursive=true, force=true)
        println("Environment deleted at $env_dir")
    else
        println("No environment found at $env_dir")
    end
end

"""
    install_package(pkg_name::String)

Installs a pip package into the current environment.
"""
function install_package(pkg_name::String)
    CondaPkg.add_pip(pkg_name)
end

"""
    update_package()

Updates the environment.
"""
function update_package()
    println("🔄 Checking for updates...")
    CondaPkg.update()
    
    println("\n📦 [BayesForge] Installation Details:")
    try
        # Run pip show within the specific Conda environment
        CondaPkg.withenv() do
            run(`pip show BayesForge`)
        end
    catch
        println("⚠️ Could not retrieve package details. Is 'BayesForge' installed?")
    end

    println("\n❗ IMPORTANT: If a new version was installed, you must RESTART your Jupyter Kernel or Julia Session for the changes to take effect.")
end

end
