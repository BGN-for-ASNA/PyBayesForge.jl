using Pkg
Pkg.activate(".")  # Activate the current project environment
using PyBayesForge

# Setup device------------------------------------------------
m = importBF(platform="cpu")

# Import Data & Data Manipulation ------------------------------------------------
# Import
data_path = m.load.howell1(only_path=true)
m.data(data_path, sep=';')
m.df = m.df[m.df.age>18] # Subset data to adults
m.scale(["weight"]) # Normalize

# Define model ------------------------------------------------
@BF function model(weight, height)
    # Priors
    a = m.dist.normal(178, 20, name='a')
    b = m.dist.log_normal(0, 1, name='b')
    s = m.dist.uniform(0, 50, name='s')
    m.dist.normal(a + b * weight, s, obs=height)
end

# Run mcmc ------------------------------------------------
m.fit(model)  # Optimize model parameters through MCMC sampling

# Summary ------------------------------------------------
m.summary() # Get posterior distributions