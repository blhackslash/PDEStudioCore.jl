using Documenter
using PDEStudioCore

makedocs(
    sitename = "PDEStudioCore.jl",
    modules = [PDEStudioCore],
    checkdocs = :exports,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://blhackslash.github.io/PDEStudioCore.jl/",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Simulation Config" => "simulation_config.md",
        "Advanced Config" => "advanced_config.md",
        "Data & Storage" => "data.md",
        "Statistics" => "statistics.md",
        "Migration Guide" => "migration.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/blhackslash/PDEStudioCore.jl.git",
    devbranch = "main",
    push_preview = true,
)
