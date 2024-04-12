using NPZ
using Documenter

DocMeta.setdocmeta!(NPZ, :DocTestSetup, :(using NPZ); recursive=true)

makedocs(;
    modules=[NPZ],
    repo=Remotes.GitHub("fhs", "NPZ.jl"),
    sitename="NPZ.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://fhs.github.io/NPZ.jl",
        assets=String[],
    ),
    pages=[
        "Reference" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/fhs/NPZ.jl",
)
