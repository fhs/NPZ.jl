using NPZ
using Documenter

DocMeta.setdocmeta!(NPZ, :DocTestSetup, :(using NPZ); recursive=true)

makedocs(;
    modules=[NPZ],
    repo="https://github.com/fhs/NPZ.jl/blob/{commit}{path}#L{line}",
    sitename="NPZ",
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
