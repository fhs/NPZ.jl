using ZipArchives: ZipWriter, zip_newfile

"""
    npzwrite(filename::AbstractString, x)

Write the variable `x` to the `npy` file `filename`. 
Unlike `numpy`, the extension `.npy` is not appened to `filename`.

!!! warn "Warning"
    Any existing file with the same name will be overwritten.

# Examples

```julia
julia> npzwrite("abc.npy", zeros(3))

julia> npzread("abc.npy")
3-element Array{Float64,1}:
 0.0
 0.0
 0.0
```
"""
function npzwrite(filename::AbstractString, x)
    open(filename, "w") do f
        npzwritearray(f, x)
    end
end

"""
    npzwrite(filename::AbstractString, vars::Dict{<:AbstractString}; compress=false, compression_level=3)
    npzwrite(filename::AbstractString, args...; compress=false, compression_level=3, kwargs...)

In the first form, write the variables in `vars` to an `npz` file named `filename`.

In the second form, collect the variables in `args` and `kwargs` and write them all
to `filename`. The variables in `args` are saved with names `arr_0`, `arr_1` 
and so on, whereas the ones in `kwargs` are saved with the specified names.

Unlike `numpy`, the extension `.npz` is not appened to `filename`.

Use `compress=true` to write the file with Zip compression, at the level specified by `compression_level`.

!!! warn "Warning"
    Any existing file with the same name will be overwritten.

# Examples

```julia
julia> npzwrite("temp.npz", Dict("x" => ones(3), "y" => 3))

julia> npzread("temp.npz")
Dict{String,Any} with 2 entries:
  "x" => [1.0, 1.0, 1.0]
  "y" => 3

julia> npzwrite("temp.npz", ones(2,2), x = ones(3), y = 3)

julia> npzread("temp.npz")
Dict{String,Any} with 3 entries:
  "arr_0" => [1.0 1.0; 1.0 1.0]
  "x"     => [1.0, 1.0, 1.0]
  "y"     => 3
```
"""
function npzwrite(filename::AbstractString, vars::Dict{<:AbstractString}; compress=false, compression_level=3)
    ZipWriter(filename) do outf
        if length(vars) == 0
            @warn "no data to be written to $filename. It might not be possible to read the file correctly."
        end
        for (name,v) in vars
            # create new file
            zip_newfile(outf, name*".npy",compress=compress, compression_level=compression_level)
            # write the data
            npzwritearray(outf, v)
        end

    end
end

function npzwrite(filename::AbstractString, args...; compress=false,  compression_level=4, kwargs...)

    dkwargs = Dict(string(k) => v for (k,v) in kwargs)
    dargs = Dict("arr_"*string(i-1) => v for (i,v) in enumerate(args))

    d = convert(Dict{String,Any}, merge(dargs, dkwargs))

    npzwrite(filename, d; compress=compress, compression_level=compression_level)
end
