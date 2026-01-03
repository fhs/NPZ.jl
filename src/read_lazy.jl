using ZipFile
using NPZ
import Base: size, eltype, getindex, close

export npzread_lazy

"""
    LazyNPZ

Struct holding the data of a lazily read .npz file
"""
struct LazyNPZ
    reader::ZipFile.Reader
    entries::Dict{String,ZipFile.ReadableFile}
    cache_f::Dict{String, Array}
    closed::Bool

    function LazyNPZ(reader::ZipFile.Reader, entries::Dict{String,ZipFile.ReadableFile})
        new(reader, entries, Dict{String,Array}(), false)
    end
end


function Base.show(io::IO, npz::LazyNPZ)
    println("LazyNPZ(files=$(keys(npz.entries)),loaded=$(keys(npz.cache_f)))")
end

function close(npz::LazyNPZ)
    if !npz.closed
        close(npz.reader)
        npz.closed = true
    end
end

"""
    npzread_lazy(filename)

Read an npz file lazily
"""
function npzread_lazy(filename::AbstractString)
    reader = ZipFile.Reader(filename)
    entries = Dict{String,ZipFile.ReadableFile}()

    for f in reader.files
        name = _maybetrimext(f.name)
        entries[name] = f
    end

    LazyNPZ(reader, entries)
end

function Base.getindex(npz::LazyNPZ, name::AbstractString)
    if !in(name,keys(npz.cache_f))
        if npz.closed
            error("Key `$name` not found. File is closed so no reads are possible")
        end
        if !in(name, keys(npz.entries))
            ("Array `$name` not found inside the archive")
        end
        f = npz.entries[name]
        arr = npzreadarray(f)
        npz.cache_f[name] = arr
    end
    npz.cache_f[name]
end

Base.keys(npz::LazyNPZ) = keys(npz.entries)
