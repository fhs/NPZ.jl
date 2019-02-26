__precompile__()
module NPZ

# NPZ file format is described in
# https://github.com/numpy/numpy/blob/v1.7.0/numpy/lib/format.py

using ZipFile, Compat

@static if VERSION >=  v"0.7.0-DEV.2575"
    import Base.CodeUnits
else
    # CodeUnits not yet supported by Compat but not needed in julia 0.6...
    # codeunits function in Compat returns uintX instead of codeunits
    # therefore this 'stump' type should work
    abstract type CodeUnits{U, S} end
end

export npzread, npzwrite

const NPYMagic = UInt8[0x93, 'N', 'U', 'M', 'P', 'Y']
const ZIPMagic = UInt8['P', 'K', 3, 4]
const Version = UInt8[1, 0]

const MaxMagicLen = maximum(length.([NPYMagic, ZIPMagic]))

const TypeMaps = [
    ("b1", Bool),
    ("i1", Int8),
    ("i2", Int16),
    ("i4", Int32),
    ("i8", Int64),
    ("u1", UInt8),
    ("u2", UInt16),
    ("u4", UInt32),
    ("u8", UInt64),
    ("f2", Float16),
    ("f4", Float32),
    ("f8", Float64),
    ("c8", Complex{Float32}),
    ("c16", Complex{Float64}),
]
const Numpy2Julia = Dict{String, DataType}()
for (s,t) in TypeMaps
    Numpy2Julia[s] = t
end

const Julia2Numpy = Dict{DataType, String}()

@static if VERSION >= v"0.4.0"
    function __init__()
        for (s,t) in TypeMaps
            Julia2Numpy[t] = s
        end
    end
else
    for (s,t) in TypeMaps
        Julia2Numpy[t] = s
    end
end

# Julia2Numpy is a dictionary that uses Types as keys.
# This is problematic for precompilation because the
# hash of a Type changes everytime Julia is run.
# The hash of the keys when NPZ is precompiled will
# not be the same as when it is later run. This can
# be fixed by rehashing the Dict when the module is
# loaded.

readle(ios::IO, ::Type{T}) where T = ltoh(read(ios, T)) # ltoh is inverse of htol

function writecheck(io::IO, x::Any)
    n = write(io, x) # returns size in bytes
    n == sizeof(x) || error("short write") # sizeof is size in bytes
end

# Endianness only pertains to multi-byte things
writele(ios::IO, x::AbstractVector{UInt8}) = writecheck(ios, x)
writele(ios::IO, x::AbstractVector{CodeUnits{UInt8, <:Any}}) = writecheck(ios, x)
# codeunits returns vector of CodeUnits in 7+, uint in 6
writele(ios::IO, x::AbstractString) = writele(ios, codeunits(x))

writele(ios::IO, x::UInt16) = writecheck(ios, htol(x))

function parsechar(s::AbstractString, c::Char)
    firstchar = s[firstindex(s)]
    if  firstchar != c
        error("parsing header failed: expected character '$c', found '$firstchar'")
    end
    SubString(s, nextind(s, 1))
end

function parsestring(s::AbstractString)
    s = parsechar(s, '\'')
    parts = split(s, '\'', limit = 2)
    length(parts) != 2 && error("parsing header failed: malformed string")
    parts[1], parts[2]
end

function parsebool(s::AbstractString)
    if SubString(s, firstindex(s), thisind(s, 4)) == "True"
        return true, SubString(s, nextind(s, 4))
    elseif SubString(s, firstindex(s), thisind(s, 5)) == "False"
        return false, SubString(s, nextind(s, 5))
    end
    error("parsing header failed: excepted True or False")
end

function parseinteger(s::AbstractString)
    isdigit(s[firstindex(s)]) || error("parsing header failed: no digits")
    tail_idx = findfirst(c -> !isdigit(c), s)
    if tail_idx == nothing
        intstr = SubString(s, firstindex(s))
    else
        intstr = SubString(s, firstindex(s), prevind(s, tail_idx))
        if s[tail_idx] == 'L' # output of firstindex should be a valid code point
            tail_idx = nextind(s, i)
        end
    end
    n = parse(Int, intstr)
    return n, SubString(s, tail_idx)
end

function parsetuple(s::AbstractString)
    s = parsechar(s, '(')
    tup = Int[]
    while true
        s = strip(s)
        if s[firstindex(s)] == ')'
            break
        end
        n, s = parseinteger(s)
        push!(tup, n)
        s = strip(s)
        if s[firstindex(s)] == ')'
            break
        end
        s = parsechar(s, ',')
    end
    s = parsechar(s, ')')
    tup, s
end

function parsedtype(s::AbstractString)
    dtype, s = parsestring(s)
    c = dtype[firstindex(s)]
    t = SubString(dtype, nextind(s, 1))
    if c == '<'
        toh = ltoh
    elseif c == '>'
        toh = ntoh
    elseif c == '|'
        toh = identity
    else
        error("parsing header failed: unsupported endian character $c")
    end
    if !haskey(Numpy2Julia, t)
        error("parsing header failed: unsupported type $t")
    end
    (toh, Numpy2Julia[t]), s
end

struct Header
    descr::Tuple{Function, DataType}
    fortran_order::Bool
    shape::Vector{Int}
end

function parseheader(s::AbstractString)
    s = parsechar(s, '{')

    dict = Dict{String,Any}()
    for _ in 1:3
        s = strip(s)
        key, s = parsestring(s)
        s = strip(s)
        s = parsechar(s, ':')
        s = strip(s)
        if key == "descr"
            dict[key], s = parsedtype(s)
        elseif key == "fortran_order"
            dict[key], s = parsebool(s)
        elseif key == "shape"
            dict[key], s = parsetuple(s)
        else
            error("parsing header failed: bad dictionary key")
        end
        s = strip(s)
        if s[firstindex(s)] == '}'
            break
        end
        s = parsechar(s, ',')
    end
    s = strip(s)
    s = parsechar(s, '}')
    s = strip(s)
    if s != ""
        error("malformed header")
    end
    Header(dict["descr"], dict["fortran_order"], dict["shape"])
end

function npzreadarray(f::IO)
    @compat b = read!(f, Vector{UInt8}(undef, length(NPYMagic)))
    if b != NPYMagic
        error("not a numpy array file")
    end
    @compat b = read!(f, Vector{UInt8}(undef, length(Version)))

    # support for version 2 files
    if b[1] == 1
        hdrlen = readle(f, UInt16)
    elseif b[1] == 2 
        hdrlen = readle(f, UInt32)
    else
        error("unsupported NPZ version")
    end

    @compat hdr = ascii(String(read!(f, Vector{UInt8}(undef, hdrlen))))
    hdr = parseheader(strip(hdr))

    toh, typ = hdr.descr
    if hdr.fortran_order
        @compat x = map(toh, read!(f, Array{typ}(undef, hdr.shape...)))
    else
        @compat x = map(toh, read!(f, Array{typ}(undef, reverse(hdr.shape)...)))
        if ndims(x) > 1
            x = permutedims(x, collect(ndims(x):-1:1))
        end
    end
    x isa Array{<:Any, 0} ? x[1] : x
end

function samestart(a::AbstractVector, b::AbstractVector)
    nb = length(b)
    length(a) >= nb && view(a, 1:nb) == b
end

function npzread(filename::AbstractString)
    # Detect if the file is a numpy npy array file or a npz/zip file.
    f = open(filename)
    @compat b = read!(f, Vector{UInt8}(undef, MaxMagicLen))
    if samestart(b, ZIPMagic)
        close(f)
        f = ZipFile.Reader(filename)
        data = npzread(f)
        close(f)
        return data
    end
    if samestart(b, NPYMagic)
        seekstart(f)
        data = npzreadarray(f)
        close(f)
        return data
    end
    error("not a NPY or NPZ/Zip file: $filename")
end

function npzread(dir::ZipFile.Reader)
    vars = Dict{String,Any}()
    for f in dir.files
        name = f.name
        if endswith(name, ".npy")
            name = name[1:end-4]
        end
        vars[name] = npzreadarray(f)
    end
    vars
end

function npzwritearray(
    f::IO, x::AbstractArray{UInt8}, T::DataType, shape::Vector{Int}
)
    if !haskey(Julia2Numpy, T)
        error("unsupported type $T")
    end
    writele(f, NPYMagic)
    writele(f, Version)

    descr =  (ENDIAN_BOM == 0x01020304 ? ">" : "<") * Julia2Numpy[T]
    dict = "{'descr': '$descr', 'fortran_order': True, 'shape': $(tuple(shape...)), }"

    # The dictionary is padded with enough whitespace so that
    # the array data is 16-byte aligned
    n = length(NPYMagic)+length(Version)+2+length(dict)
    pad = (div(n+16-1, 16)*16) - n
    if pad > 0
        dict *= " "^(pad-1) * "\n"
    end

    writele(f, UInt16(length(dict)))
    writele(f, dict)
    if write(f, x) != length(x)
        error("short write")
    end
end

function npzwritearray(f::IO, x::AbstractArray{T}) where T
    npzwritearray(f, reinterpret(UInt8, x[:]), T, [i for i in size(x)])
end

function npzwritearray(f::IO, x::T) where T<:Number
    npzwritearray(f, reinterpret(UInt8, [x]), T, Int[])
end
function npzwrite(filename::AbstractString, x)
    f = open(filename, "w")
    npzwritearray(f, x)
    close(f)
end

function npzwrite(filename::AbstractString, vars::Dict{<:AbstractString}) 
    dir = ZipFile.Writer(filename)
    for (k, v) in vars
        f = ZipFile.addfile(dir, k * ".npy")
        npzwritearray(f, v)
        close(f)
    end
    close(dir)
end

end # module
