__precompile__()
module NPZ

# NPZ file format is described in
# https://github.com/numpy/numpy/blob/v1.7.0/numpy/lib/format.py

using ZipFile, Compat

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

readle(ios::IO, ::Type{T}) where T = ltoh(read(ios, T)) #ltoh is inverse of htol

function writecheck(io::IO, x::Any)
    n = write(io, x) # returns size in bytes
    n == sizeof(x) || error("short write") # sizeof is size in bytes
end

# Endianness only pertains to multi-byte things
writele(ios::IO, x::AbstractVector{UInt8}) = writecheck(ios, x)
# NPY headers only have ascii strings (bytes) so endianness doesn't matter
writele(ios::IO, x::AbstractString) = writecheck(ios, codeunits(x))

writele(ios::IO, x::UInt16) = writecheck(ios, htol(x))

function parsechar(s::AbstractString, c::Char)
    if s[1] != c
        error("parsing header failed: expected character '$c', found '$(s[1])'")
    end
    s[2:end]
end

function parsestring(s::AbstractString)
    s = parsechar(s, '\'')
    i = something(findfirst(isequal('\''), s), 0)
    if i == 0
        error("parsing header failed: malformed string")
    end
    s[1:i-1], s[i+1:end]
end

function parsebool(s::AbstractString)
    if s[1:4] == "True"
        return true, s[5:end]
    elseif s[1:5] == "False"
        return false, s[6:end]
    end
    error("parsing header failed: excepted True or False")
end

function parseinteger(s::AbstractString)
    i = something(findfirst(c -> !isdigit(c), s), 0)
    n = parse(Int, s[1:i-1])
    if s[i] == 'L'
        i += 1
    end
    n, s[i:end]
end

function parsetuple(s::AbstractString)
    s = parsechar(s, '(')
    tup = Int[]
    while true
        s = strip(s)
        if s[1] == ')'
            break
        end
        n, s = parseinteger(s)
        tup = [tup; n]
        s = strip(s)
        if s[1] == ')'
            break
        end
        s = parsechar(s, ',')
    end
    s = parsechar(s, ')')
    tup, s
end

function parsedtype(s::AbstractString)
    dtype, s = parsestring(s)
    c, t = dtype[1], dtype[2:end]
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
        if s[1] == '}'
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
    if b != Version
        error("unsupported NPZ version")
    end
    hdrlen = readle(f, UInt16)
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

function npzwrite(
    filename::AbstractString, vars::Dict{S,Any}
) where S<:AbstractString
    dir = ZipFile.Writer(filename)
    for (k, v) in vars
        f = ZipFile.addfile(dir, k * ".npy")
        npzwritearray(f, v)
        close(f)
    end
    close(dir)
end

end # module
