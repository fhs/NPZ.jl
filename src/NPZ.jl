module NPZ

# NPZ file format is described in
# https://github.com/numpy/numpy/blob/v1.7.0/numpy/lib/format.py

using ZipFile
using Compat
import Compat: UTF8String, ASCIIString

export npzread, npzwrite

const NPYMagic = UInt8[0x93, 'N', 'U', 'M', 'P', 'Y']
const ZIPMagic = UInt8['P', 'K', 3, 4]
const Version = UInt8[1, 0]

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
	("f4", Float32),
	("f8", Float64),
	("c8", Complex64),
	("c16", Complex128),
]
Numpy2Julia = Dict(s => t for (s, t) in TypeMaps)
Julia2Numpy = Dict(t => s for (s, t) in TypeMaps)

# Julia2Numpy is a dictionary that uses Types as keys.
# This is problematic for precompilation because the
# hash of a Type changes everytime Julia is run.
# The hash of the keys when NPZ is precompiled will
# not be the same as when it is later run. This can
# be fixed by rehashing the Dict when the module is
# loaded.

if VERSION >= v"0.4.0" 
	__init__() = Base.rehash!(Julia2Numpy)
end

readle(ios::IO, ::Type{UInt16}) = htol(read(ios, UInt16))

function writele(ios::IO, x::Vector{UInt8})
	n = write(ios, x)
	if n != length(x)
		error("short write")
	end
	n
end

writele(ios::IO, x::ASCIIString) = writele(ios, convert(Vector{UInt8}, x))
writele(ios::IO, x::UInt16) = writele(ios, reinterpret(UInt8, [htol(x)]))

function parsechar(s::ASCIIString, c::Char)
	if s[1] != c
		error("parsing header failed: expected character '$c', found '$(s[1])'")
	end
	s[2:end]
end

function parsestring(s::ASCIIString)
	s = parsechar(s, '\'')
	i = findfirst(s, '\'')
	if i == 0
		error("parsing header failed: malformed string")
	end
	s[1:i-1], s[i+1:end]
end

function parsebool(s::ASCIIString)
	if s[1:4] == "True"
		return true, s[5:end]
	elseif s[1:5] == "False"
		return false, s[6:end]
	end
	error("parsing header failed: excepted True or False")
end

function parseinteger(s::ASCIIString)
	i = findfirst(c -> !isdigit(c), s)
	n = parse(Int, s[1:i-1])
	if s[i] == 'L'
		i += 1
	end
	n, s[i:end]
end

function parsetuple(s::ASCIIString)
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

function parsedtype(s::ASCIIString)
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

type Header
	descr :: @compat(Tuple{Function, DataType})
	fortran_order :: Bool
	shape :: Vector{Int}
end

function parseheader(s::ASCIIString)
	s = parsechar(s, '{')
	
	dict = @compat Dict{ASCIIString,Any}()
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
	b = read(f, UInt8, length(NPYMagic))
	if b != NPYMagic
		error("not a numpy array file")
	end
	b = read(f, UInt8, length(Version))
	if b != Version
		error("unsupported NPZ version")
	end
	hdrlen = readle(f, UInt16)
	hdr = ascii(UTF8String(read(f, UInt8, hdrlen)))
	hdr = parseheader(strip(hdr))
	
	toh, typ = hdr.descr
	if hdr.fortran_order
		x = map(toh, read(f, typ, hdr.shape...))
	else
		x = map(toh, read(f, typ, reverse(hdr.shape)...))
		if ndims(x) > 1
			x = permutedims(x, collect(ndims(x):-1:1))
		end
	end
	x
end

function npzread(filename::AbstractString)
	# Detect if the file is a numpy npy array file or a npz/zip file.
	f = open(filename)
	b = read(f, UInt8, max(length(NPYMagic), length(ZIPMagic)))
	if startswith(b, ZIPMagic)
		close(f)
		f = ZipFile.Reader(filename)
		data = npzread(f)
		close(f)
		return data
	end
	if startswith(b, NPYMagic)
		seekstart(f)
		data = npzreadarray(f)
		close(f)
		return data
	end
	error("not a NPY or NPZ/Zip file: $filename")
end

function npzread(dir::ZipFile.Reader)
	vars = Dict{UTF8String,Any}()
	for f in dir.files
		name = f.name
		if endswith(name, ".npy")
			name = name[1:end-4]
		end
		vars[name] = npzreadarray(f)
	end
	vars
end

function npzwritearray(f::IO, x::Array{UInt8}, T::DataType, shape::Vector{Int})
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
	
	writele(f, @compat UInt16(length(dict)))
	writele(f, dict)
	if write(f, x) != length(x)
		error("short write")
	end
end

function npzwritearray{T}(f::IO, x::Array{T})
	npzwritearray(f, reinterpret(UInt8, x[:]), T, [i for i in size(x)])
end

function npzwritearray{T<:Number}(f::IO, x::T)
	npzwritearray(f, reinterpret(UInt8, [x]), T, Int[])
end

function npzwrite(filename::AbstractString, x)
	f = open(filename, "w")
	npzwritearray(f, x)
	close(f)
end

function npzwrite{S<:AbstractString}(filename::AbstractString, vars::Dict{S,Any})
	dir = ZipFile.Writer(filename)
	for (k, v) in vars
		f = ZipFile.addfile(dir, k * ".npy")
		npzwritearray(f, v)
		close(f)
	end
	close(dir)
end

end	# module
