module NPZ

# NPZ file format is described in
# https://github.com/numpy/numpy/blob/v1.7.0/numpy/lib/format.py

export npzread, npzwrite

const Magic = Uint8[0x93, 'N', 'U', 'M', 'P', 'Y']
const Version = Uint8[1, 0]

const TypeMaps = [
	("b1", Bool),
	("i1", Int8),
	("i2", Int16),
	("i4", Int32),
	("i8", Int64),
	("u1", Uint8),
	("u2", Uint16),
	("u4", Uint32),
	("u8", Uint64),
	("f4", Float32),
	("f8", Float64),
	("c8", Complex64),
	("c16", Complex128),
]
Numpy2Julia = [s => t for (s, t) in TypeMaps]
Julia2Numpy = [t => s for (s, t) in TypeMaps]

readle(ios::IOStream, ::Type{Uint16}) = htol(read(ios, Uint16))

function writele(ios::IOStream, x::Vector{Uint8})
	n = write(ios, x)
	if n != length(x)
		error("short write")
	end
	n
end

writele(ios::IOStream, x::ASCIIString) = writele(ios, convert(Vector{Uint8}, x))
writele(ios::IOStream, x::Uint16) = writele(ios, reinterpret(Uint8, [htol(x)]))

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
	n = parseint(s[1:i-1])
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
		tup = [tup, n]
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
	descr :: (Function, DataType)
	fortran_order :: Bool
	shape :: Vector{Int}
end

function parseheader(s::ASCIIString)
	s = parsechar(s, '{')
	
	dict = (ASCIIString => Any)[]
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

function npzread(filename::String)
	f = open(filename)
	b = read(f, Uint8, length(Magic))
	if b != Magic
		error("not a numpy array file")
	end
	b = read(f, Uint8, length(Version))
	if b != Version
		error("unsupported NPZ version")
	end
	hdrlen = readle(f, Uint16)
	hdr = ascii(read(f, Uint8, hdrlen))
	hdr = parseheader(strip(hdr))
	
	toh, typ = hdr.descr
	if hdr.fortran_order
		x = map(toh, read(f, typ, hdr.shape...))
	else
		x = map(toh, read(f, typ, reverse(hdr.shape)...))
		if ndims(x) > 1
			x = permutedims(x, [ndims(x):-1:1])
		end
	end
	
	close(f)
	x
end

function npzwrite(filename::String, x::Array{Uint8}, T::DataType, shape::Vector{Int})
	if !haskey(Julia2Numpy, T)
		error("unsupported type $T")
	end
	f = open(filename, "w")
	writele(f, Magic)
	writele(f, Version)
	
	descr =  (ENDIAN_BOM == 0x01020304 ? ">" : "<") * Julia2Numpy[T]
	dict = "{'descr': '$descr', 'fortran_order': True, 'shape': $(tuple(shape...)), }"
	
	# The dictionary is padded with enough whitespace so that
	# the array data is 16-byte aligned
	n = length(Magic)+length(Version)+2+length(dict)
	pad = (div(n+16-1, 16)*16) - n
	if pad > 0
		dict *= " "^(pad-1) * "\n"
	end
	
	writele(f, uint16(length(dict)))
	writele(f, dict)
	write(f, x)
	close(f)
end

function npzwrite{T}(filename::String, x::Array{T})
	npzwrite(filename, reinterpret(Uint8, x[:]), T, [i for i in size(x)])
end

function npzwrite{T<:Number}(filename::String, x::T)
	npzwrite(filename, reinterpret(Uint8, [x]), T, Int[])
end


end	# module
