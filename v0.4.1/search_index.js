var documenterSearchIndex = {"docs":
[{"location":"","page":"Reference","title":"Reference","text":"CurrentModule = NPZ","category":"page"},{"location":"#NPZ.jl","page":"Reference","title":"NPZ.jl","text":"","category":"section"},{"location":"","page":"Reference","title":"Reference","text":"Modules = [NPZ]","category":"page"},{"location":"#NPZ.npzread-Tuple{AbstractString,Vararg{Any,N} where N}","page":"Reference","title":"NPZ.npzread","text":"npzread(filename::AbstractString, [vars])\n\nRead a variable or a collection of variables from filename.  The input needs to be either an npy or an npz file. The optional argument vars is used only for npz files. If it is specified, only the matching variables are read in from the file.\n\nnote: Zero-dimensional arrays\nZero-dimensional arrays are stripped while being read in, and the values that they contain are returned. This is a notable difference from numpy, where  numerical values are written out and read back in as zero-dimensional arrays.\n\nExamples\n\njulia> npzwrite(\"temp.npz\", x = ones(3), y = 3)\n\njulia> npzread(\"temp.npz\") # Reads all variables\nDict{String,Any} with 2 entries:\n  \"x\" => [1.0, 1.0, 1.0]\n  \"y\" => 3\n\njulia> npzread(\"temp.npz\", [\"x\"]) # Reads only \"x\"\nDict{String,Array{Float64,1}} with 1 entry:\n  \"x\" => [1.0, 1.0, 1.0]\n\n\n\n\n\n","category":"method"},{"location":"#NPZ.npzwrite-Tuple{AbstractString,Any}","page":"Reference","title":"NPZ.npzwrite","text":"npzwrite(filename::AbstractString, x)\n\nWrite the variable x to the npy file filename.  Unlike numpy, the extension .npy is not appened to filename.\n\nwarn: Warning\nAny existing file with the same name will be overwritten.\n\nExamples\n\njulia> npzwrite(\"abc.npy\", zeros(3))\n\njulia> npzread(\"abc.npy\")\n3-element Array{Float64,1}:\n 0.0\n 0.0\n 0.0\n\n\n\n\n\n","category":"method"},{"location":"#NPZ.npzwrite-Tuple{AbstractString,Dict{#s17,V} where V where #s17<:AbstractString}","page":"Reference","title":"NPZ.npzwrite","text":"npzwrite(filename::AbstractString, vars::Dict{<:AbstractString})\nnpzwrite(filename::AbstractString, args...; kwargs...)\n\nIn the first form, write the variables in vars to an npz file named filename.\n\nIn the second form, collect the variables in args and kwargs and write them all to filename. The variables in args are saved with names arr_0, arr_1  and so on, whereas the ones in kwargs are saved with the specified names.\n\nUnlike numpy, the extension .npz is not appened to filename.\n\nwarn: Warning\nAny existing file with the same name will be overwritten.\n\nExamples\n\njulia> npzwrite(\"temp.npz\", Dict(\"x\" => ones(3), \"y\" => 3))\n\njulia> npzread(\"temp.npz\")\nDict{String,Any} with 2 entries:\n  \"x\" => [1.0, 1.0, 1.0]\n  \"y\" => 3\n\njulia> npzwrite(\"temp.npz\", ones(2,2), x = ones(3), y = 3)\n\njulia> npzread(\"temp.npz\")\nDict{String,Any} with 3 entries:\n  \"arr_0\" => [1.0 1.0; 1.0 1.0]\n  \"x\"     => [1.0, 1.0, 1.0]\n  \"y\"     => 3\n\n\n\n\n\n","category":"method"},{"location":"#NPZ.readheader-Tuple{AbstractString,Vararg{Any,N} where N}","page":"Reference","title":"NPZ.readheader","text":"readheader(filename, [vars...])\n\nReturn a header or a collection of headers corresponding to each variable contained in filename.  The header contains information about the eltype and size of the array that may be extracted using  the corresponding accessor functions.\n\n\n\n\n\n","category":"method"}]
}