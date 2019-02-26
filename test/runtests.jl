using NPZ, Compat

@static if VERSION >= v"0.7.0-DEV.2575"
    using Test
else
    using Base.Test
end

Debug = false

tmp = mktempdir()
if Debug
    println("temporary directory: $tmp")
end

TestArrays = Any[
    true,
    false,
    Int8(-42),
    Int16(-42),
    Int32(-42),
    Int64(-42),
    UInt8(42),
    UInt16(42),
    UInt32(42),
    UInt64(42),
    Float16(3.1415),
    Float32(3.1415),
    Float64(3.1415),
    Complex{Float32}(1, 7),
    Complex{Float64}(1, 7),
    Bool[0, 1, 0, 1, 1, 0],
    Int8[-42, 0, 1, 2, 3, 4],
    Int16[-42, 0, 1, 2, 3, 4],
    Int32[-42, 0, 1, 2, 3, 4],
    Int64[-42, 0, 1, 2, 3, 4],
    UInt8[0, 1, 2, 3, 4],
    UInt16[0, 1, 2, 3, 4],
    UInt32[0, 1, 2, 3, 4],
    UInt64[0, 1, 2, 3, 4],
    Float64[-42, 0, 1, 2, 3.14, 4],
    Float32[-42, 0, 1, 2, 3.14, 4],
    Float16[-42, 0, 1, 2, 3.14, 4],
    Complex{Float32}[1+2im, 3, 4+5im, 6im, 7+8im],
    Complex{Float64}[1+2im, 3, 4+5im, 6im, 7+8im],
    [i-j for i in 1:3, j in 1:5],
    [i-j+k for i in 1:3, j in 1:4, k in 1:5],
    rand(512, 512, 16)
]

# Write a NPZ file with all the test arrays and numbers,
# and read it back in.
old = Dict{String, Any}()
for (i, x) in enumerate(TestArrays)
    old["testvar_" * string(i)] = x
end
filename = "$tmp/big.npz"
npzwrite(filename, old)
output = npzread(filename)
for (k, v) in old
    @test v == output[k]
end
@test old == output

testdict = Dict("one"=>21, "two"=>34)
dictfilename = "$tmp/dict.npz"
npzwrite(dictfilename, testdict)
dictoutput = npzread(dictfilename)
for (k, v) in testdict
    @test v == dictoutput[k]
end
# Write and then read NPY files for each array
for x in TestArrays
    let filename = "$tmp/test.npy"
        npzwrite(filename, x)
        y = npzread(filename)
        @test typeof(x) == typeof(y)
        @test size(x) == size(y)
        @test x == y
    end
end

#v2 tests
v2_solution = Array{Float64,1}([0.1*i for i in 0:10])
v2_read = npzread("linspace.npy")
@test(typeof(v2_read) == typeof(v2_solution))
@test(size(  v2_read) == size(v2_solution))
@test(       v2_read  == v2_solution)

if !Debug
    run(`rm -rf $tmp`)
end
