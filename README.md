## Overview

The NPZ package provides support for reading and writing Numpy .npy and
.npz files in Julia.

An .npy file contains a single numpy array, stored in a binary
format along with its shape, data type, etc. An .npz file contains a
collection numpy arrays each encoded in the .npy format and stored in a
ZIP file.  For more information, see the `numpy.save`, `numpy.savez`,
`numpy.savez_compressed`, and `numpy.load` functions in the [Numpy
documentation](http://docs.scipy.org/doc/numpy/reference/routines.io.html#npz-files).

[![CI](https://github.com/fhs/NPZ.jl/workflows/CI/badge.svg?branch=master&event=push)](https://github.com/fhs/NPZ.jl/actions?query=workflow%3ACI+branch%3Amaster+event%3Apush)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://fhs.github.io/NPZ.jl/dev)

## Installation

Install via the Julia package manager, `Pkg.add("NPZ")`.

## Usage

We try to provide an interface similar to the
[MAT.jl](https://github.com/simonster/MAT.jl) package.  Some usage
examples are shown below.

Write and then read back an array:

```julia
julia> using NPZ

julia> x = [i-j for i in 1:3, j in 1:4];

julia> npzwrite("x.npy", x)

julia> y = npzread("x.npy")
3x4 Int64 Array:
 0  -1  -2  -3
 1   0  -1  -2
 2   1   0  -1
```

Write and then read back a collection of arrays or numbers:

```julia
julia> npzwrite("data.npz", Dict("x" => x, "a" => collect(1:9), "n" => 42))

julia> vars = npzread("data.npz")
Dict{String,Any} with 3 entries:
  "x" => [0 -1 -2 -3; 1 0 -1 -2; 2 1 0 -1]
  "a" => [1, 2, 3, 4, 5, 6, 7, 8, 9]
  "n" => 42
```
