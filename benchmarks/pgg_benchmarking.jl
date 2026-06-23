using BenchmarkTools
using Statistics
using DataFrames
using CSV

include(joinpath(@__DIR__, "..", "src", "pgg_julia.jl"))
include(joinpath(@__DIR__, "..", "src", "pgg_julia_opt.jl"))

using .pgg_julia
using .pgg_julia_opt


Features = [
    [32, 15, 3, 1000],
    [64, 15, 3, 1000],
    [128, 15, 3, 1000],
    [256, 15, 3, 1000],
    [512, 15, 3, 1000],
    [1024, 15, 3, 1000],
    [2048, 15, 3, 1000],
]


versions_func = [pgg_julia.run_simulation, pgg_julia_opt.run_simulation]
versions_name = ["julia.csv", "julia_opt.csv"]

output_dir = joinpath(@__DIR__, "..", "data", "raw")


for (func, name) in zip(versions_func, versions_name)
    results = DataFrame(
        L = Int[],
        M = Int[],
        Sr = Float64[],
        mean_ms = Float64[],
        median_ms = Float64[],
        std_ms = Float64[],
        min_ms = Float64[],
        max_ms = Float64[],
        memory_bytes = Int[],
        allocations = Int[]
        )

    for (L, M, Sr, rounds) in Features
        
        # func(L, M, Sr, rounds)
        
        b = @benchmark ($func)($L, $M, $Sr, $rounds)
        
        push!(results, (L, M, Sr,
        mean(b.times)/1e6,
        median(b.times)/1e6,
        std(b.times)/1e6,
        minimum(b.times)/1e6,
        maximum(b.times)/1e6,
        minimum(b).memory,
        minimum(b).allocs
        ))
    end
    result_path = joinpath(output_dir, name)
    CSV.write(result_path, results)
    println("Successfully saved: $result_path")
end