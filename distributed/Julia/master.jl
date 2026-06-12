using Distributed
addprocs(6)


@everywhere include("../../src/pgg_julia_opt.jl")
import .pgg_julia_opt


Features = [
    [1024, 1, 4.5, 1000],
    [1024, 3, 4.5, 1000],
    [1024, 5, 4.5, 1000],
    [1024, 8, 4.5, 1000],
    [1024, 1, 3.9, 1000],
    [1024, 3, 3.9, 1000],
    [1024, 5, 3.9, 1000],
    [1024, 8, 3.9, 1000]
]

@everywhere function run_simulation(params)
    pgg_julia_opt.pgg_main(
        Int(params[1]),
        Int(params[2]),
        params[3],
        Int(params[4]))
end


results = pmap(run_simulation, Features)