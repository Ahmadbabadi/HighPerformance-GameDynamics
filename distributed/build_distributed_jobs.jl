using DataFrames
using CSV

function build_jobs_csv(lattice_sizes, memories, synergy_rates, ensembles)

    jobs = DataFrame(
                    job_id = Int[],
                    ensemble = Int[],
                    lattice_size = Int[],
                    memory_length = Int[],
                    synergy_rate = Float64[]
                    )
        job_id = 0
        
    for L in lattice_sizes, m in memories, r in synergy_rates, ens in 1:ensembles

        push!(jobs, ( job_id, ens, L, m, r ))
        job_id += 1
    end

    CSV.write("configs/distributed_jobs.csv", jobs)

    return jobs
end

lattice_sizes = [1024]
memories = [15]
synergy_rates = range(start=2.9, step=0.1, stop=3.3)
ens = 50

jobs = build_jobs_csv(lattice_sizes, memories, synergy_rates, ens)

println("$(nrow(jobs)) jobs")