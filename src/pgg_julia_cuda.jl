using CUDA
using Random
using Statistics

function build_units(lattice_size)
    L = Int32(lattice_size)
    units = zeros(Int32, L^2, 5)
    for i in axes(units, 1)
        x = mod1(i, L)
        y = cld(i, L)
        units[i, 1] = i
        units[i, 2] = (y - 1) * L + mod1(x - 1, L)
        units[i, 3] = (y - 1) * L + mod1(x + 1, L)
        units[i, 4] = (mod1(y - 1, L) - 1) * L + x
        units[i, 5] = (mod1(y + 1, L) - 1) * L + x
    end
    return units
end


@inline function payoff_calculator_gpu(unit_start, units, strategies, synergy_rate, cost)
    cooperators  = Int32(0)
    participants = Int32(0)
    for k in 1:5
        S = strategies[units[unit_start + k]]   
        if S == Int8(1)
            cooperators  += Int32(1)
            participants += Int32(1)
        elseif S == Int8(2)
            participants += Int32(1)
        end
    end
    if participants > Int32(1)
        c = cost * synergy_rate * cooperators / participants
        d = c + cost
        l = cost
        return c, d, l
    end
    return cost, cost, cost
end


function kernel_payoffs!(cumulative_payoffs, strategies, units_flat, synergy_rate, cost, N)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    p > N && return nothing
    s_p   = strategies[p]
    total = 0.0f0

    for gi in 1:5 
        g = units_flat[(p - 1) * 5 + gi]  
        c, d, l = payoff_calculator_gpu((g - 1) * 5, units_flat, strategies, synergy_rate, cost)
        total += s_p == Int8(1) ? c : s_p == Int8(2) ? d : l
    end

    cumulative_payoffs[p] = total
    return nothing
end


function kernel_update_memory!(memories, strategies, cumulative_payoffs, memory_pointer, memory_length, N)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    p > N && return nothing

    s = strategies[p]
    idx = (memory_pointer - 1) + (Int32(s) - 1) * memory_length + (p - 1) * memory_length * 3
    memories[idx + 1] = cumulative_payoffs[p]
    return nothing
end

@inline function gpu_rand32(seed::UInt32)
    x = seed
    x ⊻= x << 13
    x ⊻= x >> 17
    x ⊻= x << 5
    return x
end

function kernel_pick_strategy!(new_strategies, strategies, memories, units_flat,
                                beta, memory_length, N, r)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    p > N && return nothing

    seed = UInt32(p) + UInt32(r) * UInt32(1103515245)
    rng_val = gpu_rand32(seed)

    nb_col   = 1 + (rng_val & UInt32(3))
    neighbor = units_flat[(p - 1) * 5 + Int32(nb_col)]

    s_p = strategies[p]
    s_n = strategies[neighbor]

    if s_p != s_n
        base_p = (p - 1) * memory_length * 3 + (Int32(s_p) - 1) * memory_length
        mem_p  = 0.0f0
        for t in 1:memory_length
            mem_p += memories[base_p + t]
        end
        mem_p /= memory_length

        base_n = (neighbor - 1) * memory_length * 3 + (Int32(s_n) - 1) * memory_length
        mem_n  = 0.0f0
        for t in 1:memory_length
            mem_n += memories[base_n + t]
        end
        mem_n /= memory_length

        weight = 1.0f0 / (1.0f0 + exp(beta * (mem_p - mem_n)))
        
        rng_val2 = gpu_rand32(rng_val)
        rand_float = Float32(rng_val2) / Float32(typemax(UInt32))

        new_strategies[p] = rand_float < weight ? s_n : s_p
    else
        new_strategies[p] = s_p
    end
    return nothing
end


function pgg_init_gpu(lattice_size, memory_length, cooperator_rate=1/3f0, defector_rate=1/3f0)
    N = lattice_size^2

    units_cpu  = build_units(lattice_size)
    units_flat = CuArray(vec(units_cpu'))

    thresholds = Float32[cooperator_rate, cooperator_rate + defector_rate]
    r = rand(Float32, N)
    strats_cpu = Int8.(ifelse.(r .< thresholds[1], 1, ifelse.(r .< thresholds[2], 2, 3)))
    strategies = CuArray(strats_cpu)

    memories = CUDA.ones(Float32, memory_length * 3 * N)

    return units_flat, strategies, memories
end


function useful_data_gpu(strategies)
    N = length(strategies)
    c = count(==(Int8(1)), strategies)
    d = count(==(Int8(2)), strategies)
    return Float64(c) / N, Float64(d) / N
end


function run_simulation(lattice_size, memory_length, synergy_rate, rounds,
                        beta=10f0, cost=1f0/5f0,
                        cooperator_rate=1f0/3f0, defector_rate=1f0/3f0;
                        threads_per_block=256)
    N = lattice_size^2
    data = zeros(rounds + 1, 2)
    memory_pointer = Int32(1)

    units_flat, strategies, memories = pgg_init_gpu(
        lattice_size, memory_length, cooperator_rate, defector_rate)

    cumulative_payoffs = CUDA.zeros(Float32, N)
    new_strategies     = similar(strategies)

    blocks = cld(N, threads_per_block)

    data[1, :] .= useful_data_gpu(strategies)

    for r in 2:rounds+1
        @cuda threads=threads_per_block blocks=blocks kernel_payoffs!(
            cumulative_payoffs, strategies, units_flat,
            Float32(synergy_rate), Float32(cost), Int32(N))

        @cuda threads=threads_per_block blocks=blocks kernel_update_memory!(
            memories, strategies, cumulative_payoffs,
            memory_pointer, Int32(memory_length), Int32(N))

        memory_pointer = mod1(memory_pointer + Int32(1), Int32(memory_length))

        @cuda threads=threads_per_block blocks=blocks kernel_pick_strategy!(
            new_strategies, strategies, memories, units_flat,
            Float32(beta), Int32(memory_length), Int32(N), Int32(r))

        strategies .= new_strategies

        data[r, :] .= useful_data_gpu(strategies)
    end

    return data
end