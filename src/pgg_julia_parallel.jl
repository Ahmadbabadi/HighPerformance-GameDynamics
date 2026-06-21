import StatsBase as sb
import Statistics
using Random
using Base.Threads


function build_units(lattice_size)
    L = Int32(lattice_size) #  lattice_size <= 46,340$
    units = zeros(Int32, L^2, 5)
    for i = axes(units, 1)
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


function pgg_init(lattice_size, memory_length, cooperator_rate=1/3, defector_rate=1/3)
    units = build_units(lattice_size)
    item_weights = sb.Weights([cooperator_rate, defector_rate, 1-(cooperator_rate+defector_rate)])
    strategies = sb.sample(Int8[1, 2, 3], item_weights, lattice_size^2)
    memories = ones(Float32, memory_length, 3, lattice_size^2)  # [t, s, i]
    return units, strategies, memories
end


function payoff_calculator(unit, strategies, synergy_rate, cost)
    cooperator_payoff = cost
    defector_payoff = cost
    loner_payoff = cost
    cooperators  = 0
    participants = 0
    for neighbor in unit
        S = strategies[neighbor]
        if S == 1
            cooperators  += 1
            participants += 1
        elseif S == 2
            participants += 1
        end
    end

    if participants > 1
        cooperator_payoff = cost * synergy_rate * cooperators / participants
        defector_payoff   = cooperator_payoff + cost
    end

    return cooperator_payoff, defector_payoff, loner_payoff
end


function build_cumulative_payoffs(strategies, cumulative_payoffs, units, synergy_rate, cost)
    fill!(cumulative_payoffs, 0.0)

    @threads :static for p in eachindex(strategies)
        total = 0.0f0
        s_p = strategies[p]
        for g in @view units[p, :]
            unit_g = @view units[g, :]
            c, d, l = payoff_calculator(unit_g, strategies, synergy_rate, cost)
            total += s_p == 1 ? c : s_p == 2 ? d : l
        end
        cumulative_payoffs[p] = total
    end

    return cumulative_payoffs
end


function update_memory(memories, strategies, cumulative_payoffs, memory_pointer)
    memory_length = size(memories, 1)
    for i in eachindex(strategies)
        s = strategies[i]
        memories[memory_pointer, s, i] = cumulative_payoffs[i]
    end
    return memories, mod1(memory_pointer + 1, memory_length)
end


function pick_new_strategy!(new_strategies, strategies, memories, units, beta)
    @threads :static for p in eachindex(strategies)
        selected_neighbor = units[p, rand(2:5)]
        s_p = strategies[p]
        s_n = strategies[selected_neighbor]

        if s_p != s_n
            memory_player = Statistics.mean(view(memories, :, s_p, p))
            memory_neighbor = Statistics.mean(view(memories, :, s_n, selected_neighbor))
            weight = 1 / (1 + exp(beta * (memory_player - memory_neighbor)))
            new_strategies[p] = rand() < weight ? s_n : s_p
        else
            new_strategies[p] = s_p
        end
    end
    return new_strategies
end


function useful_data(strategies)
    N = length(strategies)
    cooperators = count(==(1), strategies)
    defectors   = count(==(2), strategies)
    return [cooperators / N, defectors / N]
end


function run_simulation(lattice_size, memory_length, synergy_rate, rounds,
                        beta=10, cost=1/5, cooperator_rate=1/3, defector_rate=1/3)
    N = lattice_size^2
    data = zeros(rounds + 1, 2)
    memory_pointer = 1

    units, strategies, memories = pgg_init(lattice_size, memory_length, cooperator_rate, defector_rate)
    cumulative_payoffs = zeros(Float32, N)
    new_strategies     = similar(strategies)

    data[1, :] = useful_data(strategies)

    for r in 2:rounds+1
        build_cumulative_payoffs(strategies, cumulative_payoffs, units, synergy_rate, cost)
        memories, memory_pointer = update_memory(memories, strategies, cumulative_payoffs, memory_pointer)
        pick_new_strategy!(new_strategies, strategies, memories, units, beta)
        strategies .= new_strategies
        data[r, :] = useful_data(strategies)
    end
    return data
end