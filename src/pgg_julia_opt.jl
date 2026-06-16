module pgg_julia_opt # make all files as a local julia module

import StatsBase as sb
import Statistics
using Random


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
        units[i, 5] =(mod1(y + 1, L) - 1) * L + x
    end
    return units
end


function pgg_init(lattice_size, memory_length, cooperator_rate=1/3, defector_rate=1/3)
    units = build_units(lattice_size)
    item_weights = sb.Weights([cooperator_rate, defector_rate, 1-(cooperator_rate+defector_rate)])
    strategies = sb.sample(Int8[1, 2, 3], item_weights, lattice_size^2) # use [1, 2, 3] because is compatible to julia index
    memories = ones(Float32, lattice_size^2, 3, memory_length)
    return units, strategies, memories
end



function payoff_calculator(unit, strategies, synergy_rate, cost)
    coopator_payoff = cost
    defector_payoff = cost
    loner_payoff    = cost
    cooprators = 0
    participants = 0
    for neigbor = unit
        S = strategies[neigbor]
        if S == 1
            cooprators += 1
            participants += 1
        elseif S == 2
            participants += 1
        end
    end

    if participants > 1
        coopator_payoff = cost*synergy_rate*cooprators/participants
        defector_payoff = coopator_payoff + cost
    end
    
    return coopator_payoff, defector_payoff, loner_payoff

end

function build_cumulative_payoffs(strategies, cumulative_payoffs, units, synergy_rate, cost)
    fill!(cumulative_payoffs, 0.0)
    for p = eachindex(strategies)
        unit = @view units[p, :]
        c, d, l = payoff_calculator(unit, strategies, synergy_rate, cost)
        for pp = unit
            s = strategies[pp]
            v = s == 1 ? c : s == 2 ? d : l
            cumulative_payoffs[pp] += v
        end
    end

    return cumulative_payoffs
end


function update_memory(memories, strategies, cumulative_payoffs, memory_pointer, buffer_matrix)
    sum!(buffer_matrix, memories)
    memory_length = size(memories, 3)
    buffer_matrix ./= memory_length
    for i in 1:length(strategies)
        buffer_matrix[i, strategies[i]] = cumulative_payoffs[i]
    end
    memories[:, :, memory_pointer] .= buffer_matrix
    return memories, mod1(memory_pointer + 1, memory_length)
end


function pick_new_strategy(strategies, memories, units, beta)
    new_strategies = copy(strategies)
    for p = eachindex(strategies)
        selected_neighbor = units[p, rand(2:5)]
        if strategies[p] != strategies[selected_neighbor]
            memory_player = Statistics.mean(view( memories, p, strategies[p], :))
            memory_neighbor = Statistics.mean(view( memories, selected_neighbor, strategies[selected_neighbor], :))
            weight = 1/(1+exp(beta*(memory_player-memory_neighbor)))
            if rand() < weight
                new_strategies[p] = strategies[selected_neighbor]
            end
        end
    end
    return new_strategies
end


function useful_data(strategies)
    N = length(strategies)
    cooperators = count(==(1), strategies)
    defectors = count(==(2), strategies)
    return [cooperators / N, defectors / N]
end


function run_simulation(lattice_size, memory_length, synergy_rate, rounds, beta=10, cost=1/5, cooperator_rate=1/3, defector_rate=1/3)
    N = lattice_size^2
    data = zeros(rounds+1, 2)
    memory_pointer = 1
    units, strategies, memories = pgg_init(lattice_size, memory_length, cooperator_rate, defector_rate)
    data[1, :] = useful_data(strategies)
    cumulative_payoffs = zeros(Float32, N)
    buffer_matrix = zeros(Float32, N, 3)
    for r in 2:rounds+1
        cumulative_payoffs = build_cumulative_payoffs(strategies, cumulative_payoffs, units, synergy_rate, cost)
        memories, memory_pointer = update_memory(memories, strategies, cumulative_payoffs, memory_pointer, buffer_matrix)
        strategies = pick_new_strategy(strategies, memories, units, beta)
        data[r, :] = useful_data(strategies)
    end
    return data
end

end # end the pgg_julia_opt module