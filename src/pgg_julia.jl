module pgg_julia # make all files as a local julia module

import StatsBase as sb
import Statistics
using Random

function build_units(lattice_size)
    L = lattice_size
    units = zeros(Int64, lattice_size^2, 5)
    for i = 1:lattice_size^2
        x = mod1(i, L)
        y = cld(i, L)
        units[i, :] = [i,
                        (y - 1) * L + mod1(x - 1, L),
                        (y - 1) * L + mod1(x + 1, L),
                        (mod1(y - 1, L) - 1) * L + x,
                        (mod1(y + 1, L) - 1) * L + x]
    end
    return units
end


function pgg_init(lattice_size, memory_length, cooperator_rate=1/3, defector_rate=1/3)
    units = build_units(lattice_size)
    item_weights = sb.Weights([cooperator_rate, defector_rate, 1-(cooperator_rate+defector_rate)])
    strategies = sb.sample([1, 2, 3], item_weights, lattice_size^2) # use [1, 2, 3] because is compatible to julia index
    memories = ones(lattice_size^2, 3, memory_length)
    return units, strategies, memories
end

function payoff_calculator(unit, strategies, synergy_rate, cost)
    coopator_payoff = cost
    defector_payoff = cost
    loner_payoff    = cost
    cooprators = 0
    participants = 0
    for neigbor = unit
        if strategies[neigbor] == 1
            cooprators += 1
            participants += 1
        elseif strategies[neigbor] == 2
            participants += 1
        end
    end

    if participants > 1
        coopator_payoff = cost*synergy_rate*cooprators/participants
        defector_payoff = coopator_payoff + cost
    end
    
    return coopator_payoff, defector_payoff, loner_payoff

end
function build_cumulative_payoffs(strategies, units, synergy_rate, cost)
    cumulative_payoffs = zeros(length(strategies))
    for p = eachindex(strategies)
        unit = units[p, :]
        unti_payoff = payoff_calculator(units[p, :], strategies, synergy_rate, cost)
        for pp = unit
            cumulative_payoffs[pp] += unti_payoff[strategies[pp]]
        end
    end

    return cumulative_payoffs
end

function update_memory(memories, strategies, cumulative_payoffs, memory_pointer)
    tmp_memory = dropdims(Statistics.mean(memories, dims=3), dims=3)
    tmp_memory[CartesianIndex.(1:length(strategies), strategies)] = cumulative_payoffs
    memories[:, :, memory_pointer] = tmp_memory
    return memories, mod1(memory_pointer + 1, size(memories, 3))
end

function pick_new_strategy(strategies, memories, units, beta)
    new_strategies = deepcopy(strategies)
    for p = eachindex(strategies)
        selected_neighbor = units[p, rand(2:5)]
        if strategies[p] != strategies[selected_neighbor]
            memory_player = Statistics.mean(memories[p, strategies[p], :])
            memory_neighbor = Statistics.mean(memories[selected_neighbor, strategies[selected_neighbor], :])
            weight = 1/(1+exp(beta*(memory_player-memory_neighbor)))
            if rand() < weight
                new_strategies[p] = strategies[selected_neighbor]
            end
        end
    end
    return new_strategies
    
end

function useful_data(strategies)
    cooperator_rate = sum(strategies .== 1)/length(strategies)
    defector_rate = sum(strategies .== 2)/length(strategies)
    return [cooperator_rate, defector_rate]
end

function run_simulation(lattice_size, memory_length, synergy_rate, rounds, beta=10, cost=1/5, cooperator_rate=1/3, defector_rate=1/3)
    data = zeros(rounds+1, 2)
    memory_pointer = 1
    units, strategies, memories = pgg_init(lattice_size, memory_length, cooperator_rate, defector_rate)
    data[1, :] = useful_data(strategies)
    for r in 2:rounds+1
        cumulative_payoffs = build_cumulative_payoffs(strategies, units, synergy_rate, cost)
        memories, memory_pointer = update_memory(memories, strategies, cumulative_payoffs, memory_pointer)
        strategies = pick_new_strategy(strategies, memories, units, beta)
        data[r, :] = useful_data(strategies)
    end
    return data
end

end # end the pgg_julia_opt module