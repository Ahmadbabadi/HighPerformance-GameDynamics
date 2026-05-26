import random
import math

def mean_list(python_list):
    return sum(python_list)/len(python_list)

class PggPython:
    def __init__(self, lattice_size, length_of_memeory, synergy_rate, cost=1/5, beta=10, cooprators_rate = 0.33, defectors_rate = 0.33):
        self.lattice_size = lattice_size
        self.strategies = [random.randint(0, 2) for p in range(self.lattice_size**2)]
        self.memories = [[[1 for m in range(length_of_memeory-1)]+[1] for i in range(3)] for p in range(self.lattice_size**2)]
        self.synergy_rate = synergy_rate
        self.cost = cost
        self.beta = beta
        return None
    
    def unit_of_player(self, player_index):
        L = self.lattice_size
        x = player_index % L
        y = player_index // L
        neighbors = [player_index, y * L + ((x + 1) % L),
                     y * L + ((x - 1) % L), ((y + 1) % L) * L + x,
                     ((y - 1) % L) * L + x]
        return neighbors
    
    def payoff_calculator(self, player_index):
        coopator_portion = self.cost
        defector_portion = self.cost
        loner_portion = self.cost
        unit = self.unit_of_player(player_index)
        if len(unit) - [self.strategies[p] for p in unit].count(2) > 1: # Single cooprator or deffetor beheives like group of loners. 
            coopator_portion = (self.synergy_rate * [self.strategies[p] for p in unit].count(1) *self.cost)/(len(unit) - [self.strategies[p] for p in unit].count(2))
            defector_portion = coopator_portion+self.cost
        return [defector_portion, coopator_portion, loner_portion]
    
    def build_cumulative_payoffs(self):
        cumulative_payoff = [0 for p in range(self.lattice_size**2)]
        for p in range(self.lattice_size**2):
            payoffs = self.payoff_calculator(p)
            for pp in self.unit_of_player(p):
                cumulative_payoff[pp] += payoffs[self.strategies[pp]]
        return cumulative_payoff
    
    def get_payoff_and_update_memory(self):
        cumulative_payoff = self.build_cumulative_payoffs()
        for p in range(self.lattice_size**2):
            for stra in range(2):
                temp_mean = mean_list(self.memories[p][stra])
                self.memories[p][stra][:-1] = self.memories[p][stra][1:]
                self.memories[p][stra][-1] = temp_mean
            self.memories[p][self.strategies[p]][-1] = cumulative_payoff[p]
        return 0
    
    def pick_new_strategy(self):
        temp_strategies = self.strategies.copy()
        for p in range(self.lattice_size**2):
            unit = self.unit_of_player(p)
            choosed_neighber = unit[random.randint(0, len(unit)-1)]
            weight = 1/(1+math.exp(self.beta*(mean_list(self.memories[p][self.strategies[p]]) - self.memories[choosed_neighber][self.strategies[choosed_neighber]][-1])))
            if random.random() < weight:
                temp_strategies[p] =  self.strategies[choosed_neighber]
        self.strategies = temp_strategies
    
    def useful_data(self):
        cooprators_rate = self.strategies.count(1)/self.lattice_size**2
        defectors_rate = self.strategies.count(0)/self.lattice_size**2
        return cooprators_rate, defectors_rate
    
    def main(self, rounds):
        results = [[0, 0] for r in range(rounds)]
        results[0] = self.useful_data()
        for r in range(1, rounds):
            self.get_payoff_and_update_memory()
            self.pick_new_strategy()
            results[r] = self.useful_data()
        return results
