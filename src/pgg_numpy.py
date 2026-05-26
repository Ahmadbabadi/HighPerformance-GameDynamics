import numpy as np


class PggPython:
    def __init__(self, lattice_size, length_of_memeory, synergy_rate, cost=1/5, beta=10, cooprators_rate = 1/3, defectors_rate = 1/3):
        self.lattice_size = lattice_size
        self.synergy_rate = synergy_rate
        self.cost = cost
        self.beta = beta
        self.strategies = np.random.choice([0, 1, 2], size=self.lattice_size**2, p=[defectors_rate, cooprators_rate, 1-(cooprators_rate+defectors_rate)])
        self.memories = np.ones((self.lattice_size**2, 3, length_of_memeory), dtype=np.float32)
        return None
    
    def unit_of_player(self, player_index):
        L = self.lattice_size
        x = player_index % L
        y = player_index // L
        neighbors = np.array([player_index, y * L + ((x + 1) % L),
                              y * L + ((x - 1) % L), ((y + 1) % L) * L + x,
                              ((y - 1) % L) * L + x])
        return neighbors
    
    def payoff_calculator(self, player_index):
        coopator_portion = self.cost
        defector_portion = self.cost
        loner_portion = self.cost
        unit = self.unit_of_player(player_index)
        participants = unit.size - np.sum(self.strategies[unit]==2)
        if participants > 1: # Single cooprator or deffetor beheives like group of loners.
            coopator_portion = (self.synergy_rate * np.sum(self.strategies[unit]==1) *self.cost)/participants
            defector_portion = coopator_portion+self.cost
        return [defector_portion, coopator_portion, loner_portion]
    
    def build_cumulative_payoffs(self):
        cumulative_payoff = np.zeros(self.lattice_size**2, dtype=np.float16)
        for p in range(self.lattice_size**2):
            payoffs = self.payoff_calculator(p)
            for pp in self.unit_of_player(p):
                cumulative_payoff[pp] += payoffs[self.strategies[pp]]
        return cumulative_payoff
    
    def get_payoff_and_update_memory(self):
        cumulative_payoff = self.build_cumulative_payoffs()
        temp_mean = np.mean(self.memories, axis =2)
        self.memories[:, :, :-1] = self.memories[:, :, 1:]
        self.memories[:, :, -1] = temp_mean
        self.memories[np.arange(self.lattice_size**2), self.strategies, -1] = cumulative_payoff
        return 0
    
    def pick_new_strategy(self):
        temp_strategies = np.copy(self.strategies)
        for p in range(self.lattice_size**2):
            unit = self.unit_of_player(p)
            choosed_neighber = unit[np.random.randint(len(unit))]
            weight = 1/(1+np.exp(self.beta*(np.mean(self.memories[p][self.strategies[p]]) - np.mean(self.memories[choosed_neighber][self.strategies[choosed_neighber]]) )))
            if np.random.rand() < weight:
                temp_strategies[p] =  self.strategies[choosed_neighber]
        self.strategies = temp_strategies
    
    def useful_data(self):
        cooprators_rate = np.mean(self.strategies==1)
        defectors_rate = np.mean(self.strategies==0)
        return np.array([cooprators_rate, defectors_rate])
    
    def main(self, rounds):
        results = np.zeros((rounds, 2), dtype=np.float16)
        results[0] = self.useful_data()
        for r in range(1, rounds):
            self.get_payoff_and_update_memory()
            self.pick_new_strategy()
            results[r] = self.useful_data()
        return results
    
