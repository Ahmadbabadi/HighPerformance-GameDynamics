import cupy as cp
import time




class PggPython:
    def __init__(self, lattice_size, length_of_memeory, synergy_rate, cost=1/5, beta=10, cooperators_rate = 1/3, defectors_rate = 1/3):
        self.lattice_size = lattice_size
        self.synergy_rate = synergy_rate
        self.players_number = lattice_size ** 2
        self.cost = cost
        self.beta = beta
        self.initial_featurs = cp.array([lattice_size, length_of_memeory,
                                         synergy_rate, beta,
                                         cooperators_rate, defectors_rate])
        self.strategies = cp.random.choice([0, 1, 2], size=self.players_number, p=[defectors_rate, cooperators_rate, 1-(cooperators_rate+defectors_rate)])
        self.memories = cp.ones((self.players_number, 3, length_of_memeory), dtype=cp.float32)
        return None
    
    def build_adjacency_matrix(self):
        self.neighbors = cp.empty((self.players_number ,5), dtype=cp.int32)
        self.neighbors[:, 0] = cp.arange(self.players_number)
        x = cp.remainder(self.neighbors[:, 0], self.lattice_size)
        y = cp.floor_divide(self.neighbors[:, 0], self.lattice_size)
        self.neighbors[:, 1] = y * self.lattice_size + (cp.remainder(x+1, self.lattice_size))
        self.neighbors[:, 2] = y * self.lattice_size + (cp.remainder(x-1, self.lattice_size))
        self.neighbors[:, 3] = cp.remainder(y+1, self.lattice_size) * self.lattice_size + x
        self.neighbors[:, 4] = cp.remainder(y-1, self.lattice_size) * self.lattice_size + x
        return 0
    
    def build_cumulative_payoffs(self):
        payoff_vectors = self.cost * cp.ones((self.players_number, 3), dtype=cp.float32)
        strategies_in_each_unit = self.strategies[self.neighbors]
        participants = cp.sum(strategies_in_each_unit != 2, axis=1)
        cooperators = cp.sum(strategies_in_each_unit == 1, axis=1)
        payoff_vectors[:, 1] = cp.where(participants>1,
                                        (cooperators*self.cost*self.synergy_rate)/participants,
                                        payoff_vectors[:, 1])
        payoff_vectors[:, 0] = cp.where(participants>1,
                                        payoff_vectors[:, 1] + self.cost,
                                        payoff_vectors[:, 0])
        seperated_payoff_units = payoff_vectors[cp.arange(self.players_number)[:, None], self.strategies[self.neighbors]]
        cumulative_payoff = cp.bincount(self.neighbors.ravel(), weights=seperated_payoff_units.ravel())
        return cumulative_payoff
    
    def get_payoff_and_update_memory(self, cumulative_payoff):
        temp_mean = cp.mean(self.memories, axis =2)
        self.memories[:, :, :-1] = self.memories[:, :, 1:]
        self.memories[:, :, -1] = temp_mean
        self.memories[cp.arange(self.players_number), self.strategies, -1] = cumulative_payoff
        return 0
    
    def pick_new_strategy(self):
        new_strategies = cp.copy(self.strategies)
        compared_neighbors = self.neighbors[cp.arange(self.players_number), cp.random.randint(1, 5, size=(self.players_number))]
        neighbor_strategies = self.strategies[compared_neighbors]
        not_equal_neighbors = (self.strategies != neighbor_strategies)
        comparsion_numbers = int(cp.sum(not_equal_neighbors).item())
        memory_plyers = cp.mean(self.memories[not_equal_neighbors][cp.arange(comparsion_numbers), self.strategies[not_equal_neighbors]], axis=1)
        memory_choosed_neighbors = cp.mean(self.memories[compared_neighbors[not_equal_neighbors]][cp.arange(comparsion_numbers), neighbor_strategies[not_equal_neighbors]], axis=1)
        flip_condition = cp.random.rand(comparsion_numbers) < 1/(1+ cp.exp(self.beta*(memory_plyers - memory_choosed_neighbors)) )
        new_strategies[not_equal_neighbors] = cp.where(flip_condition, self.strategies[compared_neighbors][not_equal_neighbors], self.strategies[not_equal_neighbors])
        self.strategies = new_strategies
        return 0
    
    def useful_data(self):
        cooperators_rate = cp.mean(self.strategies==1)
        defectors_rate = cp.mean(self.strategies==0)
        return cp.array([cooperators_rate, defectors_rate])
    
    def main(self, rounds):
        self.data = cp.zeros((rounds, 2), dtype=cp.float32)
        self.data[0] = self.useful_data()
        self.build_adjacency_matrix()
        for r in range(1, rounds):
            cumulative_payoff = self.build_cumulative_payoffs()
            self.get_payoff_and_update_memory(cumulative_payoff)
            self.pick_new_strategy()
            self.data[r] = self.useful_data()
        return self.data
    
    def save(self):
        cp.savez_compressed("pgg_" +
                        "{}_{}_{}.npz".format( self.initial_featurs[0], self.initial_featurs[1], self.initial_featurs[2]),
                        initial_featurs = self.initial_featurs,
                        data = self.data)
        return 0



