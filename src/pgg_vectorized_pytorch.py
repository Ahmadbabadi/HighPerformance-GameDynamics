import torch
from safetensors.torch import save_file

# Disable gradient calculation globally
torch.set_grad_enabled(False)

class PggVectorizedPyTorch:
    def __init__(self, lattice_size, memory_length, synergy_rate, cost=1/5, beta=10, cooperators_rate = 1/3, defectors_rate = 1/3):
        self.lattice_size = lattice_size
        self.synergy_rate = synergy_rate
        self.players_number = lattice_size ** 2
        self.memory_length = memory_length
        self.cost = cost
        self.beta = beta
        self.initial_featurs = torch.tensor([lattice_size, 
                                             memory_length,
                                             synergy_rate, beta,
                                             cooperators_rate, defectors_rate])
        weights = torch.tensor([defectors_rate, cooperators_rate, 1-(cooperators_rate+defectors_rate)], dtype=torch.half)
        self.strategies = torch.multinomial(weights, num_samples=self.players_number, replacement=True)
        self.memories = torch.ones((self.players_number, 3, memory_length), dtype=torch.float32)
        return None
    
    def build_adjacency_matrix(self):
        self.neighbors = torch.empty((self.players_number ,5), dtype=torch.int32)
        self.neighbors[:, 0] = torch.arange(self.players_number)
        x = torch.remainder(self.neighbors[:, 0], self.lattice_size)
        y = torch.floor_divide(self.neighbors[:, 0], self.lattice_size)
        self.neighbors[:, 1] = y * self.lattice_size + (torch.remainder(x+1, self.lattice_size))
        self.neighbors[:, 2] = y * self.lattice_size + (torch.remainder(x-1, self.lattice_size))
        self.neighbors[:, 3] = torch.remainder(y+1, self.lattice_size) * self.lattice_size + x
        self.neighbors[:, 4] = torch.remainder(y-1, self.lattice_size) * self.lattice_size + x
        return 0
    
    def build_cumulative_payoffs(self):
        payoff_vectors = self.cost * torch.ones((self.players_number, 3), dtype=torch.float32)
        strategies_in_each_unit = self.strategies[self.neighbors]
        participants = torch.sum(strategies_in_each_unit != 2, axis=1)
        cooperators = torch.sum(strategies_in_each_unit == 1, axis=1)
        payoff_vectors[:, 1] = torch.where(participants>1,
                                        (cooperators*self.cost*self.synergy_rate)/participants,
                                        payoff_vectors[:, 1])
        payoff_vectors[:, 0] = torch.where(participants>1,
                                        payoff_vectors[:, 1] + self.cost,
                                        payoff_vectors[:, 0])
        seperated_payoff_units = payoff_vectors[torch.arange(self.players_number)[:, None], self.strategies[self.neighbors]]
        cumulative_payoff = torch.bincount(torch.flatten(self.neighbors),
                                           weights=torch.flatten(seperated_payoff_units) )
        return cumulative_payoff.to(torch.float32)
    
    def get_payoff_and_update_memory(self, cumulative_payoff):
        temp_mean = torch.mean(self.memories, axis =2)
        temp_mean[torch.arange(self.players_number), self.strategies] = cumulative_payoff
        self.memories[:, :, self.last_memeory_pointer] = temp_mean
        self.last_memeory_pointer = (self.last_memeory_pointer+1)%self.memory_length
        return 0
    
    def pick_new_strategy(self):
        new_strategies = self.strategies.detach().clone()
        compared_neighbors = self.neighbors[torch.arange(self.players_number), torch.randint(1, 5, size=(self.players_number, ))]
        neighbor_strategies = self.strategies[compared_neighbors]
        not_equal_neighbors = (self.strategies != neighbor_strategies)
        comparsion_numbers = torch.sum(not_equal_neighbors)
        memory_plyers = torch.mean(self.memories[not_equal_neighbors][torch.arange(comparsion_numbers), self.strategies[not_equal_neighbors]], axis=1)
        memory_choosed_neighbors = torch.mean(self.memories[compared_neighbors[not_equal_neighbors]][torch.arange(comparsion_numbers), neighbor_strategies[not_equal_neighbors]], axis=1)
        flip_condition = torch.rand(comparsion_numbers) < 1/(1+ torch.exp(self.beta*(memory_plyers - memory_choosed_neighbors)) )
        new_strategies[not_equal_neighbors] = torch.where(flip_condition, self.strategies[compared_neighbors][not_equal_neighbors], self.strategies[not_equal_neighbors])
        self.strategies = new_strategies
        return 0
    
    def useful_data(self):
        cooperators_rate = (self.strategies==1).sum()/self.players_number
        defectors_rate = (self.strategies==0).sum()/self.players_number
        return torch.tensor([cooperators_rate, defectors_rate])
    
    def main(self, rounds):
        self.last_memeory_pointer = 0
        self.data = torch.zeros((rounds, 2), dtype=torch.float32)
        self.data[0] = self.useful_data()
        self.build_adjacency_matrix()
        for r in range(1, rounds):
            cumulative_payoff = self.build_cumulative_payoffs()
            self.get_payoff_and_update_memory(cumulative_payoff)
            self.pick_new_strategy()
            self.data[r] = self.useful_data()
        return self.data
    
    def save(self):
        tensors = {
                    "initial_featurs": self.initial_featurs,
                    "data": self.data,
                    }
        save_file(tensors, "pgg_"+"{}_{}_{}.safetensors".format(self.initial_featurs[0],
                                                                self.initial_featurs[1],
                                                                self.initial_featurs[2]))
        return 0