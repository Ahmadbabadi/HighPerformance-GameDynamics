import numpy as np
from numba import njit, prange
import time


@njit(cache=True, parallel=True)
def _build_neighbors(L):
    neighbors = np.empty((L**2, 5), dtype=np.int64)
    for p in prange(L**2):
        x = p % L
        y = p // L
        neighbors[p, 0] = p
        neighbors[p, 1] = y * L + (x + 1) % L
        neighbors[p, 2] = y * L + (x - 1) % L
        neighbors[p, 3] = ((y + 1) % L) * L + x
        neighbors[p, 4] = ((y - 1) % L) * L + x
    return neighbors


@njit(cache=True)
def _payoff_calculator(player_index, strategies, synergy_rate, cost, neighbors):
    unit = neighbors[player_index]
    participants = 0
    cooperators  = 0
    for idx in unit:
        s = strategies[idx]
        if s == 0:
            participants += 1
        if s == 1:
            cooperators += 1
            participants += 1

    coopator_portion = cost
    defector_portion = cost
    loner_portion    = cost

    if participants > 1:
        coopator_portion = (synergy_rate * cooperators * cost) / participants
        defector_portion = coopator_portion + cost

    return defector_portion, coopator_portion, loner_portion


@njit(cache=True)
def _build_cumulative_payoffs(strategies, synergy_rate, cost, neighbors):
    N = strategies.size
    cumulative_payoff = np.zeros(N, dtype=np.float32)
    for p in range(N):
        payoffs = _payoff_calculator(p, strategies, synergy_rate, cost, neighbors)
        unit = neighbors[p]
        for pp in unit:
            cumulative_payoff[pp] += payoffs[strategies[pp]]
    return cumulative_payoff


@njit(cache=True)
def _update_memory(memories, strategies, cumulative_payoff, memory_pointer):
    N = strategies.size
    temp_mean = np.zeros((N, 3), dtype=np.float32)
    memory_length = memories.shape[2]
    for i in range(N):
        for s in range(3):
            total = 0.0
            for t in range(memory_length):
                total += memories[i, s, t]
            temp_mean[i, s] = total / memory_length
    for i in range(N):
        temp_mean[i, strategies[i]] = cumulative_payoff[i]
    for i in range(N):
        for s in range(3):
            memories[i, s, memory_pointer] = temp_mean[i, s]
    memory_pointer = (memory_pointer + 1) % memory_length


@njit(cache=True)
def _pick_new_strategy(strategies, memories, beta, neighbors):
    memory_length = memories.shape[2]
    temp_strategies = strategies.copy()

    for p in range(strategies.size):
        unit = neighbors[p]
        random_idx = np.random.randint(1, len(unit))
        chosen = unit[random_idx]

        own_mean = 0.0
        for t in range(memory_length):
            own_mean += memories[p, strategies[p], t]
        own_mean /= memory_length

        chosen_mean = 0.0
        for t in range(memory_length):
            chosen_mean += memories[chosen, strategies[chosen], t]
        chosen_mean /= memory_length

        weight = 1.0 / (1.0 + np.exp(beta * (own_mean - chosen_mean)))
        if np.random.random() < weight:
            temp_strategies[p] = strategies[chosen]

    return temp_strategies


def _useful_data(state):
    cooperators_rate = np.mean(state["strategies"] == 1)
    defectors_rate = np.mean(state["strategies"] == 0)
    return np.array([cooperators_rate, defectors_rate])


def init_simulation(
                    lattice_size,
                    memory_length,
                    synergy_rate,
                    cost=1 / 5,
                    beta=10,
                    cooperators_rate=1 / 3,
                    defectors_rate=1 / 3,
                    ):

    N = lattice_size ** 2
    strategies = np.random.choice([0, 1, 2], size=N,
                                  p=[defectors_rate, cooperators_rate, 1 - (cooperators_rate + defectors_rate)],
                                  ).astype(np.int64)

    state = dict(
        lattice_size    = lattice_size,
        memory_length   = memory_length,
        synergy_rate    = synergy_rate,
        cost            = cost,
        beta            = beta,
        strategies      = strategies,
        neighbors       = _build_neighbors(lattice_size),
        memories        = np.ones((N, 3, memory_length), dtype=np.float32),
        memory_pointer  = 0,
        initial_features = np.array(
            [lattice_size, memory_length, synergy_rate, beta,
             cooperators_rate, defectors_rate]
        ),
    )
    return state


def get_payoff_and_update_memory(state):
    cumulative_payoff = _build_cumulative_payoffs(
        state["strategies"],
        state["synergy_rate"],
        state["cost"],
        state["neighbors"],
    )
    _update_memory(
        state["memories"],
        state["strategies"],
        cumulative_payoff,
        state["memory_pointer"],
    )
    state["memory_pointer"] = (state["memory_pointer"] + 1) % state["memory_length"]
    return state


def pick_new_strategy(state):
    state["strategies"] = _pick_new_strategy(state["strategies"],
                                             state["memories"],
                                             state["beta"],
                                             state["neighbors"]
                                             )
    return state


def run_simulation(state, rounds):
    data = np.zeros((rounds, 2), dtype=np.float16)
    data[0] = _useful_data(state)
    for r in range(1, rounds):
        state = get_payoff_and_update_memory(state)
        state = pick_new_strategy(state)
        data[r] = _useful_data(state)
    state["data"] = data
    return state


def save_simulation(state):
    f = state["initial_features"]
    np.savez_compressed(
                        "pgg_{}_{}_{}.npz".format(int(f[0]), int(f[1]), f[2]),
                        initial_features=f,
                        data=state["data"],
                        )


if __name__ == "__main__":
    state = init_simulation(
                            lattice_size    = 512,
                            memory_length   = 1,
                            synergy_rate    = 4.5,
                            cost            = 1 / 5,
                            beta            = 10,
                            cooperators_rate = 1 / 3,
                            defectors_rate   = 1 / 3,
                            )
    start = time.perf_counter()
    state = run_simulation(state, rounds=1000)
    print(time.perf_counter() - start)
    print(state["data"][-1])