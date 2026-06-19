import numpy as np
from numba import cuda
import math


@cuda.jit(device=True)
def _get_neighbors_device(p, L, out):
    x = p % L
    y = p // L
    out[0] = p
    out[1] = y * L + (x + 1) % L
    out[2] = y * L + (x - 1) % L
    out[3] = ((y + 1) % L) * L + x
    out[4] = ((y - 1) % L) * L + x


@cuda.jit(device=True)
def _payoff_calculator_device(player_index, strategies, synergy_rate, cost, L):
    unit = cuda.local.array(5, dtype=np.int64)
    _get_neighbors_device(player_index, L, unit)

    participants = 0
    cooperators  = 0
    for k in range(5):
        s = strategies[unit[k]]
        if s == 1:
            cooperators  += 1
            participants += 1
        elif s == 0:
            participants += 1

    coopator_portion = cost
    defector_portion = cost
    loner_portion    = cost

    if participants > 1:
        coopator_portion = (synergy_rate * cooperators * cost) / participants
        defector_portion = coopator_portion + cost

    return defector_portion, coopator_portion, loner_portion



@cuda.jit
def _build_cumulative_payoffs_kernel(strategies, synergy_rate, cost, L, cumulative_payoff):
    p = cuda.grid(1)
    N = strategies.size
    if p >= N:
        return

    unit = cuda.local.array(5, dtype=np.int64)
    _get_neighbors_device(p, L, unit)

    payoffs_d, payoffs_c, payoffs_l = _payoff_calculator_device(
        p, strategies, synergy_rate, cost, L)

    for k in range(5):
        pp = unit[k]
        s  = strategies[pp]
        if s == 0:
            cuda.atomic.add(cumulative_payoff, pp, payoffs_d)
        elif s == 1:
            cuda.atomic.add(cumulative_payoff, pp, payoffs_c)
        else:
            cuda.atomic.add(cumulative_payoff, pp, payoffs_l)


@cuda.jit
def _update_memory_kernel(memories, strategies, cumulative_payoff,
                           memory_pointer, memory_length):
    i = cuda.grid(1)
    N = strategies.size
    if i >= N:
        return

    s_i = strategies[i]

    total = 0.0
    for t in range(memory_length):
        total += memories[i, s_i, t]
    mean_val = total / memory_length

    for s in range(3):
        if s == s_i:
            memories[i, s, memory_pointer] = cumulative_payoff[i]
        else:
            tot_s = 0.0
            for t in range(memory_length):
                tot_s += memories[i, s, t]
            memories[i, s, memory_pointer] = tot_s / memory_length


@cuda.jit
def _pick_new_strategy_kernel(strategies, temp_strategies, memories,
                               beta, memory_length, L, rng_states):
    p = cuda.grid(1)
    N = strategies.size
    if p >= N:
        return

    unit = cuda.local.array(5, dtype=np.int64)
    _get_neighbors_device(p, L, unit)

    r0     = cuda.random.xoroshiro128p_uniform_float32(rng_states, p)
    chosen = unit[1 + int(r0 * 4) % 4]

    own_mean = 0.0
    for t in range(memory_length):
        own_mean += memories[p, strategies[p], t]
    own_mean /= memory_length

    chosen_mean = 0.0
    for t in range(memory_length):
        chosen_mean += memories[chosen, strategies[chosen], t]
    chosen_mean /= memory_length

    weight = 1.0 / (1.0 + math.exp(beta * (own_mean - chosen_mean)))

    r1 = cuda.random.xoroshiro128p_uniform_float32(rng_states, p)
    if r1 < weight:
        temp_strategies[p] = strategies[chosen]
    else:
        temp_strategies[p] = strategies[p]



def _useful_data(strategies_host):
    cooperators_rate = np.mean(strategies_host == 1)
    defectors_rate   = np.mean(strategies_host == 0)
    return np.array([cooperators_rate, defectors_rate])


def _cuda_cfg(N, threads=256):
    blocks = (N + threads - 1) // threads
    return blocks, threads


def init_simulation(lattice_size, memory_length, synergy_rate,
                    cost=1/5, beta=10,
                    cooperators_rate=1/3, defectors_rate=1/3):

    N = lattice_size ** 2
    strategies = np.random.choice([0, 1, 2], size=N,
                                  p=[defectors_rate, cooperators_rate,
                                  1 - (cooperators_rate + defectors_rate)]
                                  ).astype(np.int64)

    from numba.cuda.random import create_xoroshiro128p_states
    rng_states = create_xoroshiro128p_states(N, seed=np.random.randint(0, 2**31))

    state = dict(lattice_size     = lattice_size,
                 memory_length    = memory_length,
                 synergy_rate     = np.float32(synergy_rate),
                 cost             = np.float32(cost),
                 beta             = np.float32(beta),
                 strategies       = cuda.to_device(strategies),
                 temp_strategies  = cuda.device_array(N, dtype=np.int64),
                 memories         = cuda.to_device(np.ones((N, 3, memory_length), dtype=np.float32)),
                 memory_pointer   = 0,
                 rng_states       = rng_states,
                 initial_features = np.array([lattice_size, memory_length, synergy_rate,
                                              beta, cooperators_rate, defectors_rate]),
                )
    return state


def get_payoff_and_update_memory(state):
    N             = state["strategies"].size
    L             = state["lattice_size"]
    blocks, tpb   = _cuda_cfg(N)

    cumulative_payoff = cuda.device_array(N, dtype=np.float32)
    cumulative_payoff[:] = 0

    _build_cumulative_payoffs_kernel[blocks, tpb](state["strategies"], state["synergy_rate"], state["cost"],
                                                  L, cumulative_payoff)

    _update_memory_kernel[blocks, tpb](state["memories"], state["strategies"], cumulative_payoff,
                                       state["memory_pointer"], state["memory_length"])

    state["memory_pointer"] = (state["memory_pointer"] + 1) % state["memory_length"]
    return state


def pick_new_strategy(state):
    N           = state["strategies"].size
    blocks, tpb = _cuda_cfg(N)

    _pick_new_strategy_kernel[blocks, tpb](state["strategies"], state["temp_strategies"], state["memories"],
                                           state["beta"], state["memory_length"], state["lattice_size"],
                                           state["rng_states"])
    
    state["strategies"], state["temp_strategies"] = state["temp_strategies"], state["strategies"]
    return state


def run_simulation(state, rounds):
    data    = np.zeros((rounds, 2), dtype=np.float16)
    data[0] = _useful_data(state["strategies"].copy_to_host())

    for r in range(1, rounds):
        state  = get_payoff_and_update_memory(state)
        state  = pick_new_strategy(state)
        data[r] = _useful_data(state["strategies"].copy_to_host())

    state["data"] = data
    return state


def save_simulation(state):
    f = state["initial_features"]
    np.savez_compressed("pgg_{}_{}_{}.npz".format(int(f[0]), int(f[1]), f[2]),
                        initial_features=f, data=state["data"],)