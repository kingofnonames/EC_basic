import numpy as np
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------
# Utility functions
# ---------------------------
def parse_mtx_pattern_symmetric(mtx_text: str) -> np.ndarray:
    lines = mtx_text.strip().splitlines()
    data_lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith('%')]
    header = data_lines[0].split()
    nrows, ncols, _ = map(int, header[:3])
    if nrows != ncols:
        raise ValueError("Matrix must be square")
    n = nrows
    A = np.zeros((n, n), dtype=np.int8)
    for ln in data_lines[1:]:
        i, j = map(int, ln.split()[:2])
        A[i - 1, j - 1] = 1
        A[j - 1, i - 1] = 1
    return A

def neighbor_counts(labels: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pop_eq3 = (labels == 3).astype(np.int32)
    pop_eq2 = (labels == 2).astype(np.int32)
    pop_ge2 = (labels >= 2).astype(np.int32)
    count3 = pop_eq3 @ A
    count2 = pop_eq2 @ A
    count_ge2 = pop_ge2 @ A
    return count3, count2, count_ge2

def feasibility_mask(labels: np.ndarray, A: np.ndarray) -> np.ndarray:
    count3, count2, count_ge2 = neighbor_counts(labels, A)
    is0 = (labels == 0)
    is1 = (labels == 1)
    ok0 = (count3 >= 1) | (count2 >= 2)
    ok1 = (count_ge2 >= 1)
    ok = np.ones_like(labels, dtype=bool)
    ok &= (~is0 | ok0)
    ok &= (~is1 | ok1)
    return ok

def make_feasible(labels: np.ndarray, A: np.ndarray, rng: np.random.Generator, max_iter=5) -> np.ndarray:
    pop = labels.copy()
    if pop.ndim == 1:
        pop = pop[None, :]
    pop_size, n = pop.shape
    for _ in range(max_iter):
        ok = feasibility_mask(pop, A)
        if ok.all():
            break
        bad_mask = ~ok
        rnd_vals = rng.integers(1, 4, size=bad_mask.sum(), dtype=np.int8)
        pop[bad_mask] = rnd_vals
    if pop.shape[0] == 1:
        return pop[0]
    return pop

def fitness_function(pop: np.ndarray, A: np.ndarray, penalty_weight: float = 20.0) -> np.ndarray:
    pop_arr = pop if pop.ndim == 2 else pop[None, :]
    base = pop_arr.sum(axis=1).astype(float)
    ok_mask = feasibility_mask(pop_arr, A)
    violations = (~ok_mask).sum(axis=1).astype(float)
    return base + penalty_weight * violations

# ---------------------------
# Heuristics (giữ nguyên)
# ---------------------------
def tournament_select(pop, fitness, k, rng):
    pop_size = pop.shape[0]
    candidates = rng.integers(0, pop_size, size=k)
    best = candidates[np.argmin(fitness[candidates])]
    return best


def rank_select(fitness: np.ndarray, rng: np.random.Generator) -> int:
    # smaller fitness => better rank (0 best)
    order = np.argsort(fitness)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(fitness))
    # convert to probabilities proportional to (N - rank)
    N = len(fitness)
    weights = (N - ranks).astype(float)
    probs = weights / weights.sum()
    return rng.choice(len(fitness), p=probs)


def uniform_crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator, p=0.5):
    mask = rng.random(parent_a.size) < p
    child1 = np.where(mask, parent_a, parent_b).astype(np.int8)
    child2 = np.where(mask, parent_b, parent_a).astype(np.int8)
    return child1, child2


def mutate(solution: np.ndarray, mut_rate: float, rng: np.random.Generator):
    """Mutation nhẹ: với xác suất mut_rate, tăng/giảm 1 mod 4 hoặc đổi ngẫu nhiên trong vài phần."""
    n = solution.size
    for i in range(n):
        if rng.random() < mut_rate:
            # nhẹ: +/-1 (mod 4) với prob 0.8, ngẫu nhiên 0..3 với prob 0.2
            if rng.random() < 0.8:
                if rng.random() < 0.5:
                    solution[i] = (int(solution[i]) + 1) % 4
                else:
                    solution[i] = (int(solution[i]) - 1) % 4
            else:
                solution[i] = rng.integers(0, 4, dtype=np.int8)
    return solution

def heuristic1(n, A, rng):
    labels = -1 * np.ones(n, dtype=np.int8)
    remaining = np.ones(n, dtype=bool)
    while remaining.any():
        idxs = np.nonzero(remaining)[0]
        if idxs.size == 1:
            labels[idxs[0]] = 2
            remaining[idxs[0]] = False
            break
        v = rng.choice(idxs)
        labels[v] = 3
        neighs = A[v].astype(bool)
        labels[neighs & remaining] = 0
        remaining[v] = False
        remaining[neighs] = False
    labels[labels == -1] = 2
    return labels

def heuristic2(n, A, rng):
    labels = -1 * np.ones(n, dtype=np.int8)
    remaining = np.ones(n, dtype=bool)
    while remaining.any():
        idxs = np.nonzero(remaining)[0]
        v = rng.choice(idxs)
        labels[v] = 3
        neighs = A[v].astype(bool)
        labels[neighs & remaining] = 0
        remaining[v] = False
        remaining[neighs] = False
        if remaining.any():
            rem_idx = np.nonzero(remaining)[0]
            subdeg = A[np.ix_(rem_idx, rem_idx)].sum(axis=1)
            iso = rem_idx[subdeg == 0]
            labels[iso] = 2
            remaining[iso] = False
    labels[labels == -1] = 2
    return labels

def heuristic3(n, A, rng):
    labels = -1 * np.ones(n, dtype=np.int8)
    remaining = np.ones(n, dtype=bool)
    degs = A.sum(axis=1)
    while remaining.any():
        idxs = np.nonzero(remaining)[0]
        if idxs.size == 1:
            labels[idxs[0]] = 2
            remaining[idxs[0]] = False
            break
        rem_degs = degs[idxs]
        maxd = rem_degs.max()
        candidates = idxs[rem_degs == maxd]
        v = rng.choice(candidates)
        labels[v] = 3
        neighs = A[v].astype(bool)
        labels[neighs & remaining] = 0
        remaining[v] = False
        remaining[neighs] = False
        if remaining.any():
            rem_idx = np.nonzero(remaining)[0]
            subdeg = A[np.ix_(rem_idx, rem_idx)].sum(axis=1)
            iso = rem_idx[subdeg == 0]
            labels[iso] = 2
            remaining[iso] = False
    labels[labels == -1] = 2
    return labels

# ---------------------------
# Double Q-Learning
# ---------------------------
def init_population(n, A, pop_size, rng, mix=(0.4, 0.4, 0.2)):
    pop = np.zeros((pop_size, n), dtype=np.int8)
    k1 = int(pop_size * mix[0])
    k2 = int(pop_size * mix[1])
    k3 = int(pop_size * mix[2])
    idx = 0
    for _ in range(k1):
        pop[idx] = heuristic1(n, A, rng)
        idx += 1
    for _ in range(k2):
        pop[idx] = heuristic2(n, A, rng)
        idx += 1
    for _ in range(k3):
        pop[idx] = heuristic3(n, A, rng)
        idx += 1
    while idx < pop_size:
        pop[idx] = heuristic1(n, A, rng)
        idx += 1
    pop = make_feasible(pop, A, rng)
    return pop


class GraphEnvironment:
    def __init__(self, A: np.ndarray):
        self.A = A
        self.n_nodes = A.shape[0]

    def step(self, labels: np.ndarray, action: tuple):
        node, new_label = action
        new_labels = labels.copy()
        new_labels[node] = new_label
        reward = -new_labels.sum()
        done = feasibility_mask(new_labels, self.A).all()
        if not done:
            reward -= 100
        return new_labels, reward, done

class DoubleQLearning:
    def __init__(self, env: GraphEnvironment, n_labels=4, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.env = env
        self.n_nodes = env.n_nodes
        self.n_labels = n_labels
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q1 = np.zeros((self.n_nodes, n_labels))
        self.Q2 = np.zeros((self.n_nodes, n_labels))

    def choose_action(self, labels: np.ndarray, node: int):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_labels)
        return np.argmax(self.Q1[node] + self.Q2[node])

    def generate_population(self, n_pop=20, max_steps=50):
        pop = []
        for _ in range(n_pop):
            labels = np.zeros(self.n_nodes, dtype=int)
            for _ in range(max_steps):
                node = np.random.randint(self.n_nodes)
                action_label = self.choose_action(labels, node)
                labels, _, done = self.env.step(labels, (node, action_label))
                if done:
                    break
            pop.append(make_feasible(labels, self.env.A, np.random.default_rng(), max_iter=5))
        return np.array(pop, dtype=np.int8)



# ---------------------------
# GA main with QL-pop reset
# ---------------------------
def ga_mdr_with_ql_reset(A: np.ndarray,
                          pop_size: int = 200,
                          generations: int = 200,
                          reinit_frac: float = 0.15,
                          reinit_patience: int = 40,
                          ql_pop_size: int = 20,
                          rng_seed: Optional[int] = 42,
                          verbose: bool = True):
    rng = np.random.default_rng(rng_seed)
    n = A.shape[0]

    # Step 1: init population hoàn toàn bằng heuristic
    pop = init_population(n, A, pop_size, rng)
    fitness = fitness_function(pop, A)
    best_idx = int(np.argmin(fitness))
    best_sol = pop[best_idx].copy()
    best_w = float(fitness[best_idx])

    no_improve_counter = 0
    best_since = best_w

    # Step 2: khởi tạo agent Double Q-Learning (dùng cho reset)
    env = GraphEnvironment(A)
    ql_agent = DoubleQLearning(env)
    ql_pop = ql_agent.generate_population(n_pop=ql_pop_size)

    for gen in range(1, generations + 1):
        new_pop = np.empty_like(pop)

        # Elitism: copy 5% tốt nhất
        elite_count = max(1, int(pop_size*0.05))
        elite_idx = np.argsort(fitness)[:elite_count]
        new_pop[:elite_count] = pop[elite_idx].copy()
        i = elite_count

        # Sinh cá thể mới: crossover + mutation
        while i < pop_size:
            p1_idx = rng.integers(0, pop_size)
            p2_idx = rng.integers(0, pop_size)
            while p2_idx == p1_idx:
                p2_idx = rng.integers(0, pop_size)
            child1, child2 = uniform_crossover(pop[p1_idx], pop[p2_idx], rng)
            child1 = mutate(child1, 0.03, rng)
            child2 = mutate(child2, 0.03, rng)
            new_pop[i] = child1
            i += 1
            if i < pop_size:
                new_pop[i] = child2
                i += 1

        # Repair
        new_pop = make_feasible(new_pop, A, rng)
        pop = new_pop
        fitness = fitness_function(pop, A)

        current_best_idx = int(np.argmin(fitness))
        current_best_w = float(fitness[current_best_idx])

        # Update best
        if current_best_w < best_w - 1e-9:
            best_w = current_best_w
            best_sol = pop[current_best_idx].copy()
            no_improve_counter = 0
            best_since = best_w
            if verbose:
                print(f"gen {gen}: new best {best_w:.3f}")
        else:
            no_improve_counter += 1

        # Reset/reinit bằng QL-pop nếu kẹt
        if no_improve_counter >= reinit_patience:
            n_reinit = max(1, int(pop_size * reinit_frac))
            worst_idx = np.argsort(fitness)[-n_reinit:]
            for pos, idx_w in enumerate(worst_idx):
                pop[idx_w] = ql_pop[pos % ql_pop_size]
            pop = make_feasible(pop, A, rng)
            fitness = fitness_function(pop, A)
            no_improve_counter = 0
            if verbose:
                print(f"gen {gen}: reset {n_reinit} worst individuals using Double Q-Learning")

    # Final repair
    best_sol = make_feasible(best_sol, A, rng, max_iter=10)
    best_w = float(fitness_function(best_sol, A))
    return best_sol, best_w

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    mtx_path = r"D:\Materials\EC_code\data\DROMD\ash85.mtx"
    with open(mtx_path, "r") as f:
        mtx_text = f.read()
    A = parse_mtx_pattern_symmetric(mtx_text)

    best_sol, best_w = ga_mdr_with_ql_reset(A, pop_size=100, generations=2000, ql_pop_size=20)
    print("Best weight:", best_w)
    print("Solution labels:", best_sol)
