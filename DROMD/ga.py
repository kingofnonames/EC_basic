# import numpy as np

# np.random.seed(42)


# def weights(pop: np.ndarray) -> np.ndarray:
#     """Trả về tổng label trên mỗi cá thể (shape (pop_size,))."""
#     return pop.sum(axis=1)


# def neighbor_counts(labels: np.ndarray, A: np.ndarray):
#     pop_eq3 = (labels == 3).astype(np.int32)
#     pop_eq2 = (labels == 2).astype(np.int32)
#     pop_ge2 = (labels >= 2).astype(np.int32)

#     # nếu labels shape = (n,) thì kết quả là (n,), nếu labels shape = (pop_size, n) 
#     # thì cần xử lý ngoài (hiện bạn dùng cho vector đơn lẻ hoặc ma trận pop)
#     count3 = pop_eq3 @ A
#     count2 = pop_eq2 @ A
#     count_ge2 = pop_ge2 @ A

#     return count3, count2, count_ge2


# def feasibility_mask(labels: np.ndarray, A: np.ndarray):
#     """Trả về mask boolean cùng shape với labels: True nếu vị trí đó 'ok'."""
#     count3, count2, count_ge2 = neighbor_counts(labels, A)
#     is0 = (labels == 0)
#     is1 = (labels == 1)
#     ok0 = (count3 >= 1) | (count2 >= 2)
#     ok1 = (count_ge2 >= 1)
#     ok = np.ones_like(labels, dtype=bool)
#     ok = ok & (~is0 | ok0)
#     ok = ok & (~is1 | ok1)
#     return ok


# def make_feasible(labels: np.ndarray, A: np.ndarray, max_iter=5):
#     """
#     Trả về một bản sao feasible của labels (không sửa labels tại chỗ).
#     labels có thể là shape (n,) hoặc (pop_size, n).
#     Nếu là (pop_size, n), chúng ta xử lý hàng theo hàng.
#     """
#     pop = labels.copy()
#     # nếu pop 2D (pop_size, n), xử lý theo cá thể
#     if pop.ndim == 2:
#         for i in range(pop.shape[0]):
#             for _ in range(max_iter):
#                 ok = feasibility_mask(pop[i], A)
#                 if ok.all():
#                     break
#                 bad_position = ~ok
#                 pop[i, bad_position] = 2
#     else:
#         # pop 1D
#         for _ in range(max_iter):
#             ok = feasibility_mask(pop, A)
#             if ok.all():
#                 break
#             bad_position = ~ok
#             pop[bad_position] = 2
#     return pop


# def heuristic1(n, A, rng):
#     labels = -1 * np.ones(n, dtype=np.int8)
#     remaining = np.ones(n, dtype=bool)
#     while remaining.any():
#         idxs = np.nonzero(remaining)[0]
#         if idxs.size == 1:
#             labels[idxs[0]] = 2
#             remaining[idxs[0]] = False
#             break
#         v = rng.choice(idxs)
#         labels[v] = 3
#         neighs = A[v].astype(bool)
#         labels[neighs & remaining] = 0
#         remaining[v] = False
#         remaining[neighs] = False
#     return labels


# def heuristic2(n, A, rng):
#     labels = -1 * np.ones(n, dtype=np.int8)
#     remaining = np.ones(n, dtype=bool)
#     while remaining.any():
#         idxs = np.nonzero(remaining)[0]
#         v = rng.choice(idxs)
#         labels[v] = 3
#         neighs = A[v].astype(bool)
#         labels[neighs & remaining] = 0
#         remaining[v] = False
#         remaining[neighs] = False

#         if remaining.any():
#             rem_idx = np.nonzero(remaining)[0]
#             subdeg = A[np.ix_(rem_idx, rem_idx)].sum(axis=1)
#             iso = rem_idx[subdeg == 0]
#             labels[iso] = 2
#             remaining[iso] = False
#     return labels


# def heuristic3(n, A, rng):
#     labels = -1 * np.ones(n, dtype=np.int8)
#     remaining = np.ones(n, dtype=bool)
#     degs = A.sum(axis=1)
#     while remaining.any():
#         idxs = np.nonzero(remaining)[0]
#         if idxs.size == 1:
#             labels[idxs[0]] = 2
#             remaining[idxs[0]] = False
#             break

#         rem_degs = degs[idxs]
#         maxd = rem_degs.max()
#         candidates = idxs[rem_degs == maxd]
#         v = rng.choice(candidates)
#         labels[v] = 3
#         neighs = A[v].astype(bool)
#         labels[neighs & remaining] = 0
#         remaining[v] = False
#         remaining[neighs] = False

#         if remaining.any():
#             rem_idx = np.nonzero(remaining)[0]
#             subdeg = A[np.ix_(rem_idx, rem_idx)].sum(axis=1)
#             iso = rem_idx[subdeg == 0]
#             labels[iso] = 2
#             remaining[iso] = False
#     return labels


# def init_population(n, A, pop_size, rng, mix=(0.4, 0.4, 0.2)):
#     pop = np.zeros((pop_size, n), dtype=np.int8)
#     k1 = int(pop_size * mix[0])
#     k2 = int(pop_size * mix[1])
#     k3 = int(pop_size * mix[2])
#     idx = 0
#     for _ in range(k1):
#         pop[idx] = heuristic1(n, A, rng)
#         idx += 1
#     for _ in range(k2):
#         pop[idx] = heuristic2(n, A, rng)
#         idx += 1
#     for _ in range(k3):
#         pop[idx] = heuristic3(n, A, rng)
#         idx += 1
#     while idx < pop_size:
#         pop[idx] = heuristic1(n, A, rng)
#         idx += 1

#     pop = make_feasible(pop, A)
#     return pop


# def tournament_select(pop, fitness, k, rng):
#     pop_size = pop.shape[0]
#     candidates = rng.integers(0, pop_size, size=k)
#     best = candidates[np.argmin(fitness[candidates])]
#     return best


# def roulette_select(fitness, rng):
#     scores = 1.0 / (1.0 + fitness)
#     probs = scores / scores.sum()
#     return rng.choice(len(fitness), p=probs)


# def two_point_crossover(parent_a, parent_b, rng):
#     n = parent_a.size
#     c1 = rng.integers(0, n)
#     c2 = rng.integers(0, n)
#     if c1 == c2:
#         c2 = (c1 + 1) % n
#     if c1 > c2:
#         c1, c2 = c2, c1
#     a = parent_a.copy()
#     b = parent_b.copy()
#     a[c1:c2], b[c1:c2] = b[c1:c2].copy(), a[c1:c2].copy()
#     return a, b


# def mutate(solution, mut_rate, rng):
#     n = solution.size
#     mask = rng.random(n) < mut_rate
#     if mask.any():
#         solution[mask] = rng.integers(0, 4, size=mask.sum(), dtype=np.int8)
#     return solution


# def ga_mdr(A,
#            pop_size=200,
#            generation=200,
#            tournament_k=3,
#            crossover_rate=0.9,
#            mutation_rate=0.02,
#            rng_seed=42,
#            init_mix=(0.4, 0.4, 0.2),
#            verbose=False
#            ):
#     rng = np.random.default_rng(rng_seed)
#     n = A.shape[0]
#     pop = init_population(n, A, pop_size, rng, init_mix)
#     fitness = weights(pop)
#     pop = make_feasible(pop, A)
#     fitness = weights(pop)
#     best_idx = int(np.argmin(fitness))
#     best_sol = pop[best_idx].copy()
#     best_w = int(fitness[best_idx])
#     history = [best_w]

#     if verbose:
#         print(f"Init best weight = {best_w}")

#     for gen in range(generation):
#         new_pop = np.empty_like(pop)
#         i = 0
#         new_pop[0] = best_sol.copy()
#         i = 1
#         while i < pop_size:
#             p1_idx = tournament_select(pop, fitness, tournament_k, rng)
#             p2_idx = roulette_select(fitness, rng)
#             while p2_idx == p1_idx:
#                 p2_idx = roulette_select(fitness, rng)
#             parent1, parent2 = pop[p1_idx], pop[p2_idx]
#             if rng.random() < crossover_rate:
#                 child1, child2 = two_point_crossover(parent1, parent2, rng)
#             else:
#                 child1 = parent1.copy()
#                 child2 = parent2.copy()
#             child1 = mutate(child1, mutation_rate, rng)
#             child2 = mutate(child2, mutation_rate, rng)
#             if i < pop_size:
#                 new_pop[i] = child1
#                 i += 1
#             if i < pop_size:
#                 new_pop[i] = child2
#                 i += 1
#         new_pop = make_feasible(new_pop, A)
#         pop = new_pop
#         fitness = weights(pop)
#         current_best_idx = int(np.argmin(fitness))
#         current_best_w = int(fitness[current_best_idx])
#         if current_best_w < best_w:
#             best_w = current_best_w
#             best_sol = pop[current_best_idx].copy()
#             if verbose:
#                 print(f"gen {gen + 1}: new best {best_w}")
#         history.append(best_w)
#     return best_sol, best_w, history


# def parse_mtx_pattern_symmetric(mtx_text: str) -> np.ndarray:

#     lines = mtx_text.strip().splitlines()
#     data_lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith('%')]

#     if not data_lines:
#         raise ValueError("No data lines found in MTX text")
#     header = data_lines[0].split()
#     if len(header) < 3:
#         raise ValueError("Header line must be 'nrows ncols nnz'")
#     nrows, ncols, nnz = map(int, header[:3])
#     if nrows != ncols:
#         raise ValueError("Expected a square matrix (nrows == ncols) for adjacency")
#     n = nrows
#     A = np.zeros((n, n), dtype=np.int8)
#     for ln in data_lines[1:]:
#         parts = ln.split()
#         if len(parts) < 2:
#             continue
#         i = int(parts[0]) - 1
#         j = int(parts[1]) - 1
#         if i < 0 or j < 0 or i >= n or j >= n:
#             raise IndexError(f"Index out of range in line: {ln}")
#         A[i, j] = 1
#         A[j, i] = 1

#     return A

# if __name__ == "__main__":
#     rng = np.random.default_rng(42)
#     mtx_path = r"D:\Materials\EC_code\data\DROMD\can___61.mtx"
#     # mtx_path = r"D:\Materials\EC_code\data\DROMD\ash85.mtx"
#     with open(mtx_path, "r") as f:
#         mtx_text = f.read()
#     A = parse_mtx_pattern_symmetric(mtx_text)

#     sol, w, hist = ga_mdr(A, pop_size=100, generation=500, rng_seed=7, verbose=True)
#     print("best weight", w)
#     print("solution", sol)


"""
GA MDR (improved) - Full script
- Hỗ trợ đọc MatrixMarket 'coordinate pattern symmetric' từ file hoặc chuỗi.
- Fitness = sum(labels) + penalty * (# vi phạm feasibility)
- Elitism, rank selection, uniform crossover, light mutation.
- make_feasible vectorized (hỗ trợ pop shape (pop_size, n)).
- Reinit nếu không cải thiện trong vài thế hệ (diversity boost).
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------
# Utility: parse .mtx (pattern symmetric)
# ---------------------------
def parse_mtx_pattern_symmetric(mtx_text: str) -> np.ndarray:
    lines = mtx_text.strip().splitlines()
    data_lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith('%')]
    if not data_lines:
        raise ValueError("No data lines found in MTX text")
    header = data_lines[0].split()
    if len(header) < 3:
        raise ValueError("Header line must be 'nrows ncols nnz'")
    nrows, ncols, nnz = map(int, header[:3])
    if nrows != ncols:
        raise ValueError("Expected a square matrix (nrows == ncols) for adjacency")
    n = nrows
    A = np.zeros((n, n), dtype=np.int8)
    for ln in data_lines[1:]:
        parts = ln.split()
        if len(parts) < 2:
            continue
        i = int(parts[0]) - 1
        j = int(parts[1]) - 1
        if i < 0 or j < 0 or i >= n or j >= n:
            raise IndexError(f"Index out of range in line: {ln}")
        A[i, j] = 1
        A[j, i] = 1
    return A


# ---------------------------
# Core functions (vectorized where appropriate)
# ---------------------------
def weights(pop: np.ndarray) -> np.ndarray:
    """Tính tổng nhãn cho từng cá thể trong pop.
    pop shape: (pop_size, n) hoặc (n,) -> trả về (pop_size,) hoặc scalar array"""
    if pop.ndim == 1:
        return np.array([pop.sum()])
    return pop.sum(axis=1)


def neighbor_counts(labels: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Nếu labels shape = (n,) => trả về (n,) arrays
    Nếu labels shape = (pop_size, n) => trả về (pop_size, n) arrays
    """
    pop_eq3 = (labels == 3).astype(np.int32)
    pop_eq2 = (labels == 2).astype(np.int32)
    pop_ge2 = (labels >= 2).astype(np.int32)
    # Use matrix multiplication; if pop_eq3 is 2D, result is (pop_size, n)
    count3 = pop_eq3 @ A
    count2 = pop_eq2 @ A
    count_ge2 = pop_ge2 @ A
    return count3, count2, count_ge2


def feasibility_mask(labels: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Trả về mask boolean cùng shape với labels:
    True nếu vị trí 'ok' (khả thi theo quy tắc trong code gốc).
    Hỗ trợ labels 1D hoặc 2D.
    """
    count3, count2, count_ge2 = neighbor_counts(labels, A)
    is0 = (labels == 0)
    is1 = (labels == 1)
    ok0 = (count3 >= 1) | (count2 >= 2)
    ok1 = (count_ge2 >= 1)
    ok = np.ones_like(labels, dtype=bool)
    ok = ok & (~is0 | ok0)
    ok = ok & (~is1 | ok1)
    return ok


def make_feasible(labels: np.ndarray, A: np.ndarray, rng: np.random.Generator, max_iter=5) -> np.ndarray:
    """
    Trả về bản sao feasible của labels (không sửa tại chỗ).
    Nếu labels 2D (pop_size, n) xử lý đồng thời.
    Tuy nhiên sửa lặp lại tối đa max_iter lần để cố đạt feasibility.
    Khi sửa, thay vì set cứng về 2, ta random trong {1,2,3} để tránh đông cứng.
    """
    pop = labels.copy()
    if pop.ndim == 1:
        pop = pop[None, :]  # shape -> (1, n) for uniform processing

    pop_size, n = pop.shape
    for it in range(max_iter):
        ok = feasibility_mask(pop, A)  # shape (pop_size, n)
        all_ok = ok.all(axis=1)
        if all(all_ok):
            break
        # For each individual, set bad positions randomly to 1/2/3
        bad_mask = ~ok  # True where violation
        # sample values in {1,2,3} for each bad entry
        # generate integers in [1,4)
        rnd_vals = rng.integers(1, 4, size=bad_mask.sum(), dtype=np.int8)
        pop[bad_mask] = rnd_vals
    if pop.shape[0] == 1:
        return pop[0]
    return pop


# ---------------------------
# Heuristics (keep originals but return feasible outputs)
# ---------------------------
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
    # fill any -1 with 2 (conservative)
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
# Population init
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


# ---------------------------
# Selection / Crossover / Mutation improvements
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


# ---------------------------
# Fitness (with penalty)
# ---------------------------
def fitness_function(pop: np.ndarray, A: np.ndarray, penalty_weight: float = 20.0) -> np.ndarray:
    """
    pop: (pop_size, n) or (n,)
    Returns: fitness array (pop_size,)
    """
    pop_arr = pop if pop.ndim == 2 else pop[None, :]
    base = pop_arr.sum(axis=1).astype(float)  # sum of labels
    ok_mask = feasibility_mask(pop_arr, A)  # (pop_size, n)
    violations = (~ok_mask).sum(axis=1).astype(float)
    return base + penalty_weight * violations


# ---------------------------
# GA main (improved)
# ---------------------------
def ga_mdr_improved(A: np.ndarray,
                    pop_size: int = 200,
                    generations: int = 200,
                    tournament_k: int = 3,
                    crossover_rate: float = 0.9,
                    mutation_rate: float = 0.02,
                    rng_seed: Optional[int] = 42,
                    init_mix: Tuple[float, float, float] = (0.4, 0.4, 0.2),
                    penalty_weight: float = 20.0,
                    elite_frac: float = 0.05,
                    reinit_patience: int = 50,
                    reinit_frac: float = 0.15,
                    verbose: bool = False
                    ):
    rng = np.random.default_rng(rng_seed)
    n = A.shape[0]

    # init
    pop = init_population(n, A, pop_size, rng, init_mix)
    fitness = fitness_function(pop, A, penalty_weight)
    best_idx = int(np.argmin(fitness))
    best_sol = pop[best_idx].copy()
    best_w = float(fitness[best_idx])
    history = [best_w]

    elite_count = max(1, int(pop_size * elite_frac))

    # for reinit check
    no_improve_counter = 0
    best_since = best_w

    if verbose:
        print(f"[Init] best fitness = {best_w:.3f}")

    for gen in range(1, generations + 1):
        new_pop = np.empty_like(pop)

        # Elitism: copy top-k
        elite_idx = np.argsort(fitness)[:elite_count]
        new_pop[:elite_count] = pop[elite_idx].copy()
        i = elite_count

        # produce rest
        while i < pop_size:
            # parent selection: tournament + rank for diversity
            p1_idx = tournament_select(pop, fitness, tournament_k, rng)
            p2_idx = rank_select(fitness, rng)
            while p2_idx == p1_idx:
                p2_idx = rank_select(fitness, rng)
            parent1, parent2 = pop[p1_idx], pop[p2_idx]

            if rng.random() < crossover_rate:
                child1, child2 = uniform_crossover(parent1, parent2, rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = mutate(child1, mutation_rate, rng)
            child2 = mutate(child2, mutation_rate, rng)

            # next fill
            new_pop[i] = child1
            i += 1
            if i < pop_size:
                new_pop[i] = child2
                i += 1

        # enforce feasibility (repair)
        new_pop = make_feasible(new_pop, A, rng, max_iter=5)

        pop = new_pop
        fitness = fitness_function(pop, A, penalty_weight)

        current_best_idx = int(np.argmin(fitness))
        current_best_w = float(fitness[current_best_idx])

        # update overall best
        if current_best_w < best_w - 1e-9:
            best_w = current_best_w
            best_sol = pop[current_best_idx].copy()
            no_improve_counter = 0
            best_since = best_w
            if verbose:
                print(f"gen {gen}: new best {best_w:.3f}")
        else:
            no_improve_counter += 1

        # reinit if stuck
        if no_improve_counter >= reinit_patience:
            n_reinit = max(1, int(pop_size * reinit_frac))
            if verbose:
                print(f"gen {gen}: no improve for {no_improve_counter} gens -> reinit {n_reinit} individuals")
            # replace worst n_reinit individuals with heuristics
            worst_idx = np.argsort(fitness)[-n_reinit:]
            for pos in worst_idx:
                # randomly pick a heuristic
                h = rng.choice([heuristic1, heuristic2, heuristic3])
                pop[pos] = h(n, A, rng)
            pop = make_feasible(pop, A, rng, max_iter=3)
            fitness = fitness_function(pop, A, penalty_weight)
            no_improve_counter = 0  # reset

        history.append(best_w)

        if verbose and (gen % max(1, generations // 10) == 0 or gen <= 10):
            print(f"gen {gen:4d} | best {best_w:.3f} | mean {fitness.mean():.3f} | std {fitness.std():.3f}")

    # final ensure feasible best_sol
    best_sol = make_feasible(best_sol, A, rng, max_iter=10)
    best_w = float(fitness_function(best_sol, A, penalty_weight))
    return best_sol, best_w, history


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":

    mtx_path = r"D:\Materials\EC_code\data\DROMD\ash85.mtx"
    with open(mtx_path, "r") as f:
        mtx_text = f.read()
    A = parse_mtx_pattern_symmetric(mtx_text)

    sol, w, hist = ga_mdr_improved(A,
                                  pop_size=200,
                                  generations=500,
                                  tournament_k=3,
                                  crossover_rate=0.9,
                                  mutation_rate=0.03,
                                  rng_seed=7,
                                  penalty_weight=30.0,
                                  elite_frac=0.05,
                                  reinit_patience=40,
                                  reinit_frac=0.4,
                                  verbose=True)
    print("\nKẾT QUẢ:")
    print("best fitness:", w)
    print("solution labels:", sol)
    print("history (first 10):", hist[:10])
    print("history (last 10):", hist[-10:])
