import numpy as np
from typing import Tuple


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
    ok = ok & (~is0 | ok0)
    ok = ok & (~is1 | ok1)
    return ok


def is_feasible(labels: np.ndarray, A: np.ndarray) -> bool:
    return feasibility_mask(labels, A).all()


def weight_of(labels: np.ndarray) -> int:
    return int(np.sum(labels))


def make_feasible(labels: np.ndarray, A: np.ndarray, max_iter=5) -> np.ndarray:
    labels = labels.copy()
    for _ in range(max_iter):
        ok = feasibility_mask(labels, A)
        if ok.all():
            break
        bad = ~ok
        labels[bad] = 2
    return labels


def parse_mtx_pattern_symmetric(mtx_text: str) -> np.ndarray:
    lines = mtx_text.strip().splitlines()
    data_lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith('%')]
    header = data_lines[0].split()
    nrows, ncols, nnz = map(int, header[:3])
    if nrows != ncols:
        raise ValueError("Matrix must be square")
    n = nrows
    A = np.zeros((n, n), dtype=np.int8)
    for ln in data_lines[1:]:
        i, j = map(int, ln.split()[:2])
        A[i - 1, j - 1] = 1
        A[j - 1, i - 1] = 1
    return A


def choose_vertex_prob(V_idx, deg, tau, rng, dACO_rate=0.7):
    vals = deg[V_idx] * tau[V_idx]
    if rng.random() <= dACO_rate:
        maxv = vals.max()
        cands = V_idx[vals == maxv]
        return rng.choice(cands)
    else:
        s = vals.sum()
        if s == 0:
            return rng.choice(V_idx)
        probs = vals / s
        return rng.choice(V_idx, p=probs)


def construct_solution(A, tau, rng, dACO_rate=0.7):
    n = A.shape[0]
    labels = -np.ones(n, dtype=np.int8)
    remaining = np.ones(n, dtype=bool)
    deg = A.sum(axis=1)

    while remaining.any():
        rem = np.nonzero(remaining)[0]
        if rem.size == 1:
            labels[rem[0]] = 2
            remaining[rem[0]] = False
            break
        v = choose_vertex_prob(rem, deg, tau, rng, dACO_rate)
        labels[v] = 3
        neighs = A[v].astype(bool)
        labels[neighs & remaining] = 0
        remaining[v] = False
        remaining[neighs] = False

    return make_feasible(labels, A)


def extend_solution(labels, A, tau, rng, r_aug=0.05, drate=0.9):
    labels = labels.copy()
    V02 = np.where((labels == 0) | (labels == 2))[0]
    if V02.size == 0:
        return labels
    iters = int(np.floor(r_aug * V02.size))
    deg = A.sum(axis=1)
    for _ in range(iters):
        if V02.size == 0:
            break
        v = choose_vertex_prob(V02, deg, tau, rng, drate)
        labels[v] = 3
        neighs = A[v].astype(bool)
        labels[neighs & (labels == -1)] = 0
        V02 = np.where((labels == 0) | (labels == 2))[0]
    return make_feasible(labels, A)


def reduce_solution(labels, A):
    labels = labels.copy()
    n = A.shape[0]
    deg = A.sum(axis=1)
    order = np.argsort(deg)
    for u in order:
        if labels[u] in (3, 2):
            orig = labels[u]
            labels[u] = 0
            if not is_feasible(labels, A):
                labels[u] = orig
    return labels


def destroy_solution(labels, A, rng, dmin=0.2, dmax=0.5):
    labels = labels.copy()
    n = A.shape[0]
    d = rng.uniform(dmin, dmax)
    num = int(np.ceil(n * d))
    cand = np.where((labels == 0) | (labels == 2))[0]
    if cand.size == 0:
        return labels
    chosen = rng.choice(cand, size=min(num, cand.size), replace=False)
    labels[chosen] = -1
    unl = np.where(labels == -1)[0]
    if unl.size:
        subA = A[np.ix_(unl, unl)]
        tau_sub = np.ones(unl.size)
        sub_labels = construct_solution(subA, tau_sub, rng)
        labels[unl] = sub_labels
    return make_feasible(labels, A)


def random_variable_neighbourhood_search(labels, A, rng, k_max=5, dmin=0.2, dmax=0.5,
                                         max_no_improve=10, max_itr=150, drate=0.9, r_aug=0.05):
    best = labels.copy()
    best_w = weight_of(best)
    k = 1
    no_impr = 0
    itr = max_itr
    while no_impr < max_no_improve and itr > 0:
        Sprime = destroy_solution(best, A, rng, dmin=dmin, dmax=dmax)
        Sprime = extend_solution(Sprime, A, np.ones(A.shape[0]), rng, r_aug=r_aug, drate=drate)
        Sprime = reduce_solution(Sprime, A)
        Sprime = make_feasible(Sprime, A)
        w = weight_of(Sprime)
        if w < best_w:
            best, best_w = Sprime, w
            k = 1
            no_impr = 0
        else:
            k = min(k + 1, k_max)
            no_impr += 1
        itr -= 1
    return best


def compute_conv_fact(tau):
    n = tau.size
    tmax, tmin = tau.max(), tau.min()
    s = np.abs(tau - (tmax + tmin) / 2).sum()
    val = 2 * (s / (n * (tmax + tmin + 1e-12))) - 1
    return float(np.clip(val, 0.0, 1.0))


def update_pheromone(tau, curr_best, best, rho=0.2, r_aug=0.05):
    tau = (1.0 - rho) * tau
    kb, kc = weight_of(best), weight_of(curr_best)
    add_best = np.where(best == 3, 1.0 / kb, np.where(best == 2, 0.5 / kb, 0.0))
    add_curr = np.where(curr_best == 3, 1.0 / kc, np.where(curr_best == 2, 0.5 / kc, 0.0))
    tau += rho * (add_best + add_curr) * r_aug
    return np.clip(tau, 1e-6, 1.0 - 1e-6)


def aco_mdr(A, num_ants=5, outer_iters=40, inner_iters=5, rho=0.2,
            dACO_rate=0.7, r_aug=0.05, dmin=0.2, dmax=0.5,
            max_no_impr=10, max_itr_rvns=150, rng_seed=None, verbose=False):
    rng = np.random.default_rng(rng_seed)
    n = A.shape[0]
    tau = np.full(n, 0.5)
    best_overall = None
    best_w = np.inf
    history = []
    for out in range(outer_iters):
        curr_best = None
        curr_best_w = np.inf
        for _ in range(inner_iters):
            S = construct_solution(A, tau, rng, dACO_rate)
            S = extend_solution(S, A, tau, rng, r_aug, dACO_rate)
            S = reduce_solution(S, A)
            S = random_variable_neighbourhood_search(S, A, rng,
                                                     k_max=5, dmin=dmin, dmax=dmax,
                                                     max_no_improve=max_no_impr,
                                                     max_itr=max_itr_rvns,
                                                     drate=dACO_rate, r_aug=r_aug)
            w = weight_of(S)
            if w < curr_best_w:
                curr_best, curr_best_w = S, w
        if curr_best_w < best_w:
            best_overall, best_w = curr_best.copy(), curr_best_w
        tau = update_pheromone(tau, curr_best, best_overall, rho, r_aug)
        conv = compute_conv_fact(tau)
        if verbose:
            print(f"Iter {out+1}/{outer_iters}: curr_best={curr_best_w}, best={best_w}, conv={conv:.4f}")
        if conv > 0.99:
            tau[:] = 0.5
            if verbose:
                print("Pheromone reset due to stagnation.")
        history.append(best_w)
    return best_overall, best_w, history


if __name__ == "__main__":
    # mtx_path = r"D:\Materials\EC_code\data\DROMD\can___61.mtx"
    mtx_path = r"D:\Materials\EC_code\data\DROMD\ash85.mtx"
    with open(mtx_path, "r") as f:
        mtx_text = f.read()
    A = parse_mtx_pattern_symmetric(mtx_text)
    best, w, hist = aco_mdr(A, num_ants=10, outer_iters=30, inner_iters=5, rng_seed=42, verbose=True)
    print("Best weight:", w)
    print("Best labels:", best)
