
import math
import time
import numpy as np
from collections import defaultdict

def build_distance_matrix(coords):
    X = np.asarray(coords, dtype=float)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))

def tour_length(tour, DIST):
    arr = np.asarray(tour, dtype=int)
    return float(np.sum(DIST[arr, np.roll(arr, -1)]))

def undirected_edge_set(tour):
    n = len(tour)
    edges = set()
    for i in range(n):
        a = int(tour[i])
        b = int(tour[(i + 1) % n])
        if a < b:
            edges.add((a, b))
        else:
            edges.add((b, a))
    return edges

def dissimilarity(tour1, tour2):
    e1 = undirected_edge_set(tour1)
    e2 = undirected_edge_set(tour2)
    # dùng symmetric difference để đo khác biệt
    diff = e1.symmetric_difference(e2)
    return len(diff)

def nearest_neighbor_tour(start, n, DIST):
    tour = [int(start)]
    unvisited = set(range(n))
    unvisited.remove(start)
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda x: DIST[cur, x])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour

# ---------------- EAX core ----------------

def build_adj_map(tour):
    n = len(tour)
    adj = defaultdict(list)
    for i, v in enumerate(tour):
        a = int(v)
        b = int(tour[(i + 1) % n])
        adj[a].append(b)
        adj[b].append(a)
    return adj  # dict-like with small lists of length <=2

def generate_ab_cycles(parentA, parentB):
    n = len(parentA)
    eA = undirected_edge_set(parentA)
    eB = undirected_edge_set(parentB)
    edge_diff = eA.symmetric_difference(eB)
    if not edge_diff:
        return []

    adjA = build_adj_map(parentA)
    adjB = build_adj_map(parentB)

    diff_adj = defaultdict(list)
    for (u, v) in edge_diff:
        diff_adj[u].append(v)
        diff_adj[v].append(u)

    cycles = []
    visited_nodes = set()

    for start in diff_adj.keys():
        if start in visited_nodes:
            continue

        for start_mode in ('A', 'B'):
            mode = start_mode
            cycle = [start]
            cur = start
            used_edges = set()
            while True:
                neighbors = adjA[cur] if mode == 'A' else adjB[cur]
                next_node = None
                for nb in neighbors:
                    e = (min(cur, nb), max(cur, nb))
                    if e not in edge_diff:
                        continue
                    if (cur, nb) in used_edges or (nb, cur) in used_edges:
                        continue
                    next_node = nb
                    break
                if next_node is None:
                    break
                used_edges.add((cur, next_node))
                cycle.append(next_node)
                cur = next_node
                mode = 'B' if mode == 'A' else 'A'
                if cur == start:
                    break
            if len(cycle) >= 4 and cycle[0] == cycle[-1]:
                cyc = cycle[:-1]
                cycles.append(cyc)
                for node in cyc:
                    visited_nodes.add(node)
                break
    return cycles

def apply_cycles_make_edges(parentA, parentB, selected_cycles):
    edgesA = undirected_edge_set(parentA)
    edgesB = undirected_edge_set(parentB)
    new_edges = set(edgesA)
    for cyc in selected_cycles:
        m = len(cyc)
        for i in range(m):
            u, v = int(cyc[i]), int(cyc[(i + 1) % m])
            e = (u, v) if u < v else (v, u)
            if e in edgesA and e in new_edges:
                new_edges.discard(e)
            elif e in edgesB:
                new_edges.add(e)
    return new_edges

def subtours_from_edges(edges, n):
    adj = defaultdict(list)
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = set()
    tours = []
    for i in range(n):
        if i in visited:
            continue
        if i not in adj:
            tours.append([i])
            visited.add(i)
            continue
        comp = []
        stack = [i]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.append(v)
            for nb in adj[v]:
                if nb not in visited:
                    stack.append(nb)
        tours.append(comp)
    return tours

def connect_subtours_greedy(tours, DIST):
    tours_cycle = [list(t) for t in tours]
    while len(tours_cycle) > 1:
        best_inc = float('inf')
        best_choice = None
        K = len(tours_cycle)
        for i in range(K):
            Ti = tours_cycle[i]
            lenTi = len(Ti)
            for j in range(i + 1, K):
                Tj = tours_cycle[j]
                lenTj = len(Tj)
                for pa in range(lenTi):
                    a = Ti[pa]; a_next = Ti[(pa + 1) % lenTi]
                    for qb in range(lenTj):
                        b = Tj[qb]; b_next = Tj[(qb + 1) % lenTj]
                        before = DIST[a, a_next] + DIST[b, b_next]
                        after1 = DIST[a, b] + DIST[a_next, b_next]
                        after2 = DIST[a, b_next] + DIST[a_next, b]
                        if after1 < after2:
                            after, flip = after1, False
                        else:
                            after, flip = after2, True
                        inc = after - before
                        if inc < best_inc:
                            best_inc = inc
                            best_choice = (i, j, pa, qb, flip)
        if best_choice is None:
            tours_cycle[0].extend(tours_cycle[-1])
            tours_cycle.pop()
            continue
        i, j, pa, qb, flip = best_choice
        Ti, Tj = tours_cycle[i], tours_cycle[j]
        Ti_rot = Ti[pa + 1:] + Ti[:pa + 1]
        Tj_rot = Tj[qb + 1:] + Tj[:qb + 1]
        if flip:
            Tj_rot.reverse()
        merged = Ti_rot + Tj_rot
        if i < j:
            tours_cycle[i] = merged
            del tours_cycle[j]
        else:
            tours_cycle[j] = merged
            del tours_cycle[i]
    return tours_cycle[0]

def reconnect_tour_from_edges(edges, n, DIST):
    tours = subtours_from_edges(edges, n)
    if len(tours) == 1:
        return tours[0]
    return connect_subtours_greedy(tours, DIST)

def h_eax_crossover(parentA, parentB, DIST, rng, cycle_select_rate=0.5):
    n = len(parentA)
    cycles = generate_ab_cycles(parentA, parentB)
    if not cycles:
        return list(parentA)
    k = max(1, int(math.ceil(len(cycles) * cycle_select_rate)))
    if len(cycles) > 1:
        idxs = rng.choice(len(cycles), size=k, replace=False)
        selected = [cycles[i] for i in idxs]
    else:
        selected = [cycles[0]]
    new_edges = apply_cycles_make_edges(parentA, parentB, selected)
    child = reconnect_tour_from_edges(new_edges, n, DIST)
    if len(set(child)) != n:
        child = nearest_neighbor_tour(parentA[0], n, DIST)
    return child


def two_opt(tour, DIST, max_iter=1000):
    n = len(tour)
    if n <= 3:
        return tour
    best = tour[:]
    best_len = tour_length(best, DIST)
    it = 0
    improved = True
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, n - 2):
            a, a_next = best[i - 1], best[i]
            for j in range(i + 1, n):
                if j == n - 1 and i == 0:
                    continue
                b, b_next = best[j - 1], best[j % n]
                delta = DIST[a, b] + DIST[a_next, b_next] - DIST[a, a_next] - DIST[b, b_next]
                if delta < -1e-12:
                    new_tour = best[:i] + best[i:j][::-1] + best[j:]
                    best = new_tour
                    best_len += delta
                    improved = True
                    break
            if improved:
                break
    return best

def heterogeneous_pairing_selection(population, rng):
    p1 = rng.choice(population)
    p2 = max(population, key=lambda x: dissimilarity(p1, x))
    if p1 is p2:
        cand = [p for p in population if p is not p1]
        p2 = rng.choice(cand)
    return p1, p2

def family_competition_replace(A_curr, C_new, DIST):
    next_pop = []
    for a, c in zip(A_curr, C_new):
        la, lc = tour_length(a, DIST), tour_length(c, DIST)
        next_pop.append(c if lc <= la else a)
    return next_pop

def RMGA_EAX_only(coords, pop_size=50, generation=100, cycle_select_rate=0.3,
                  twoopt_after=True, twoopt_maxiter=200, seed=None, verbose=True,
                  benchmark=False):
    rng = np.random.default_rng(seed)
    n = len(coords)
    DIST = build_distance_matrix(coords)

    population = [rng.permutation(n).tolist() for _ in range(pop_size)]
    best_history = []

    t_all_start = time.time()
    for gen in range(generation):
        t_gen_start = time.time()
        children = []
        for _ in range(pop_size):
            p1, p2 = heterogeneous_pairing_selection(population, rng)
            child = h_eax_crossover(p1, p2, DIST, rng, cycle_select_rate)
            if twoopt_after:
                child = two_opt(child, DIST, max_iter=twoopt_maxiter)
            children.append(child)
        population = family_competition_replace(population, children, DIST)
        fitness = [tour_length(t, DIST) for t in population]
        best = min(fitness)
        best_history.append(best)

        t_gen_end = time.time()
        if verbose and (gen % 10 == 0 or gen < 5):
            print(f"Gen {gen:4d} | best = {best:.4f} | avg = {np.mean(fitness):.4f} | gen_time = {t_gen_end - t_gen_start:.2f}s")
        elif benchmark:
            print(f"Gen {gen:4d} | best = {best:.4f} | gen_time = {t_gen_end - t_gen_start:.2f}s")

    t_all_end = time.time()
    if benchmark:
        print(f"Total time: {t_all_end - t_all_start:.2f}s")
    best_idx = int(np.argmin(fitness))
    return population[best_idx], fitness[best_idx], best_history


if __name__ == "__main__":
    coords = []
    try:
        with open("tsp_data.tsp") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    coords.append(list(map(float, parts[1:3])))
    except FileNotFoundError:
        print("tsp_data.tsp not found. Create a file with 'id x y' per line.")
    if not coords:
        np.random.seed(42)
        coords = np.random.rand(200, 2).tolist()
        print("Using random 200 points as demo data.")

    best_tour, best_len, hist = RMGA_EAX_only(
        coords, pop_size=100, generation=500, cycle_select_rate=0.3,
        twoopt_after=True, twoopt_maxiter=200, seed=23, verbose=True, benchmark=True
    )
    print("\nBest length:", best_len)
    print("Best tour (first 30 nodes):", best_tour[:30])
