
import os
import random
import numpy as np
import networkx as nx
from scipy.io import mmread
import json
from joblib import Parallel, delayed
from collections import OrderedDict

# ================= LOAD GRAPH =================
def load_mtx_graph(filepath):
    print(f"-> Đang đọc file: {filepath}...")
    sparse_matrix = mmread(filepath)
    sparse_matrix.data[:] = 1
    G = nx.from_scipy_sparse_array(sparse_matrix)

    if G.is_directed():
        G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    G = nx.convert_node_labels_to_integers(G)
    print(f"-> Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ================= TASKS =================
class MFEA_Tasks:
    def __init__(self, graph, max_cache_size=50000):
        self.G = graph
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)
        self.idx2node = dict(enumerate(self.nodes))

        self.adj = {u: np.fromiter(graph.neighbors(u), dtype=np.int32)
                    for u in self.nodes}
        self.deg = {u: len(self.adj[u]) for u in self.nodes}

        self.mdr_cache = OrderedDict()
        self.max_cache_size = max_cache_size

    # ---------- MDR ----------
    # def decode_mdr(self, genotype):
    #     labels = np.zeros(self.n, dtype=np.int8)
    #     labels[genotype >= 0.6] = 2
    #     labels[genotype >= 0.85] = 3
    #     # labels[(genotype >= 0.5) & (genotype < 0.6)] = 1
    #     # labels[(genotype >= 0.6) & (genotype < 0.85)] = 2
    #     # labels[genotype >= 0.85] = 3
    #     return dict(zip(self.nodes, labels))
    def decode_mdr(self, genotype):
        labels = np.zeros(self.n, dtype=np.int8)
        labels[(genotype >= 0.45) & (genotype < 0.6)] = 1
        labels[(genotype >= 0.6) & (genotype < 0.85)] = 2
        labels[genotype >= 0.85] = 3
        return dict(zip(self.nodes, labels))


    def is_feasible_global(self, sol):
        for u in self.nodes:
            val = sol[u]
            nbr_vals = [sol[v] for v in self.adj[u]]

            c1 = nbr_vals.count(1)
            c2 = nbr_vals.count(2)
            c3 = nbr_vals.count(3)

            if val == 0:
                if not (c3 >= 1 or c2 >= 2 or (c2 >= 1 and c1 >= 1)):
                    return False

            elif val == 1:
                if c2 + c3 < 1:
                    return False

        return True


    def repair_mdr_smart(self, sol):
        repaired = sol.copy()

        for u in sorted(self.nodes, key=lambda x: -self.deg[x]):
            if repaired[u] != 0:
                continue

            nbrs = self.adj[u]
            vals = [repaired[v] for v in nbrs]

            c1 = vals.count(1)
            c2 = vals.count(2)
            c3 = vals.count(3)

            if c3 >= 1 or c2 >= 2 or (c2 >= 1 and c1 >= 1):
                continue

            # ưu tiên nâng 0 → 1 trước
            if c2 >= 1:
                repaired[u] = 1
            else:
                # nâng hàng xóm degree cao nhất
                cand = max(nbrs, key=lambda x: self.deg[x])
                repaired[cand] = max(repaired[cand], 2)

        return repaired


    def optimize_mdr(self, sol):
        improved = sol.copy()

        for u in sorted(self.nodes, key=lambda x: self.deg[x]):
            orig = improved[u]
            if orig == 0:
                continue

            for new_val in [0, 1, 2]:
                if new_val >= orig:
                    continue
                improved[u] = new_val
                if self.is_feasible_global(improved):
                    break
                improved[u] = orig

        return improved


    def calculate_mdr_cost(self, genotype):
        decoded = self.decode_mdr(genotype)
        key = tuple(decoded[u] for u in self.nodes)

        if key in self.mdr_cache:
            self.mdr_cache.move_to_end(key)
            return self.mdr_cache[key]

        sol = self.optimize_mdr(self.repair_mdr_smart(decoded))
        cost = sum(int(v) for v in sol.values())

        self.mdr_cache[key] = cost
        if len(self.mdr_cache) > self.max_cache_size:
            self.mdr_cache.popitem(last=False)

        return cost

    # ---------- GCP ----------
    def calculate_gcp_cost(self, genotype):
        order = np.argsort(genotype)[::-1]
        colors = {}
        for idx in order:
            u = self.idx2node[idx]
            used = {colors[v] for v in self.adj[u] if v in colors}
            c = 0
            while c in used:
                c += 1
            colors[u] = c
        return len(set(colors.values()))


# ================= INDIVIDUAL =================
class Individual:
    __slots__ = ("genotype", "skill_factor", "factorial_costs", "scalar_fitness")

    def __init__(self, dim):
        self.genotype = np.random.rand(dim)
        self.skill_factor = None
        self.factorial_costs = [float("inf"), float("inf")]
        self.scalar_fitness = 0.0


# ================= MFEA-II =================
class MFEA:
    def __init__(self, tasks, pop_size=100, generations=100,
                 rmp=0.6, rmm=0.1, n_jobs=1):

        self.tasks = tasks
        self.dim = tasks.n
        self.pop_size = pop_size
        self.generations = generations
        self.rmm = rmm
        self.n_jobs = n_jobs

        self.rmp_dynamic = rmp
        self.success_transfer = 0
        self.fail_transfer = 0

        self.population = []

    # ---------- EVALUATION ----------
    def evaluate_individual(self, ind):
        c0 = self.tasks.calculate_mdr_cost(ind.genotype) \
            if ind.skill_factor in (None, 0) else float("inf")
        c1 = self.tasks.calculate_gcp_cost(ind.genotype) \
            if ind.skill_factor in (None, 1) else float("inf")
        return c0, c1

    def evaluate_population(self, pop):
        if self.n_jobs > 1:
            res = Parallel(n_jobs=self.n_jobs)(
                delayed(self.evaluate_individual)(ind) for ind in pop
            )
            for ind, (c0, c1) in zip(pop, res):
                ind.factorial_costs = [c0, c1]
        else:
            for ind in pop:
                ind.factorial_costs = list(self.evaluate_individual(ind))

    # ---------- FITNESS ----------
    def assign_scalar_fitness(self):
        for ind in self.population:
            ind.scalar_fitness = 0.0

        pop_mdr = sorted(
            [i for i in self.population if i.factorial_costs[0] < float("inf")],
            key=lambda x: x.factorial_costs[0]
        )
        pop_gcp = sorted(
            [i for i in self.population if i.factorial_costs[1] < float("inf")],
            key=lambda x: x.factorial_costs[1]
        )

        for r, ind in enumerate(pop_mdr):
            ind.scalar_fitness = max(ind.scalar_fitness, 1.0 / (r + 1))
        for r, ind in enumerate(pop_gcp):
            ind.scalar_fitness = max(ind.scalar_fitness, 1.0 / (r + 1))

    # ---------- MFEA-II TRANSFER ----------
    def can_transfer(self, p1, p2):
        if p1.skill_factor == p2.skill_factor:
            return True
        # ONLY GCP → MDR
        if {p1.skill_factor, p2.skill_factor} == {0, 1}:
            return random.random() < self.rmp_dynamic
        return False

    def mutate(self, parent):
        child = Individual.__new__(Individual)
        child.genotype = parent.genotype.copy()
        child.skill_factor = parent.skill_factor
        child.factorial_costs = [float("inf"), float("inf")]
        child.scalar_fitness = 0.0

        mask = np.random.rand(self.dim) < 0.1
        child.genotype[mask] += np.random.normal(0, 0.2, np.sum(mask))
        np.clip(child.genotype, 0, 1, out=child.genotype)
        return child

    def mate_or_mutate(self, p1, p2):
        c1, c2 = Individual(self.dim), Individual(self.dim)

        if self.can_transfer(p1, p2):
            mask = np.random.rand(self.dim) < 0.5
            c1.genotype = np.where(mask, p1.genotype, p2.genotype)
            c2.genotype = np.where(mask, p2.genotype, p1.genotype)
            c1.skill_factor = random.choice([p1.skill_factor, p2.skill_factor])
            c2.skill_factor = random.choice([p1.skill_factor, p2.skill_factor])
        else:
            c1, c2 = self.mutate(p1), self.mutate(p2)

        if random.random() < self.rmm:
            c1 = self.mutate(c1)
        if random.random() < self.rmm:
            c2 = self.mutate(c2)

        return c1, c2

    # ---------- RUN ----------
    def run(self):
        self.population = [Individual(self.dim) for _ in range(self.pop_size)]
        for ind in self.population:
            ind.skill_factor = random.choice([0, 1])

        self.evaluate_population(self.population)
        self.assign_scalar_fitness()

        hist_mdr, hist_gcp = [], []

        for gen in range(self.generations):
            old_best = min(ind.factorial_costs[0] for ind in self.population)

            random.shuffle(self.population)
            offspring = []
            for i in range(0, self.pop_size, 2):
                offspring += self.mate_or_mutate(
                    self.population[i], self.population[i + 1]
                )

            self.evaluate_population(offspring)
            new_best = min(ind.factorial_costs[0] for ind in offspring)

            if new_best < old_best:
                self.success_transfer += 1
            else:
                self.fail_transfer += 1

            # adaptive RMP
            if self.success_transfer + self.fail_transfer > 5:
                ratio = self.success_transfer / (self.success_transfer + self.fail_transfer)
                if ratio < 0.3:
                    self.rmp_dynamic = max(0.05, self.rmp_dynamic * 0.9)
                elif ratio > 0.6:
                    self.rmp_dynamic = min(0.9, self.rmp_dynamic * 1.1)
                self.success_transfer = 0
                self.fail_transfer = 0

            self.population.extend(offspring)
            self.assign_scalar_fitness()
            self.population.sort(key=lambda x: x.scalar_fitness, reverse=True)
            self.population = self.population[:self.pop_size]

            hist_mdr.append(min(i.factorial_costs[0] for i in self.population))
            hist_gcp.append(min(i.factorial_costs[1] for i in self.population))

            if gen % 10 == 0:
                print(f"Gen {gen}: MDR={hist_mdr[-1]} | RMP={self.rmp_dynamic:.2f}")

        best = min(
            [i for i in self.population if i.skill_factor == 0],
            key=lambda x: x.factorial_costs[0]
        )

        sol = self.tasks.optimize_mdr(
            self.tasks.repair_mdr_smart(
                self.tasks.decode_mdr(best.genotype)
            )
        )
        return sol, best.factorial_costs[0], hist_mdr, hist_gcp


# ================= RUN =================
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def run_experiment_on_file(mtx_file):
    G = load_mtx_graph(mtx_file)
    tasks = MFEA_Tasks(G)
    mfea = MFEA(tasks, pop_size=100, generations=300, rmp=0.6, n_jobs=4)
    sol, cost, hist_mdr, _ = mfea.run()
    print("FINAL MDR =", cost)


if __name__ == "__main__":
    mtx_files = ["../data/DROMD/lshp1009.mtx"]
    # mtx_files = ["../data/DROMD/ash85.mtx"]
    for file in mtx_files:
        run_experiment_on_file(file)