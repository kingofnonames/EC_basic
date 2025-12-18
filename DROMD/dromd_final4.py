
# import os
# import random
# import numpy as np
# import networkx as nx
# from scipy.io import mmread
# import json
# from joblib import Parallel, delayed
# from collections import OrderedDict

# def load_mtx_graph(filepath):
#     print(f"-> Đang đọc file: {filepath}...")
#     sparse_matrix = mmread(filepath)
#     sparse_matrix.data[:] = 1
#     G = nx.from_scipy_sparse_array(sparse_matrix)

#     if G.is_directed():
#         G = G.to_undirected()
#     G.remove_edges_from(nx.selfloop_edges(G))

#     if not nx.is_connected(G):
#         largest_cc = max(nx.connected_components(G), key=len)
#         G = G.subgraph(largest_cc).copy()

#     G = nx.convert_node_labels_to_integers(G)
#     print(f"-> Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
#     return G

# class MFEA_Tasks:
#     def __init__(self, graph, max_cache_size=50000):
#         self.G = graph
#         self.nodes = list(graph.nodes())
#         self.n = len(self.nodes)
#         self.idx2node = dict(enumerate(self.nodes))
#         self.adj = {u: np.fromiter(graph.neighbors(u), dtype=np.int32)
#                     for u in self.nodes}
#         self.deg = {u: len(self.adj[u]) for u in self.nodes}
#         self.mdr_cache = OrderedDict()
#         self.max_cache_size = max_cache_size


#     ### Với bài toán có ít cạnh thì cần tăng 2, 1 giảm bớt 3
#     # def decode_mdr(self, genotype):
#     #     labels = np.zeros(self.n, dtype=)

#     #     labels[(genotype >= 0.57) & (genotype < 0.7)] = 1
#     #     labels[(genotype >= 0.7) & (genotype < 0.88)] = 2
#     #     labels[genotype >= 0.88] = 3

#     #     return dict(zip(self.nodes, labels))
#     ### Với bài toán có nhiều cạnh thì có thể tăng trọng số 3 lên, giảm trọng số 1 và 2
#     def decode_mdr(self, genotype):
#         labels = np.zeros(self.n, dtype=)

#         labels[(genotype >= 0.6) & (genotype < 0.7)] = 1
#         labels[(genotype >= 0.7) & (genotype < 0.88)] = 2
#         labels[genotype >= 0.88] = 3

#         return dict(zip(self.nodes, labels))



#     def is_feasible_global(self, sol):
#         for u in self.nodes:
#             val = sol[u]
#             nbr_vals = [sol[v] for v in self.adj[u]]
#             c2 = nbr_vals.count(2)
#             c3 = nbr_vals.count(3)

#             if val == 0:
#                 if not (c3 >= 1 or c2 >= 2):
#                     return False

#             elif val == 1:
#                 if c2 + c3 < 1:
#                     return False

#         return True


#     def repair_mdr_smart(self, sol):
#         repaired = sol.copy()

#         for u in sorted(self.nodes, key=lambda x: -self.deg[x]):
#             if repaired[u] != 0:
#                 continue

#             nbrs = self.adj[u]
#             vals = [repaired[v] for v in nbrs]

#             c1 = vals.count(1)
#             c2 = vals.count(2)
#             c3 = vals.count(3)

#             if c3 >= 1 or c2 >= 2:
#                 continue

#             if c2 >= 1:
#                 repaired[u] = 1
#             else:
#                 cand = max(nbrs, key=lambda x: self.deg[x])
#                 repaired[cand] = max(repaired[cand], 2)

#         return repaired


#     def optimize_mdr(self, sol):
#         improved = sol.copy()

#         for u in sorted(self.nodes, key=lambda x: self.deg[x]):
#             orig = improved[u]
#             if orig == 0:
#                 continue

#             for new_val in [0, 1, 2]:
#                 if new_val >= orig:
#                     continue
#                 improved[u] = new_val
#                 if self.is_feasible_global(improved):
#                     break
#                 improved[u] = orig

#         return improved


#     def calculate_mdr_cost(self, genotype):
#         decoded = self.decode_mdr(genotype)
#         key = tuple(decoded[u] for u in self.nodes)

#         if key in self.mdr_cache:
#             self.mdr_cache.move_to_end(key)
#             return self.mdr_cache[key]

#         sol = self.optimize_mdr(self.repair_mdr_smart(decoded))
#         cost = sum(int(v) for v in sol.values())

#         self.mdr_cache[key] = cost
#         if len(self.mdr_cache) > self.max_cache_size:
#             self.mdr_cache.popitem(last=False)

#         return cost

#     def calculate_gcp_cost(self, genotype):
#         order = np.argsort(genotype)[::-1]
#         colors = {}
#         for idx in order:
#             u = self.idx2node[idx]
#             used = {colors[v] for v in self.adj[u] if v in colors}
#             c = 0
#             while c in used:
#                 c += 1
#             colors[u] = c
#         return len(set(colors.values()))


# class Individual:
#     __slots__ = ("genotype", "skill_factor", "factorial_costs", "scalar_fitness")

#     def __init__(self, dim):
#         self.genotype = np.random.rand(dim)
#         self.skill_factor = None
#         self.factorial_costs = [float("inf"), float("inf")]
#         self.scalar_fitness = 0.0


# class MFEA:
#     def __init__(self, tasks, pop_size=100, generations=100,
#                  rmp=0.6, rmm=0.1, n_jobs=1):

#         self.tasks = tasks
#         self.dim = tasks.n
#         self.pop_size = pop_size
#         self.generations = generations
#         self.rmm = rmm
#         self.n_jobs = n_jobs

#         self.rmp_dynamic = rmp
#         self.success_transfer = 0
#         self.fail_transfer = 0

#         self.population = []

#     def evaluate_individual(self, ind):
#         c0 = self.tasks.calculate_mdr_cost(ind.genotype) \
#             if ind.skill_factor in (None, 0) else float("inf")
#         c1 = self.tasks.calculate_gcp_cost(ind.genotype) \
#             if ind.skill_factor in (None, 1) else float("inf")
#         return c0, c1

#     def evaluate_population(self, pop):
#         if self.n_jobs > 1:
#             res = Parallel(n_jobs=self.n_jobs)(
#                 delayed(self.evaluate_individual)(ind) for ind in pop
#             )
#             for ind, (c0, c1) in zip(pop, res):
#                 ind.factorial_costs = [c0, c1]
#         else:
#             for ind in pop:
#                 ind.factorial_costs = list(self.evaluate_individual(ind))

#     def assign_scalar_fitness(self):
#         for ind in self.population:
#             ind.scalar_fitness = 0.0

#         pop_mdr = sorted(
#             [i for i in self.population if i.factorial_costs[0] < float("inf")],
#             key=lambda x: x.factorial_costs[0]
#         )
#         pop_gcp = sorted(
#             [i for i in self.population if i.factorial_costs[1] < float("inf")],
#             key=lambda x: x.factorial_costs[1]
#         )

#         for r, ind in enumerate(pop_mdr):
#             ind.scalar_fitness = max(ind.scalar_fitness, 1.0 / (r + 1))
#         for r, ind in enumerate(pop_gcp):
#             ind.scalar_fitness = max(ind.scalar_fitness, 1.0 / (r + 1))

#     def can_transfer(self, p1, p2):
#         if p1.skill_factor == p2.skill_factor:
#             return True
#         if {p1.skill_factor, p2.skill_factor} == {0, 1}:
#             return random.random() < self.rmp_dynamic
#         return False

#     def mutate(self, parent):
#         child = Individual.__new__(Individual)
#         child.genotype = parent.genotype.copy()
#         child.skill_factor = parent.skill_factor
#         child.factorial_costs = [float("inf"), float("inf")]
#         child.scalar_fitness = 0.0

#         mask = np.random.rand(self.dim) < 0.1
#         child.genotype[mask] += np.random.normal(0, 0.2, np.sum(mask))
#         np.clip(child.genotype, 0, 1, out=child.genotype)
#         return child

#     def mate_or_mutate(self, p1, p2):
#         c1, c2 = Individual(self.dim), Individual(self.dim)

#         if self.can_transfer(p1, p2):
#             mask = np.random.rand(self.dim) < 0.5
#             c1.genotype = np.where(mask, p1.genotype, p2.genotype)
#             c2.genotype = np.where(mask, p2.genotype, p1.genotype)
#             c1.skill_factor = random.choice([p1.skill_factor, p2.skill_factor])
#             c2.skill_factor = random.choice([p1.skill_factor, p2.skill_factor])
#         else:
#             c1, c2 = self.mutate(p1), self.mutate(p2)

#         if random.random() < self.rmm:
#             c1 = self.mutate(c1)
#         if random.random() < self.rmm:
#             c2 = self.mutate(c2)

#         return c1, c2

#     def run(self):
#         self.population = [Individual(self.dim) for _ in range(self.pop_size)]
#         for ind in self.population:
#             ind.skill_factor = random.choice([0, 1])

#         self.evaluate_population(self.population)
#         self.assign_scalar_fitness()

#         hist_mdr, hist_gcp = [], []

#         for gen in range(self.generations):
#             old_best = min(ind.factorial_costs[0] for ind in self.population)

#             random.shuffle(self.population)
#             offspring = []
#             for i in range(0, self.pop_size, 2):
#                 offspring += self.mate_or_mutate(
#                     self.population[i], self.population[i + 1]
#                 )

#             self.evaluate_population(offspring)
#             new_best = min(ind.factorial_costs[0] for ind in offspring)

#             if new_best < old_best:
#                 self.success_transfer += 1
#             else:
#                 self.fail_transfer += 1

#             if self.success_transfer + self.fail_transfer > 5:
#                 ratio = self.success_transfer / (self.success_transfer + self.fail_transfer)
#                 if ratio < 0.3:
#                     self.rmp_dynamic = max(0.05, self.rmp_dynamic * 0.9)
#                 elif ratio > 0.6:
#                     self.rmp_dynamic = min(0.9, self.rmp_dynamic * 1.1)
#                 self.success_transfer = 0
#                 self.fail_transfer = 0

#             self.population.extend(offspring)
#             self.assign_scalar_fitness()
#             self.population.sort(key=lambda x: x.scalar_fitness, reverse=True)
#             self.population = self.population[:self.pop_size]

#             hist_mdr.append(min(i.factorial_costs[0] for i in self.population))
#             hist_gcp.append(min(i.factorial_costs[1] for i in self.population))

#             if gen % 10 == 0:
#                 print(f"Gen {gen}: MDR={hist_mdr[-1]} | GCP={hist_gcp[-1]}|RMP={self.rmp_dynamic:.2f}")

#         best = min(
#             [i for i in self.population if i.skill_factor == 0],
#             key=lambda x: x.factorial_costs[0]
#         )

#         sol = self.tasks.optimize_mdr(
#             self.tasks.repair_mdr_smart(
#                 self.tasks.decode_mdr(best.genotype)
#             )
#         )
#         return sol, best.factorial_costs[0], hist_mdr, hist_gcp


# def set_global_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)

# def run_experiment_on_file(
#     mtx_file,
#     seeds=[223, 42, 77],
#     pop_size=150,
#     generations=200,
#     rmp=0.6,
#     n_jobs=4,
#     output_folder="./results_mfea2"
# ):
#     file_name = os.path.basename(mtx_file)
#     G = load_mtx_graph(mtx_file)
#     if G is None:
#         print("Không load được graph")
#         return None

#     print(f"\n=== BẮT ĐẦU CHẠY FILE: {file_name} ===")
#     print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

#     tasks = MFEA_Tasks(G)

#     results = []
#     all_hist_mdr = []
#     all_hist_gcp = []

#     best_mdr = float("inf")
#     best_gcp = float("inf")

#     for seed in seeds:
#         print(f"\n>>> Chạy với seed = {seed}")
#         set_global_seed(seed)

#         mfea = MFEA(
#             tasks,
#             pop_size=pop_size,
#             generations=generations,
#             rmp=rmp,
#             n_jobs=n_jobs
#         )

#         final_sol, final_weight, hist_mdr, hist_gcp = mfea.run()

#         hist_mdr = list(map(float, hist_mdr))
#         hist_gcp = list(map(float, hist_gcp))

#         results.append({
#             "seed": seed,
#             "final_mdr": float(final_weight),
#             "history_mdr": hist_mdr,
#             "history_gcp": hist_gcp
#         })

#         all_hist_mdr.append(hist_mdr)
#         all_hist_gcp.append(hist_gcp)

#         best_mdr = min(best_mdr, final_weight)
#         best_gcp = min(best_gcp, min(hist_gcp))

#         print(f"Seed {seed} → FINAL MDR = {final_weight}")

#     arr_mdr = np.array(all_hist_mdr)
#     arr_gcp = np.array(all_hist_gcp)

#     summary = {
#         "file": file_name,
#         "num_nodes": G.number_of_nodes(),
#         "num_edges": G.number_of_edges(),
#         "pop_size": pop_size,
#         "generations": generations,
#         "rmp_init": rmp,
#         "seeds": seeds,
#         "best_mdr": float(best_mdr),
#         "best_gcp": float(best_gcp),
#         "mean_mdr": np.mean(arr_mdr, axis=0).tolist(),
#         "var_mdr": np.var(arr_mdr, axis=0).tolist(),
#         "mean_gcp": np.mean(arr_gcp, axis=0).tolist(),
#         "var_gcp": np.var(arr_gcp, axis=0).tolist(),
#         "runs": results
#     }

#     os.makedirs(output_folder, exist_ok=True)
#     output_file = os.path.join(output_folder, f"{file_name}.json")

#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2, ensure_ascii=False)

#     print("\n=== TỔNG KẾT FILE", file_name, "===")
#     print(f"BEST MDR = {best_mdr}")
#     print(f"BEST GCP = {best_gcp}")
#     print(f"→ Đã lưu kết quả vào {output_file}\n")

#     return output_file



# if __name__ == "__main__":
#     # mtx_files = ["../data/DROMD/lshp1009.mtx"]
#     mtx_files = ["../data/DROMD/lshp1009.mtx"]
#     for file in mtx_files:
#         run_experiment_on_file(file)





import os
import random
import json
import numpy as np
import networkx as nx
from scipy.io import mmread
from collections import OrderedDict
from joblib import Parallel, delayed
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
class MFEA_Tasks:
    def __init__(self, graph, max_cache_size=50000):
        self.G = graph
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)

        self.idx2node = dict(enumerate(self.nodes))
        self.adj = {u: list(graph.neighbors(u)) for u in self.nodes}
        self.deg = {u: len(self.adj[u]) for u in self.nodes}

        degs = np.array(list(self.deg.values()))
        self.dmin, self.dmax = degs.min(), degs.max()
        self.norm_deg = {
            u: (self.deg[u] - self.dmin) / (self.dmax - self.dmin + 1e-9)
            for u in self.nodes
        }

        self.density = 2 * graph.number_of_edges() / (self.n * self.n)

        self.mdr_cache = OrderedDict()
        self.max_cache_size = max_cache_size
    def decode_mdr(self, genotype):
        labels = np.zeros(self.n, dtype=np.int32)

        if self.density < 0.01:
            t1, t2, t3 = 0.65, 0.78, 0.92
        elif self.density > 0.05:
            t1, t2, t3 = 0.55, 0.70, 0.85
        else:
            t1, t2, t3 = 0.60, 0.75, 0.88

        for i, u in enumerate(self.nodes):
            g = genotype[i]
            w = self.norm_deg[u]

            a1 = t1 - 0.10 * w
            a2 = t2 - 0.10 * w
            a3 = t3 - 0.05 * w

            if g >= a3:
                labels[i] = 3
            elif g >= a2:
                labels[i] = 2
            elif g >= a1:
                labels[i] = 1
            else:
                labels[i] = 0

        return dict(zip(self.nodes, labels))
    
    def init_genotype(self, mode="mixed"):
        g = np.random.rand(self.n)

        if mode == "random":
            return g

        if mode == "degree":
            for i, u in enumerate(self.nodes):
                g[i] += 0.35 * self.norm_deg[u]
            return np.clip(g, 0, 1)

        if mode == "noisy":
            g += np.random.normal(0, 0.5, self.n)
            return np.clip(g, 0, 1)

        r = random.random()
        if r < 0.4:
            return self.init_genotype("random")
        elif r < 0.8:
            return self.init_genotype("degree")
        else:
            return self.init_genotype("noisy")
    def is_feasible_global(self, sol):
        for u in self.nodes:
            val = sol[u]
            nbr_vals = [sol[v] for v in self.adj[u]]

            c2 = nbr_vals.count(2)
            c3 = nbr_vals.count(3)

            if val == 0:
                if not (c3 >= 1 or c2 >= 2):
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

            if vals.count(2) >= 1:
                repaired[u] = 1
            else:
                cand = max(self.adj[u], key=lambda x: self.deg[x])
                repaired[cand] = max(repaired[cand], 2)

        return repaired

    def optimize_mdr(self, sol):
        improved = sol.copy()

        for u in sorted(self.nodes, key=lambda x: self.deg[x]):
            orig = improved[u]
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
        cost = sum(sol.values())

        self.mdr_cache[key] = cost
        if len(self.mdr_cache) > self.max_cache_size:
            self.mdr_cache.popitem(last=False)

        return cost

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
class Individual:
    __slots__ = ("genotype", "skill_factor", "factorial_costs", "scalar_fitness")

    def __init__(self, dim, tasks, skill):
        self.skill_factor = skill
        if skill == 0:
            self.genotype = tasks.init_genotype("mixed")
        else:
            self.genotype = np.random.rand(dim)

        self.factorial_costs = [float("inf"), float("inf")]
        self.scalar_fitness = 0.0
class MFEA:
    def __init__(self, tasks, pop_size=150, generations=200,
                 rmp=0.6, rmm=0.1, n_jobs=1):

        self.tasks = tasks
        self.dim = tasks.n
        self.pop_size = pop_size
        self.generations = generations
        self.rmp_dynamic = rmp
        self.rmm = rmm
        self.n_jobs = n_jobs

        self.success_transfer = 0
        self.fail_transfer = 0
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
    def assign_scalar_fitness(self):
        for ind in self.population:
            ind.scalar_fitness = 0.0

        for k in [0, 1]:
            ranked = sorted(
                [i for i in self.population if i.factorial_costs[k] < float("inf")],
                key=lambda x: x.factorial_costs[k]
            )
            for r, ind in enumerate(ranked):
                ind.scalar_fitness = max(ind.scalar_fitness, 1 / (r + 1))
    def can_transfer(self, p1, p2):
        if p1.skill_factor == p2.skill_factor:
            return True
        return random.random() < self.rmp_dynamic
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
        if self.can_transfer(p1, p2):
            mask = np.random.rand(self.dim) < 0.5
            c1 = Individual(self.dim, self.tasks, random.choice([p1.skill_factor, p2.skill_factor]))
            c2 = Individual(self.dim, self.tasks, random.choice([p1.skill_factor, p2.skill_factor]))
            c1.genotype = np.where(mask, p1.genotype, p2.genotype)
            c2.genotype = np.where(mask, p2.genotype, p1.genotype)
        else:
            c1, c2 = self.mutate(p1), self.mutate(p2)

        if random.random() < self.rmm:
            c1 = self.mutate(c1)
        if random.random() < self.rmm:
            c2 = self.mutate(c2)

        return c1, c2
    def run(self):
        self.population = [
            Individual(self.dim, self.tasks, random.choice([0, 1]))
            for _ in range(self.pop_size)
        ]

        self.evaluate_population(self.population)
        self.assign_scalar_fitness()

        hist_mdr, hist_gcp = [], []

        for gen in range(self.generations):
            random.shuffle(self.population)
            offspring = []

            for i in range(0, self.pop_size, 2):
                offspring += self.mate_or_mutate(
                    self.population[i], self.population[i + 1]
                )

            self.evaluate_population(offspring)

            self.population.extend(offspring)
            self.assign_scalar_fitness()
            self.population.sort(key=lambda x: x.scalar_fitness, reverse=True)
            self.population = self.population[:self.pop_size]

            hist_mdr.append(min(i.factorial_costs[0] for i in self.population))
            hist_gcp.append(min(i.factorial_costs[1] for i in self.population))

            if gen % 10 == 0:
                print(f"Gen {gen}: MDR={hist_mdr[-1]} | GCP={hist_gcp[-1]} | RMP={self.rmp_dynamic:.2f}")

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
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    mtx_file = "../data/DROMD/dwt__419.mtx"
    G = load_mtx_graph(mtx_file)

    tasks = MFEA_Tasks(G)
    mfea = MFEA(tasks, pop_size=150, generations=200, n_jobs=4)

    sol, best_cost, hist_mdr, hist_gcp = mfea.run()

    print("\nFINAL MDR COST:", best_cost)