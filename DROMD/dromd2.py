import os
import random
import numpy as np
import networkx as nx
from scipy.io import mmread
import json
from utils.load_file import list_files

def load_mtx_graph(filepath):
    print(f"-> Đang đọc file: {filepath}...")

    try:
        sparse_matrix = mmread(filepath)
        G = nx.from_scipy_sparse_array(sparse_matrix)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None

    if G.is_directed():
        G = G.to_undirected()

    G.remove_edges_from(nx.selfloop_edges(G))

    if not nx.is_connected(G):
        print(" (Cảnh báo: Đồ thị gốc không liên thông. Đang trích xuất thành phần lớn nhất...)")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    G = nx.convert_node_labels_to_integers(G)

    print(f"-> Đồ thị đã tải: {len(G.nodes())} đỉnh, {len(G.edges())} cạnh.")
    return G

class MFEA_Tasks:
    def __init__(self, graph):
        self.G = graph
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)
        self.idx2node = {i: node for i, node in enumerate(self.nodes)}

    def decode_mdr(self, genotype):
        solution = {}
        for i, val in enumerate(genotype):
            node = self.idx2node[i]

            if val < 0.6:
                label = 0
            elif val < 0.85:
                label = 2
            else:
                label = 3

            solution[node] = label

        return solution

    def is_feasible_global(self, solution):
        for u in self.nodes:
            val = solution[u]

            if val == 0:
                neighbors = list(self.G.neighbors(u))
                n3 = sum(1 for v in neighbors if solution[v] == 3)
                n2 = sum(1 for v in neighbors if solution[v] == 2)
                if not (n3 >= 1 or n2 >= 2):
                    return False

            elif val == 1:
                neighbors = list(self.G.neighbors(u))
                nge2 = sum(1 for v in neighbors if solution[v] >= 2)
                if nge2 < 1:
                    return False

        return True

    def repair_mdr_smart(self, solution):
        repaired_sol = solution.copy()

        for u in self.nodes:
            if repaired_sol[u] == 0:
                neighbors = list(self.G.neighbors(u))
                n3 = sum(1 for v in neighbors if repaired_sol[v] == 3)
                n2 = sum(1 for v in neighbors if repaired_sol[v] == 2)

                if not (n3 >= 1 or n2 >= 2):
                    neighbors_2 = [v for v in neighbors if repaired_sol[v] == 2]

                    if neighbors_2:
                        best_nbr = max(neighbors_2, key=lambda x: self.G.degree(x))
                        repaired_sol[best_nbr] = 3
                    else:
                        repaired_sol[u] = 2

        return repaired_sol

    def optimize_mdr(self, solution):
        sorted_nodes = sorted(self.nodes, key=lambda x: self.G.degree(x))
        improved_sol = solution.copy()

        for u in sorted_nodes:
            original_val = improved_sol[u]
            if original_val == 0:
                continue

            improved_sol[u] = 0
            if self.is_feasible_global(improved_sol):
                continue

            if original_val == 3:
                improved_sol[u] = 2
                if self.is_feasible_global(improved_sol):
                    continue

            improved_sol[u] = original_val

        return improved_sol

    def calculate_mdr_cost(self, genotype):
        sol = self.decode_mdr(genotype)
        sol = self.repair_mdr_smart(sol)
        sol = self.optimize_mdr(sol)
        return sum(sol.values())

    def decode_gcp(self, genotype):
        indices = np.argsort(genotype)[::-1]
        sorted_nodes = [self.idx2node[i] for i in indices]
        colors = {}

        for node in sorted_nodes:
            neighbor_colors = {colors[nbr] for nbr in self.G.neighbors(node) if nbr in colors}
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[node] = c

        return colors

    def calculate_gcp_cost(self, genotype):
        colors = self.decode_gcp(genotype)
        return len(set(colors.values()))


class Individual:
    def __init__(self, dim):
        self.genotype = np.random.rand(dim)
        self.skill_factor = None
        self.factorial_costs = [float('inf'), float('inf')]
        self.scalar_fitness = float('inf')

class MFEA:
    def __init__(self, tasks, pop_size=100, generations=100, rmp=0.6, rmm=0.1):
        self.tasks = tasks
        self.dim = tasks.n
        self.pop_size = pop_size
        self.generations = generations
        self.rmp = rmp
        self.rmm = rmm
        self.population = []

    def run(self):
        for _ in range(self.pop_size):
            ind = Individual(self.dim)
            ind.factorial_costs[0] = self.tasks.calculate_mdr_cost(ind.genotype)
            ind.factorial_costs[1] = self.tasks.calculate_gcp_cost(ind.genotype)
            ind.skill_factor = random.choice([0, 1])
            self.population.append(ind)

        self.assign_scalar_fitness()

        history_mdr, history_gcp = [], []

        for gen in range(self.generations):
            offspring_pop = []
            random.shuffle(self.population)

            for i in range(0, self.pop_size, 2):
                p1 = self.population[i]
                p2 = self.population[i + 1]

                c1 = Individual(self.dim)
                c2 = Individual(self.dim)

                can_mate = (p1.skill_factor == p2.skill_factor) or \
                           (random.random() < self.rmp)

                if can_mate:
                    mask = np.random.rand(self.dim) < 0.5
                    c1.genotype = np.where(mask, p1.genotype, p2.genotype)
                    c2.genotype = np.where(mask, p2.genotype, p1.genotype)

                    c1.skill_factor = random.choice([p1.skill_factor, p2.skill_factor])
                    c2.skill_factor = random.choice([p1.skill_factor, p2.skill_factor])
                else:
                    c1 = self.mutate(p1)
                    c2 = self.mutate(p2)

                if random.random() < self.rmm:
                    c1 = self.mutate(c1)
                if random.random() < self.rmm:
                    c2 = self.mutate(c2)

                offspring_pop.extend([c1, c2])

            for child in offspring_pop:
                if child.skill_factor == 0:
                    child.factorial_costs[0] = self.tasks.calculate_mdr_cost(child.genotype)
                else:
                    child.factorial_costs[1] = self.tasks.calculate_gcp_cost(child.genotype)

            self.population += offspring_pop
            self.assign_scalar_fitness()
            self.population.sort(key=lambda x: x.scalar_fitness, reverse=True)
            self.population = self.population[:self.pop_size]
            best_mdr = min(ind.factorial_costs[0] for ind in self.population)
            best_gcp = min(ind.factorial_costs[1] for ind in self.population)
            history_mdr.append(best_mdr)
            history_gcp.append(best_gcp)

            if gen % 10 == 0:
                print(f"Gen {gen}: Best MDR Weight = {best_mdr}, Best Colors = {best_gcp}")

        best_mdr_ind = min(
            [ind for ind in self.population if ind.skill_factor == 0],
            key=lambda x: x.factorial_costs[0]
        )

        raw_sol = self.tasks.decode_mdr(best_mdr_ind.genotype)
        repaired_sol = self.tasks.repair_mdr_smart(raw_sol)
        final_solution = self.tasks.optimize_mdr(repaired_sol)

        return final_solution, best_mdr_ind.factorial_costs[0], history_mdr, history_gcp

    def mutate(self, p):
        child = Individual(self.dim)
        child.skill_factor = p.skill_factor
        child.genotype = p.genotype.copy()

        mask = np.random.rand(self.dim) < 0.1
        noise = np.random.normal(0, 0.2, self.dim)

        child.genotype[mask] += noise[mask]
        child.genotype = np.clip(child.genotype, 0, 1)
        return child

    def assign_scalar_fitness(self):
        pop_mdr = [ind for ind in self.population if ind.factorial_costs[0] != float('inf')]
        pop_gcp = [ind for ind in self.population if ind.factorial_costs[1] != float('inf')]

        pop_mdr.sort(key=lambda x: x.factorial_costs[0])
        pop_gcp.sort(key=lambda x: x.factorial_costs[1])

        for rank, ind in enumerate(pop_mdr):
            ind.scalar_fitness = 1.0 / (rank + 1)

        for rank, ind in enumerate(pop_gcp):
            ind.scalar_fitness = 1.0 / (rank + 1)

        for ind in self.population:
            if ind.scalar_fitness == float('inf'):
                ind.scalar_fitness = 0

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def run_experiment_on_file(
    mtx_file,
    seeds=[223, 42, 77],
    pop_size=100,
    generations=100,
    rmp=0.6,
    output_folder="./results"
):

    file_name = os.path.basename(mtx_file)

    if not os.path.exists(mtx_file):
        print(f"Không tìm thấy file {file_name}.")
        return None

    G = load_mtx_graph(mtx_file)
    if G is None:
        print("Lỗi load graph.")
        return None

    print(f"\n=== BẮT ĐẦU CHẠY FILE: {file_name} ===")

    tasks = MFEA_Tasks(G)

    results = []
    all_hist_mdr = []
    all_hist_gcp = []
    best_gcp = float('inf')

    for seed in seeds:
        print("\n" + "=" * 60)
        print(f">>> Chạy với seed = {seed}")
        print("=" * 60)

        set_global_seed(seed)

        mfea = MFEA(tasks, pop_size=pop_size, generations=generations, rmp=rmp)
        final_sol, final_weight, hist_mdr, hist_gcp = mfea.run()

        results.append({
            "seed": seed,
            "weight": float(final_weight),
            "history_mdr": list(map(float, hist_mdr)),
            "history_gcp": list(map(float, hist_gcp)),
        })

        best_gcp = min(best_gcp, min(hist_gcp))
        all_hist_mdr.append(hist_mdr)
        all_hist_gcp.append(hist_gcp)

        print(f"Seed {seed} → MDR Weight = {final_weight}")

    best_mdr_result = min(results, key=lambda x: x["weight"])
    best_mdr = best_mdr_result["weight"]

    print("\n=== TỔNG KẾT FILE", file_name, "===")
    print(f"BEST MDR = {best_mdr}")
    print(f"BEST GCP = {best_gcp}")

    arr_mdr = np.array(all_hist_mdr)
    arr_gcp = np.array(all_hist_gcp)

    mean_mdr = np.mean(arr_mdr, axis=0).tolist()
    var_mdr  = np.var(arr_mdr, axis=0).tolist()

    mean_gcp = np.mean(arr_gcp, axis=0).tolist()
    var_gcp  = np.var(arr_gcp, axis=0).tolist()

    output = {
        "seeds_results": results,
        "mean_mdr": mean_mdr,
        "var_mdr": var_mdr,
        "best_mdr": best_mdr,
        "best_gcp": best_gcp,
        "mean_gcp": mean_gcp,
        "var_gcp": var_gcp
    }
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{file_name}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Đã lưu kết quả vào {output_file}\n")

    return output_file

if __name__ == "__main__":
    # data_path = '../data/DROMD/'
    # mtx_files = list_files(data_path)
    mtx_files = ['../data/DROMD/662_bus.mtx']
    for file in mtx_files:
        run_experiment_on_file(file)
    # mtx_file = "../data/DROMD/can___61.mtx"
    # file_name = os.path.basename(mtx_file)

    # if not os.path.exists(mtx_file):
    #     print(f"Không tìm thấy file {file_name}.")
    #     exit()

    # G = load_mtx_graph(mtx_file)
    # if G is None:
    #     exit()
    # best_gcp = float('inf')
    # tasks = MFEA_Tasks(G)
    # seeds = [0, 42, 77, 123, 999]
    # results = []
    # all_hist_mdr = []
    # all_hist_gcp = []

    # for seed in seeds:
    #     print("\n" + "=" * 60)
    #     print(f">>> Bắt đầu chạy với seed = {seed}")
    #     print("=" * 60)
    #     set_global_seed(seed)

    #     mfea = MFEA(tasks, pop_size=100, generations=100, rmp=0.6)
    #     final_sol, final_weight, hist_mdr, hist_gcp = mfea.run()

    #     results.append({
    #         "seed": seed,
    #         "weight": float(final_weight),
    #         "history_mdr": list(map(float, hist_mdr)),
    #         "history_gcp": list(map(float, hist_gcp))
    #     })
    #     best_gcp = min(best_gcp, min(hist_gcp))
    #     all_hist_mdr.append(hist_mdr)
    #     all_hist_gcp.append(hist_gcp)

    #     print(f"Seed {seed} → MDR Weight = {final_weight}")



    # best_mdr = min(results, key=lambda x: x["weight"])['weight']
    # print(f"BEST MDR: {best_mdr}")
    # print(f"BEST GCP: {best_gcp}")
    # arr_mdr = np.array(all_hist_mdr)
    # arr_gcp = np.array(all_hist_gcp)

    # mean_mdr = np.mean(arr_mdr, axis=0).tolist()
    # var_mdr = np.var(arr_mdr, axis=0).tolist()

    # mean_gcp = np.mean(arr_gcp, axis=0).tolist()
    # var_gcp = np.var(arr_gcp, axis=0).tolist()
    # output = {
    #     "seeds_results": results,
    #     "mean_mdr": mean_mdr,
    #     "var_mdr": var_mdr,
    #     "best_mdr": best_mdr,
    #     "best_gcp": best_gcp,
    #     "mean_gcp": mean_gcp,
    #     "var_gcp": var_gcp
    # }
    # output_file = f'./results/{file_name}.json'
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(output, f, indent=2, ensure_ascii=False)

    # print(f"\nĐã lưu toàn bộ kết quả vào {output_file}")
    