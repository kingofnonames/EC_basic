import numpy as np

rng = np.random.default_rng(seed=22)
def euclid_distance(a, b):
    return np.linalg.norm(a - b)

def distance_matrix(cities):
    N = len(cities)
    dist_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dist = euclid_distance(cities[i], cities[j])
            dist_mat[i, j] = dist_mat[j, i] = dist
    return dist_mat

def two_opt(tour, dist_mat):
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 2, len(tour)):
                a, b = tour[i-1], tour[i]
                c, d = tour[j-1], tour[j]
                delta = (dist_mat[a, c] + dist_mat[b, d]) - (dist_mat[a, b] + dist_mat[c, d])
                if delta < -1e-9:
                    tour[i:j] = tour[i:j][::-1]
                    improved = True
    return tour

class TSP_GA:
    def __init__(self, cities, mutation="inversion", crossover="ox", selection="rank"):
        self.cities = np.array(cities)
        self.num_cities = len(cities)
        self.dist_mat = distance_matrix(self.cities)
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.population = None
        self.best_tour = None
        self.best_cost = np.inf
        self.fitnesses = None

    def init_population(self, pop_size=100):
        return np.array([rng.permutation(self.num_cities) for _ in range(pop_size)], dtype=np.int64)

    def cost_computation(self, tour):
        N = self.num_cities
        return sum(self.dist_mat[tour[i], tour[(i+1)%N]] for i in range(N))

    def fitness(self, tour, alpha=1e-6):
        return 1.0 / (self.cost_computation(tour) + alpha)

    def rank_selection(self):
        sorted_idx = np.argsort(self.fitnesses)
        ranks = np.arange(start=1, stop=(len(self.population) + 1))
        probs = ranks / sum(ranks)
        idx = rng.choice(range(len(sorted_idx)), p=probs)
        return self.population[sorted_idx[idx]].copy()
    
    def tournament_selection(self, k=5):
        candidates = rng.choice(self.population, size=k, replace=False)
        par = max(candidates, key=self.fitness)
        return par.copy()

    def roulette_selection(self):
        sum_fitness = sum(self.fitnesses)
        prob_fitness = [self.fitnesses[pop] for pop in range(len(self.population))] / sum_fitness
        idx = rng.choice(range(len(self.population)), p=prob_fitness)
        return self.population[idx].copy()

    def selection_op(self):
        return getattr(self, f"{self.selection}_selection")()

    def ox(self, par1, par2):
        N = self.num_cities
        c1, c2 = sorted(rng.choice(N, size=2, replace=False))
        child = np.full(N, -1, dtype=np.int64)
        child[c1:c2] = par1[c1:c2]
        pos = c2 % N
        for city in par2:
            if city not in child:
                child[pos] = city
                pos = (pos + 1) % N
        return child

    def pmx(self, par1, par2):
        N = self.num_cities
        c1, c2 = sorted(rng.choice(range(N), size=2, replace=False))
        child = np.full(N, -1, dtype=np.int64)
        child[c1:c2] = par1[c1:c2]
        mapping = {}
        for i in range(c1, c2):
            mapping[par1[i]] = par2[i]
        for i in range(N):
            if child[i] != -1:
                continue
            gen = par2[i]
            while gen in child:
                gen = mapping.get(gen, gen)
            child[i] = gen
        return child

    def cx(self, par1, par2):
        map_par1 = {val: i for i, val in enumerate(par1)}
        N = self.num_cities
        child = np.full(shape=N, fill_value=-1, dtype=np.int64)
        visited = [False] * N
        idx = 0
        while not visited[par1[idx]]:
            visited[par1[idx]] = True
            child[idx] = par1[idx]
            idx = map_par1[par2[idx]]
        for i in range(N):
            if child[i] == -1:
                child[i] = par2[i]
        return child
    
    def inversion_mutation(self, par):
        N = self.num_cities
        c1, c2 = sorted(rng.choice(N, size=2, replace=False))
        par[c1:c2] = np.flip(par[c1:c2])
        return par

    def swap_mutation(self, par):
        N = self.num_cities
        c1, c2 = rng.choice(N, size=2, replace=False)
        par[c1], par[c2] = par[c2], par[c1]
        return par

    def insert_mutation(self, par):
        N = self.num_cities
        c1 = rng.integers(N)  
        city = par[c1]
        par = np.delete(par, c1)
        pos_new = rng.integers(len(par)+1)
        par = np.insert(par, pos_new, city)
        return par

    def displacement_mutation(self, par):
        N = self.num_cities
        c1, c2 = sorted(rng.choice(N, size=2, replace=False))
        segment = par[c1:c2].copy()
        par = np.delete(par, np.arange(c1, c2))
        pos_new = rng.choice(len(par)+1)
        par = np.insert(par, pos_new, segment)
        return par

    def scramble_mutation(self, par):
        N = self.num_cities
        c1, c2 = sorted(rng.choice(N, size=2, replace=False))
        segment = par[c1:c2].copy()
        rng.shuffle(segment)
        par[c1:c2] = segment
        return par

    def mutate(self, par):
        return getattr(self, f"{self.mutation}_mutation")(par)

    def crossover_op(self, par1, par2):
        return getattr(self, f"{self.crossover}")(par1, par2)
    
    def solve(self, generations=500, pop_size=100, pc=0.9, pm=0.2, elitism=2, use_2opt=True):
        self.population = self.init_population(pop_size)
        costs = np.array([self.cost_computation(t) for t in self.population])
        idx_best = np.argmin(costs)
        self.best_tour = self.population[idx_best].copy()
        self.best_cost = costs[idx_best]

        for gen in range(generations):
            self.fitnesses = [self.fitness(pop) for pop in self.population]
            new_pop = []
            elite_idx = np.argsort([self.fitness(t) for t in self.population])[-elitism:]
            for i in elite_idx:
                new_pop.append(self.population[i].copy())

            while len(new_pop) < pop_size:
                p1 = self.selection_op()
                p2 = self.selection_op()
                if rng.random() < pc:
                    c1 = self.crossover_op(p1, p2)
                    c2 = self.crossover_op(p2, p1)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                if rng.random() < pm:
                    c1 = self.mutate(c1)
                if rng.random() < pm:
                    c2 = self.mutate(c2)

                new_pop.extend([c1, c2])

            self.population = np.array(new_pop[:pop_size], dtype=np.int64)

            if use_2opt:
                top_idx = np.argsort([self.fitness(t) for t in self.population])[-5:]
                for i in top_idx:
                    self.population[i] = two_opt(self.population[i].copy(), self.dist_mat)

            costs = np.array([self.cost_computation(t) for t in self.population])
            idx_best = np.argmin(costs)
            if costs[idx_best] < self.best_cost:
                self.best_cost = costs[idx_best]
                self.best_tour = self.population[idx_best].copy()

            if ((gen + 1) % 10 == 0) or (gen == generations - 1):
                print(f"Gen {gen+1}: Best cost = {self.best_cost:.2f}")

        return self.best_cost, self.best_tour
    

if __name__ == "__main__":
    cities = []
    with open(r"tsp_data.tsp", encoding="utf-8") as f:
        for line in f.readlines():
            cities.append(np.array(list(map(int, line.split())))[1:])
    tsp_solver = TSP_GA(cities, mutation="swap", crossover="ox", selection="tournament")
    best_cost, best_tour = tsp_solver.solve(
        generations=500,
        pop_size=100,
        pc=0.9,
        pm=0.2,
        elitism=2,
        use_2opt=False
    )

    print("\n=== Kết quả cuối cùng ===")
    print("Tổng chiều dài tour:", round(best_cost, 2))
    print("Tour:", best_tour)
