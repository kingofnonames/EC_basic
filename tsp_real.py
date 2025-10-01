import numpy as np

rng = np.random.default_rng(seed=7)

def distance(a, b):
    return np.linalg.norm(a - b)


def distance_matrix(cities, num_cities):
    dist_mat = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            d = distance(cities[i], cities[j])
            dist_mat[i, j] = dist_mat[j, i] = d
    return dist_mat

def two_opt(dist_mat, tour, max_iter=50):
    N = len(tour)
    new_tour = tour.copy()
    for _ in range(max_iter):
        i, j = sorted(rng.choice(N, 2, replace=False))
        if j - i < 2:
            continue
        a, b = new_tour[i], new_tour[i + 1]
        c, d = new_tour[j], new_tour[(j + 1) % N]
        delta = (dist_mat[a, c] + dist_mat[b, d]) - (dist_mat[a, b] + dist_mat[c, d])
        if delta < -1e-9:
            new_tour[i + 1:j + 1] = new_tour[i + 1:j + 1][::-1]
    return new_tour


class TSP_real:
    def __init__(self, cities, mutation="polynomial", selection="rank", crossover="sbx"):
        self.cities = cities
        self.num_cities = len(cities)
        self.population = None
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
        self.dist_mat = distance_matrix(self.cities, self.num_cities)
        self.best_tour = None
        self.best_cost = np.inf

    def init_population(self, pop_size):
        return np.array([rng.uniform(0, 1, size=self.num_cities) for _ in range(pop_size)])

    def computation_cost(self, tour):
        N = self.num_cities
        return sum(self.dist_mat[tour[i], tour[(i + 1) % N]] for i in range(N))

    def fitness(self, cost, alpha=1e-6):
        return 1.0 / (cost + alpha)
    
    def decode(self, tour_enc):
        return np.argsort(tour_enc)

    # def decode(self, tour_enc):
    #     idx_sort = np.argsort(tour_enc)
    #     tour_dec = np.zeros(shape=len(tour_enc), dtype=np.int64)
    #     for idx, val in enumerate(idx_sort):
    #         tour_dec[val] = idx
    #     return tour_dec
    
    def encode(self, tour_dec, low=0.0, high=1.0):
        N = len(tour_dec)
        step = (high - low) / N
        keys = np.empty(N, dtype=float)
        for rank, city in enumerate(tour_dec):
            keys[city] = low + rank * step + rng.uniform(0, step)
        return np.clip(keys, low, high)
    
    # def encode(self, tour_dec, low=0.0, high=1.0):
    #     N = len(tour_dec)
    #     step = (high - low) / N
    #     keys = np.empty(N, dtype=float)
    #     for idx, city in enumerate(tour_dec):
    #         keys[idx] = low + city * step + rng.uniform(0, step)
    #     return keys

    def evaluate_population(self, population):
        tours = [self.decode(ind) for ind in population]
        costs = np.array([self.computation_cost(t) for t in tours])
        fitnesses = 1.0 / (costs + 1e-6)
        return tours, costs, fitnesses
    
    def rank_selection(self, fitnesses):
        N = len(fitnesses)
        sorted_idx = np.argsort(fitnesses) 
        ranks = np.arange(1, N + 1)
        probs = ranks / ranks.sum()
        idx = rng.choice(sorted_idx, p=probs)
        return self.population[idx].copy()
    
    def tournament_selection(self, fitnesses, k=5):
        candidates = rng.choice(self.population, size=5)
        par = max(candidates, key=self.fitness)
        return par.copy()

    def sbx_crossover(self, par1, par2, eta=2.0, low=0.0, high=1.0):
        N = len(par1)
        u = rng.uniform(size=N)
        beta = np.where(
            u <= 0.5,
            (2 * u) ** (1 / (eta + 1)),
            (0.5 / (1 - u)) ** (1 / (eta + 1)),
        )
        child1 = 0.5 * ((1 + beta) * par1 + (1 - beta) * par2)
        child2 = 0.5 * ((1 - beta) * par1 + (1 + beta) * par2)
        return np.clip(child1, low, high), np.clip(child2, low, high)
    
    def polynomial_mutation(self, par, eta=2.0, low=0.0, high=1.0, p_m=0.2):
        child = par.copy()
        for i in range(len(par)):
            if rng.uniform() < p_m:
                u = rng.uniform()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                child[i] += delta * (high - low)
                child[i] = np.clip(child[i], low, high)
        return child
    
    def mutation_op(self, par):
        return getattr(self, f"{self.mutation}_mutation")(par)

    def crossover_op(self, par1, par2):
        return getattr(self, f"{self.crossover}_crossover")(par1, par2)
    
    def solve(self, generations=500, pop_size=100, pc=0.9, pm=0.2, elitism=2, use_2opt=True):
        self.population = self.init_population(pop_size)

        for gen in range(generations):
            tours, costs, fitnesses = self.evaluate_population(self.population)

            idx_best = np.argmin(costs)
            if costs[idx_best] < self.best_cost:
                self.best_cost = costs[idx_best]
                self.best_tour = tours[idx_best]

            elite_idx = np.argsort(fitnesses)[-elitism:]
            new_pop = [self.population[i].copy() for i in elite_idx]

            while len(new_pop) < pop_size:
                p1 = self.rank_selection(fitnesses)
                p2 = self.rank_selection(fitnesses)
                if rng.random() < pc:
                    c1, c2 = self.crossover_op(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                if rng.random() < pm:
                    c1 = self.mutation_op(c1)
                if rng.random() < pm:
                    c2 = self.mutation_op(c2)
                new_pop.extend([c1, c2])

            self.population = np.array(new_pop[:pop_size], dtype=np.float64)
            if use_2opt:
                for i in elite_idx:
                    improved_tour = two_opt(self.dist_mat, tours[i], max_iter=50)
                    self.population[i] = self.encode(improved_tour)

            if (gen + 1) % 10 == 0 or gen == generations - 1:
                print(f"Gen {gen+1}: Best cost = {self.best_cost:.2f}")

        return self.best_cost, self.best_tour
if __name__ == "__main__":
    cities = []
    with open(r"tsp_data.tsp") as f:
        for line in f.readlines():
            cities.append(np.array(list(map(int, line.split()))[1:]))
    tsp_solver = TSP_real(cities, selection="rank")
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