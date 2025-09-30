import numpy as np

rng = np.random.default_rng(seed=7)

def distance(a, b):
    return np.linalg.norm(a - b)


def distance_matrix(cities, num_cities):
    dist_mat = np.zeros(shape=(num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_mat[i, j] = dist_mat[j, i] = distance(cities[i], cities[j])
    return dist_mat

def two_opt(dist_mat, tour):
    N = len(tour)
    improved = True
    new_tour = tour.copy()

    while improved:
        improved = False
        for i in range(N - 1):
            for j in range(i + 2, N):
                if j == N - 1 and i == 0:
                    continue
                a, b = new_tour[i], new_tour[i + 1]
                c, d = new_tour[j], new_tour[(j + 1) % N]
                delta = (dist_mat[a, c] + dist_mat[b, d]) - (dist_mat[a, b] + dist_mat[c, d])
                if delta < -1e-9:
                    new_tour[i + 1:j + 1] = new_tour[i + 1:j + 1][::-1]
                    improved = True
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

    def init_population(self, pop_size):
        return np.array([rng.uniform(0, 1, size=self.num_cities) for _ in range(pop_size)])


    def computation_cost(self, tour):
        N = self.num_cities
        return sum(self.dist_mat[tour[i], tour[(i + 1) % N]] for i in range(N))

    def fitness(self, tour, alpha=1e-6):
        return 1 / (self.computation_cost(tour) + alpha)
    
    def decode(self, tour_enc):
        sorted_val = sorted(enumerate(tour_enc), key=lambda x: x[1])
        tour_dec = [idx for idx, _ in sorted_val]
        return np.array(tour_dec, dtype=np.int64)
    
    def encode(self, tour_dec, low=0.0, high=1.0):
        N = len(tour_dec)
        step = (high - low) / N
        keys = np.empty(N, dtype=float)
        for rank, city in enumerate(tour_dec):
            keys[city] = low + rank * step + rng.uniform(0, step)
        return keys

    
    def rank_selection(self):
        N = len(self.population)
        population_sorted = sorted(self.population, key=lambda x: self.fitness(self.decode(x)), reverse=True)
        ranks = np.arange(N, 0, -1)
        rank_probs = ranks / ranks.sum()
        idx = np.random.choice(N, p=rank_probs)
        return population_sorted[idx]

    def sbx_crossover(self, par1, par2, eta=7.0, low=0.0, high=1.0):
        N = len(par1)
        u = rng.uniform(size=N)
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (0.5 / (1 - u)) ** (1 / (eta + 1)))
        child1 = 0.5 * ((1 + beta) * par1 + (1 - beta) * par2)
        child2 = 0.5 * ((1 - beta) * par1 + (1 + beta) * par2)
        return np.clip(child1, low, high), np.clip(child2, low, high)
    
    def polynomial_mutation(self, par, eta=7.0, low=0.0, high=1.0, p_m=0.1):
        child = par.copy()
        for i in range(len(par)):
            if rng.uniform() < p_m:
                u = rng.uniform()
                if u < 0.5:
                    delta = (2*u)**(1/(eta+1)) - 1
                else:
                    delta = 1 - (2*(1-u))**(1/(eta+1))
                child[i] = child[i] + delta * (high - low)
                child[i] = np.clip(child[i], low, high)
        return child
    
    def mutation_op(self, par):
        return getattr(self, f"{self.mutation}_mutation")(par)
    
    def crossover_op(self, par1, par2):
        return getattr(self, f"{self.crossover}_crossover")(par1, par2)
    
    def solve(
            self, 
            generations=500, 
            pop_size=100, 
            pc=0.9, 
            pm=0.2, 
            elitism=2, 
            use_2opt=True
    ):
        self.population = self.init_population(pop_size)
        costs = np.array([self.computation_cost(self.decode(t)) for t in self.population])
        idx_best = np.argmin(costs)
        self.best_tour = self.decode(self.population[idx_best].copy())
        self.best_cost = costs[idx_best]
        for gen in range(generations):
            new_pop = []
            elite_idx = np.argsort([self.fitness(self.decode(t)) for t in self.population])[-elitism:]
            for i in elite_idx:
                new_pop.append(self.population[i].copy())

            while len(new_pop) < pop_size:
                p1 = self.rank_selection()
                p2 = self.rank_selection()
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
                top_idx = np.argsort([self.fitness(self.decode(t)) for t in self.population])[-5:]
                for i in top_idx:
                    tour = self.decode(self.population[i].copy())
                    self.population[i] = self.encode(two_opt(self.dist_mat, tour))

            costs = np.array([self.computation_cost(self.decode(t)) for t in self.population])

            idx_best = np.argmin(costs)
            if costs[idx_best] < self.best_cost:
                self.best_cost = costs[idx_best]
                self.best_tour = self.decode(self.population[idx_best].copy())

            if ((gen + 1) % 10 == 0) or (gen == generations - 1):
                print(f"Gen {gen+1}: Best cost = {self.best_cost:.2f}")

        return self.best_cost, self.best_tour
if __name__ == "__main__":
    cities = []
    with open(r"tsp_data.tsp") as f:
        for line in f.readlines():
            cities.append(np.array(list(map(int, line.split()))[1:]))
    tsp_solver = TSP_real(cities)
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