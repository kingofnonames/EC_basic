import numpy as np
from typing import List

rng = np.random.default_rng(seed=22)

class GA:
    def __init__(self, n_items: int, capacity: int, weights: List[int], values: List[int]):
        self.n_items = n_items
        self.capacity = capacity
        self.weights = weights
        self.values = values
        self.population = None
        self.fitnesses = None
        self.best_fit = 0.0
        self.best_solution = None

    def compute_fitness(self, gen):
        w, fit = 0, 0
        for i in range(self.n_items):
            if gen[i] >= 0.5:
                w += self.weights[i]
                fit += self.values[i]
        if w > self.capacity:
            return self.capacity - w
        return fit

    def init_population(self, pop_size):
        dim = self.n_items
        return [rng.uniform(0, 1, size=dim) for _ in range(pop_size)]

    def rank_selection(self):
        rank = np.arange(start=1, stop=len(self.population) + 1)
        idx_sort = np.argsort(self.fitnesses)
        probs = rank / np.sum(rank)
        idx_par1 = rng.choice(range(len(idx_sort)), p=probs)
        idx_par2 = rng.choice(range(len(idx_sort)), p=probs)
        while idx_par2 == idx_par1:
            idx_par2 = rng.choice(range(len(idx_sort)), p=probs)
        return self.population[idx_sort[idx_par1]], self.population[idx_sort[idx_par2]]

    def crossover(self, par1, par2, eta_x=7.0):
        dim = len(par1)
        u = rng.uniform(0, 1, size=dim)
        beta = np.where(u < 0.5, (2 * u) ** (1.0 / (eta_x + 1)), (0.5 / (1 - u)) ** (1.0 / (eta_x + 1)))
        child1 = 0.5 * ((1 + beta) * par1 + (1 - beta) * par2)
        child2 = 0.5 * ((1 - beta) * par1 + (1 + beta) * par2)
        return np.clip(child1, 0, 1), np.clip(child2, 0, 1)

    def mutation(self, par, sigma=0.5):
        child = par.copy()
        dim = len(child)
        mutate_rate = 1.0 / dim
        for i in range(dim):
            if rng.random() < mutate_rate:
                child[i] += rng.normal(0, sigma)
        return np.clip(child, 0, 1)

    def decode(self, gen):
        dec = (gen >= 0.5).astype(int)
        return dec.tolist()

    def ga(self, pop_size=100, generations=500, elitism=3, rmx=0.9, rmm=0.3, eta_x=7.0):
        self.population = self.init_population(pop_size)
        for g in range(generations):
            self.fitnesses = np.array([self.compute_fitness(pop) for pop in self.population])
            idx_best = np.argmax(self.fitnesses)
            self.best_fit = self.fitnesses[idx_best]
            self.best_solution = self.population[idx_best]

            if (g + 1) % 10 == 0 or (g + 1) == generations:
                print(f"========= GEN {g+1} =========")
                print(f"Best sol: {self.decode(self.best_solution)}")
                print(f"Best fit: {self.best_fit}")

            best_tmp_idx = np.argsort(self.fitnesses)[-elitism:]
            new_population = [self.population[idx] for idx in best_tmp_idx]

            offsprings = []
            while len(offsprings) < (pop_size - elitism):
                par1, par2 = self.rank_selection()
                if rng.random() < rmx:
                    child1, child2 = self.crossover(par1, par2, eta_x)
                else:
                    child1, child2 = par1.copy(), par2.copy()
                if rng.random() < rmm:
                    child1 = self.mutation(child1)
                    child2 = self.mutation(child2)
                offsprings.extend([child1, child2])

            new_population.extend(offsprings[:pop_size - len(new_population)])
            self.population = new_population

        return self.decode(self.best_solution), self.best_fit


if __name__ == "__main__":
    with open("./data/kp.kp", "r", encoding='utf-8') as f:
        lines = f.readlines()
        n_items = int(lines[0])
        capacity = int(lines[1])
        weights = [0] * n_items
        values = [0] * n_items
        for i in range(n_items):
            weights[i], values[i] = map(int, lines[i + 2].split())

    ga = GA(n_items, capacity, weights, values)
    best_sol, best_fit = ga.ga(pop_size=50, generations=500)
    print("Best solution:", best_sol)
    print("Best fitness:", best_fit)
    print(capacity)
    print("Total weight:", sum(weights[i] * best_sol[i] for i in range(len(best_sol))))
