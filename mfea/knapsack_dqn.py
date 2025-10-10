import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

rng = np.random.default_rng(seed=22)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def push(self, transition):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        idx = rng.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in idx]
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s_next, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )


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
        beta = np.where(u < 0.5, (2 * u) ** (1.0 / (eta_x + 1)),
                        (0.5 / (1 - u)) ** (1.0 / (eta_x + 1)))
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

    def ga_dqn(self, dqn, target_dqn, buffer, optimizer, pop_size=100, generations=500,
               elitism=3, rmx=0.9, rmm=0.3, eta_x=7.0, epsilon=0.1,
               gamma=0.9, batch_size=32):

        self.population = self.init_population(pop_size)
        state_dim = 4
        action_dim = 5
        old_best = 0

        for g in range(generations):
            self.fitnesses = np.array([self.compute_fitness(pop) for pop in self.population])
            idx_best = np.argmax(self.fitnesses)
            self.best_fit = self.fitnesses[idx_best]
            self.best_solution = self.population[idx_best]

            mean_fit = np.mean(self.fitnesses)
            std_fit = np.std(self.fitnesses)
            state = np.array([self.best_fit, mean_fit, std_fit, g / generations], dtype=np.float32)

            if rng.random() < epsilon:
                action = rng.integers(action_dim)
            else:
                with torch.no_grad():
                    q_values = dqn(torch.tensor(state).unsqueeze(0))
                    action = int(torch.argmax(q_values))

            if action == 0:
                rmm = max(0.05, rmm - 0.05)
            elif action == 1:
                rmm = min(0.9, rmm + 0.05)
            elif action == 2:
                rmx = max(0.4, rmx - 0.05)
            elif action == 3:
                rmx = min(0.95, rmx + 0.05)

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

            self.fitnesses = np.array([self.compute_fitness(pop) for pop in self.population])
            idx_best = np.argmax(self.fitnesses)
            new_best = self.fitnesses[idx_best]
            mean_fit = np.mean(self.fitnesses)
            std_fit = np.std(self.fitnesses)
            next_state = np.array([new_best, mean_fit, std_fit, (g + 1) / generations], dtype=np.float32)
            reward = new_best - old_best
            old_best = new_best
            done = float(g + 1 == generations)

            buffer.push((state, action, reward, next_state, done))

            if len(buffer.buffer) >= batch_size:
                s, a, r, s_next, done = buffer.sample(batch_size)
                q_values = dqn(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target_q = r + gamma * (1 - done) * target_dqn(s_next).max(1)[0]
                loss = nn.MSELoss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dqn.parameters(), 1.0)
                optimizer.step()

            if g % 20 == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            if (g + 1) % 10 == 0 or (g + 1) == generations:
                print(f"=== GEN {g + 1} ===")
                print(f"Best fit: {new_best:.2f}, rmx={rmx:.2f}, rmm={rmm:.2f}, reward={reward:.2f}")

        return self.decode(self.best_solution), self.best_fit


if __name__ == "__main__":
    with open("./data/kp.kp", "r", encoding='utf-8') as f:
        lines = f.readlines()
        n_items = int(lines[0])
        capacity = int(lines[1])
        weights, values = [], []
        for i in range(n_items):
            w, v = map(int, lines[i + 2].split())
            weights.append(w)
            values.append(v)

    ga = GA(n_items, capacity, weights, values)
    state_dim, action_dim = 4, 5

    dqn = DQN(state_dim, action_dim)
    target_dqn = DQN(state_dim, action_dim)
    target_dqn.load_state_dict(dqn.state_dict())

    buffer = ReplayBuffer(max_size=5000)
    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)

    best_sol, best_fit = ga.ga_dqn(
        dqn, target_dqn, buffer, optimizer,
        pop_size=50, generations=500
    )

    print("\nBest solution:", best_sol)
    print("Best fitness:", best_fit)
    print("Total weight:", sum(weights[i] * best_sol[i] for i in range(len(best_sol))))
