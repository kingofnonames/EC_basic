import numpy as np
from mfea2.task import Task, TSP, Knapsack
from mfea2.population import Individual, Population
from typing import List
import matplotlib.pyplot as plt
rng = np.random.default_rng(seed=227)

class GA:
    def __init__(self, 
                 tasks: List[Task], 
                 popsize: int=100, 
                 pm: float=0.3, 
                 generation: int=200, 
                 selection_type: str="tournament",
                 crossover_type: str="sbx",
                 mutation_type: str="random",
                 ):
        self.tasks = tasks
        self.popsize = popsize
        self.pm = pm
        self.population = Population(self.popsize, self.tasks)
        self.generation = generation
        self.best_solution = None
        self.mutation_type = mutation_type
        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.best_scores = [[] for _ in range(len(self.tasks))]

    def run(self):
        num_tasks = len(self.tasks)
        self.population.init_population()
        self.best_solution = [self.population.individuals[i] for i in range(num_tasks)]
        for gen in range(self.generation):
            if gen % 10 == 0 or gen + 1 == self.generation:
                print(f"===================GEN {gen}===================")
            for t in range(num_tasks):
                ind = self.population.get_best_individual_per_task(t)
                if self.best_solution[t].fitness_tasks[t] < ind.fitness_tasks[t]:
                    self.best_solution[t] = ind
                self.best_scores[t].append(self.best_solution[t].fitness_tasks[t])
                if gen % 10 == 0 or gen + 1 == self.generation:
                    if t == 0:
                        print(f"Task {t}: {self.tasks[t].compute_distance(self.best_solution[t].gen)}")
                    else:
                        print(f"Task {t}: {self.best_solution[t].fitness_tasks[t]} | Weight: {self.tasks[t].sum_weight(ind.gen)}")
            offsprings = []
            for _ in range(self.popsize // 2):
                par1, par2 = self.selection()
                skill_1 = par1.skill_factor
                skill_2 = par2.skill_factor
                if (skill_1 == skill_2) or rng.uniform() > self.pm:
                    child1, child2 = self.crossover(par1, par2)
                    offsprings.extend([child1, child2])
                else:
                    offsprings.extend([self.mutation(par1), self.mutation(par2)])
            self.population.add_offsprings(offsprings)
            self.population.individuals = sorted(self.population.individuals, key=lambda x: x.scalar_fitness)[-self.popsize:]
            self.population.update_rank_population()

    def selection(self) -> List[Individual]:
        selected = self.__getattribute__(f"selection_{self.selection_type}")()
        return selected
        
    def selection_tournament(self, k: int = 3):
        inds = self.population.individuals
        selected = []
        for _ in range(2):
            candidates = rng.choice(inds, size=k, replace=False)
            winner = max(candidates, key=lambda x: x.scalar_fitness)
            selected.append(winner)
        return selected
    
    def crossover(self, par1: Individual, par2: Individual):
        child1, child2 = self.__getattribute__(f"crossover_{self.crossover_type}")(par1, par2)
        return child1, child2

    def crossover_sbx(self, par1: Individual, par2: Individual, eta_c: float = 2.0):
        gen1 = par1.gen.copy()
        gen2 = par2.gen.copy()
        child1 = np.empty_like(gen1)
        child2 = np.empty_like(gen2)

        for i in range(len(gen1)):
            if rng.uniform() <= 0.5:
                if abs(gen1[i] - gen2[i]) > 1e-14:
                    x1, x2 = sorted([gen1[i], gen2[i]])
                    u = rng.uniform()
                    beta = (2 * u) ** (1.0 / (eta_c + 1)) if u <= 0.5 else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1))
                    child1[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                    child2[i] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                else:
                    child1[i], child2[i] = gen1[i], gen2[i]
            else:
                child1[i], child2[i] = gen1[i], gen2[i]

        child1 = np.clip(child1, 0, 1)
        child2 = np.clip(child2, 0, 1)
        return self._evaluate_children(child1, child2, par1, par2)
    
    def crossover_onepoint(self, par1: Individual, par2: Individual):
        gen_par1 = par1.gen.copy()
        gen_par2 = par2.gen.copy()
        t = rng.integers(1, len(gen_par1))
        child1_gen = np.concatenate([gen_par1[:t], gen_par2[t:]])
        child2_gen = np.concatenate([gen_par2[:t], gen_par1[t:]])
        return self._evaluate_children(child1_gen, child2_gen, par1, par2)

    def _evaluate_children(self, gen1, gen2, par1, par2):
        c1 = Individual(gen1, None)
        c2 = Individual(gen2, None)
        skill1 = par1.skill_factor if rng.uniform() < 0.5 else par2.skill_factor
        skill2 = par2.skill_factor if rng.uniform() < 0.5 else par1.skill_factor
        c1.skill_factor = skill1
        c2.skill_factor = skill2
        c1.fitness_tasks = [float("-inf")] * len(self.tasks)
        c2.fitness_tasks = [float("-inf")] * len(self.tasks)
        c1.fitness_tasks[skill1] = self.tasks[skill1].compute_fitness(gen1)
        c2.fitness_tasks[skill2] = self.tasks[skill2].compute_fitness(gen2)
        return c1, c2

    def mutation(self, ind: Individual, rate: float=0.3):
        return self.__getattribute__(f'mutation_{self.mutation_type}')(ind, rate)
    
    def mutation_random(self, ind: Individual, rate: float=0.3) -> Individual:
        new_gen = ind.gen.copy()
        len_gen = len(new_gen)
        mask = rng.uniform(size=len_gen) < rate
        new_gen[mask] = rng.uniform(size=np.sum(mask))
        new_gen = np.clip(new_gen, 0, 1)
        new_ind = Individual(new_gen, None)
        skill_factor = ind.skill_factor
        new_ind.skill_factor = skill_factor
        fitness_tasks = [float("-inf")] * len(self.tasks)
        fitness_tasks[skill_factor] = self.tasks[skill_factor].compute_fitness(new_gen)
        new_ind.fitness_tasks = fitness_tasks
        return new_ind

    def mutation_polynomial(self, ind: Individual, eta: float=2.0, rate: float=0.3) -> Individual:
        new_gen = ind.gen.copy()
        len_gen = len(new_gen)
        for i in range(len_gen):
            if rng.uniform() < rate:
                u = rng.uniform()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                new_gen[i] += delta
        new_gen = np.clip(new_gen, 0, 1)
        new_ind = Individual(new_gen, None)
        skill_factor = ind.skill_factor
        new_ind.skill_factor = skill_factor
        fitness_tasks = [float("-inf")] * len(self.tasks)
        fitness_tasks[skill_factor] = self.tasks[skill_factor].compute_fitness(new_gen)
        new_ind.fitness_tasks = fitness_tasks
        return new_ind
    

if __name__ == "__main__":
    tasks = []
    tasks.append(TSP())
    tasks.append(Knapsack())
    g = GA(tasks, 
           popsize=100, 
           pm=0.3, 
           mutation_type="polynomial",
           generation=500)
    g.run()