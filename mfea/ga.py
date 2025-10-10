import numpy as np
from mfea.task import Task, TSP, Knapsack
from mfea.population import Individual, Population
from typing import List
from copy import deepcopy
rng = np.random.default_rng(seed=22)

class GA:
    def __init__(self, tasks: List[Task], pop_size: int, pm: float, generation=200):
        self.tasks = tasks
        self.pop_size = pop_size
        self.pm = pm
        self.population = Population(self.pop_size, self.tasks)
        self.generation = generation
        self.best_solution = None

    def run(self, num_child=100):
        num_tasks = len(self.tasks)
        self.population.init_population()
        self.best_solution = [self.population.individuals[i] for i in range(num_tasks)]

        for gen in range(self.generation):
            if gen % 10 == 0 or gen + 1 == self.generation:
                print(f"===================GEN {gen}===================")
            for t in range(num_tasks):
                ind = self.population.get_best_individual_per_task(t)
                if self.best_solution[t].fitness_tasks[t] > ind.fitness_tasks[t]:
                    self.best_solution[t] = ind
                if gen % 10 == 0 or gen + 1 == self.generation:
                    print(f"Task {t}: {self.best_solution[t].fitness_tasks[t]}")
            individuals = self.population.individuals
            offsprings = []
            for _ in range(num_child):
                idx_1, idx_2 = rng.choice(len(individuals), size=2, replace=False)
                par1 = individuals[idx_1]
                par2 = individuals[idx_2]
                skill_1 = par1.skill_factor
                skill_2 = par2.skill_factor
                if (skill_1 == skill_2) or rng.uniform() > self.pm:
                    child1 = self.crossover(par1, par2)
                    child2 = self.crossover(par2, par1)
                    offsprings.extend([child1, child2])
                else:
                    offsprings.extend([self.mutation(par1), self.mutation(par2)])
            self.population.add_offsprings(offsprings)
            self.selection()
            # self.re_compute_fitness_for_child(offsprings)
            self.population.update_rank_population()


    def re_compute_fitness_for_child(self, offsprings: List[Individual]):
        for off in offsprings:
            for j, task in enumerate(self.tasks):
                if off.fitness_tasks[j] == float("inf"):
                    off.fitness_tasks[j] = task.compute_fitness(off.gen)
        
    def mutation(self, ind: Individual, rate=0.2) -> Individual:
        new_gen = ind.gen.copy()
        len_gen = len(new_gen)
        mask = rng.uniform(size=len_gen) < rate
        new_gen[mask] = rng.uniform(size=np.sum(mask))
        if not self.population.check_valid_gen(new_gen):
            self.population.make_valid_gen(new_gen)
        new_ind = Individual(new_gen, None)
        skill_factor = ind.skill_factor
        new_ind.skill_factor = skill_factor
        fitness_tasks = [float("inf")] * len(self.tasks)
        fitness_tasks[skill_factor] = self.tasks[skill_factor].compute_fitness(new_gen)
        new_ind.fitness_tasks = fitness_tasks
        return new_ind
    
    def selection(self):
        self.population.individuals = sorted(self.population.individuals, key=lambda x: x.scalar_fitness)[-self.pop_size:]

    def crossover(self, par1: Individual, par2: Individual) -> Individual:
        gen_par1 = par1.gen.copy()
        gen_par2 = par2.gen.copy()
        t = rng.integers(1, len(gen_par1))
        gen_child = np.concatenate([gen_par1[:t], gen_par2[t:]])
        if not self.population.check_valid_gen(gen_child):
            self.population.make_valid_gen(gen_child)
        new_child = Individual(gen_child, None)
        skill_factor = par1.skill_factor if rng.uniform() < 0.5 else par2.skill_factor
        fitness_tasks = [float("inf")] * len(self.tasks)
        new_child.skill_factor = skill_factor
        fitness_tasks[skill_factor] = self.tasks[skill_factor].compute_fitness(gen_child)
        new_child.fitness_tasks = fitness_tasks
        return new_child
    

if __name__ == "__main__":
    tasks = []
    tasks.append(TSP())
    tasks.append(Knapsack())
    g = GA(tasks, pop_size=100, pm=0.3)
    g.run()