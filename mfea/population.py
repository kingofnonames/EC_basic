import numpy as np
from typing import List
from mfea.task import Task
rng = np.random.default_rng(seed=22)

class Individual:
    def __init__(self, gen : np.ndarray, fitness_tasks: List[float] | None):
        self.gen = gen
        self.skill_factor = None
        self.fitness_tasks = fitness_tasks
        self.factorial_rank = None
        self.scalar_fitness = None

class Population:
    def __init__(self, pop_size: int, tasks: List[Task]):
        self.tasks = tasks
        self.pop_size = pop_size
        self.individuals = None
        self.len_gen = max(task.dimension for task in tasks)
    
    def init_population(self):
        individuals = []
        for _ in range(self.pop_size):
            gen = rng.uniform(size=self.len_gen)
            if not self.check_valid_gen(gen):
                gen = self.make_valid_gen(gen)
            fitness_ta = [task.compute_fitness(gen) for task in self.tasks]
            individuals.append(Individual(gen, fitness_ta))
        self.individuals = individuals
        self.update_rank_population()

    def check_valid_gen(self, gen) -> bool:
        return all(task.check_valid_gen(gen) for task in self.tasks)

    def make_valid_gen(self, gen):
        gen_valid = gen.copy()
        for task in self.tasks:
            if not task.check_valid_gen(gen_valid):
                gen_valid = task.make_valid_gen(gen_valid)
        return gen_valid
            
    def update_rank_population(self):
        self._assign_ranks(self.individuals, self.pop_size)

    def add_offsprings(self, offsprings: List[Individual]):
        self.individuals.extend(offsprings)
        self._assign_ranks(self.individuals, self.pop_size)

    def _assign_ranks(self, individuals: List[Individual], base_size: int):
        num_tasks = len(self.tasks)
        num_inds = len(individuals)
        rank_in_task = np.full((num_inds, num_tasks), fill_value=num_inds + 1)

        for task in range(num_tasks):
            rank = [num_inds + 1] * num_inds
            idx_sort = sorted(range(num_inds), key=lambda i: individuals[i].fitness_tasks[task])
            for idx, ind_idx in enumerate(idx_sort):
                rank[ind_idx] = idx + 1
            rank_in_task[:, task] = np.array(rank)

        for i in range(num_inds):
            ind = individuals[i]
            ind.factorial_rank = list(rank_in_task[i, :])
            ind.skill_factor = np.argmin(ind.factorial_rank)
            ind.scalar_fitness = 1.0 / min(ind.factorial_rank)

    def get_best_individual_per_task(self, task):
        return min(self.individuals, key=lambda x: x.factorial_rank[task])