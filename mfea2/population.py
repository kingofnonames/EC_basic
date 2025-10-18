import numpy as np
from .task import Task
from typing import List
rng = np.random.default_rng(seed=227)
class Individual:
    def __init__(self, gen: np.ndarray, fitness_tasks: List[float]):
        self.gen = gen
        self.skill_factor = None
        self.fitness_tasks = fitness_tasks
        self.scalar_fitness = None
        self.factorial_rank = None


class Population:
    def __init__(self, popsize: int, tasks: List[Task]):
        self.tasks = tasks
        self.popsize = popsize
        self.individuals = None
        self.dimension = max(task.dimension for task in tasks)
    
    def init_population(self):
        individuals = []
        gens = rng.uniform(size=(self.popsize, self.dimension))
        for gen in gens:
            fitness_ta = [task.compute_fitness(gen) for task in self.tasks]
            individuals.append(Individual(gen.copy(), fitness_ta))
        self.individuals = individuals
        self.update_rank_population()

    def add_offsprings(self, offsprings: List[Individual]):
        if self.individuals is None:
            self.individuals = []
        self.individuals.extend(offsprings)
        self.update_rank_population()

    def update_rank_population(self):
        self._assign_ranks()

    def _assign_ranks(self):
        num_tasks = len(self.tasks)
        num_inds = len(self.individuals)
        fitness_matrix = np.array([ind.fitness_tasks for ind in self.individuals])
        sorted_indices = np.argsort(-fitness_matrix, axis=0)
        rank_in_tasks = np.empty_like(sorted_indices, dtype=int)
        for task in range(num_tasks):
            rank_in_tasks[sorted_indices[:, task], task] = np.arange(1, num_inds + 1)
        min_ranks = np.min(rank_in_tasks, axis=1)
        skill_factors = np.argmin(rank_in_tasks, axis=1)
        scalar_fitnesses = 1.0 / min_ranks
        for i, ind in enumerate(self.individuals):
            ind.factorial_rank = rank_in_tasks[i, :].tolist()
            ind.skill_factor = int(skill_factors[i])
            ind.scalar_fitness = float(scalar_fitnesses[i])
    
    def get_best_individual_per_task(self, task_index: int):
        return max(self.individuals, key=lambda ind: ind.fitness_tasks[task_index])
if __name__ == "__main__":
    pass
        
        