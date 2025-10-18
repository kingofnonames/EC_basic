from abc import ABC, abstractmethod
import numpy as np
from typing import List
rng = np.random.default_rng(seed=227)
class Task(ABC):
    def __init__(self):
        self.dimension = 0
        self.capacity = 0
    
    @abstractmethod
    def decode(self, gen: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def compute_fitness(self, gen: np.ndarray) -> float:
        pass

    @abstractmethod
    def load_data(self, filepath: str):
        pass

class TSP(Task):
    def __init__(self):
        super().__init__()
        self.cities = []
        self.dist_matrix = None
        self.load_data()
    @staticmethod
    def euclid_distance(city_1: np.ndarray, city_2: np.ndarray) -> float:
        return np.linalg.norm(city_1 - city_2)
    
    @staticmethod
    def distance_matrix(cities: List[np.ndarray]) -> np.ndarray:
        num_cities = len(cities)
        dist_matrix = np.zeros(shape=(num_cities, num_cities))
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                dist_matrix[i, j] = dist_matrix[j, i] = TSP.euclid_distance(cities[i], cities[j])
        return dist_matrix

    def load_data(self, file_name: str="D:/Test/mfea/data/tsp.tsp"):
        with open(file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.cities.append(np.array(list(map(int, line.split())))[1:])
        self.dimension = len(self.cities)
        self.dist_matrix = TSP.distance_matrix(self.cities)

    def compute_fitness(self, gen: np.ndarray, alpha=1e-6) -> float:
        tour = self.decode(gen)
        distances = self.dist_matrix[tour, np.roll(tour, -1)]
        return 1.0 / (np.sum(distances) + alpha)

    def compute_distance(self, gen: np.ndarray) -> float:
        tour = self.decode(gen)
        return np.sum(self.dist_matrix[tour, np.roll(tour, -1)])

    def decode(self, gen: np.ndarray) -> np.ndarray:
        N = self.dimension
        gen_dec = gen[:N]
        gen_dec = np.argsort(gen_dec)
        return gen_dec

class Knapsack(Task):
    def __init__(self):
        super().__init__()
        self.load_data()

    def load_data(self, file_name="D:/Test/mfea/data/kp.kp"):
        with open(file_name, "r") as f:
            lines = f.readlines()
            self.dimension = int(lines[0].strip())
            self.capacity  = int(lines[1].strip())
            self.weights = np.empty(self.dimension, dtype=int)
            self.prices  = np.empty(self.dimension, dtype=int)
            self.profits = np.empty(self.dimension, dtype=float)
            
            for i in range(2, len(lines)):
                w, p = map(int, lines[i].split())
                self.weights[i - 2] = w
                self.prices[i - 2]  = p
                self.profits[i - 2] = p / max(1, w)
            idx_sorted = np.argsort(self.profits)
            self.idx_item = np.arange(self.dimension)[idx_sorted]
    
    def compute_fitness(self, gen: np.ndarray) -> float:
        gen_dec = self.decode(gen)
        fitness = np.sum(self.prices * gen_dec)
        return fitness
    
    def sum_weight(self, gen: np.ndarray) -> float:
        gen_dec = self.decode(gen)
        return np.sum(self.weights * gen_dec)
    
    def decode(self, gen: np.ndarray) -> np.ndarray:
        N = self.dimension
        gen_dec = np.zeros(shape=N, dtype=int)
        ranks = np.arange(start=1, stop=N + 1)
        value_ranks = ranks / sum(ranks)

        gen_sum = gen[:N].copy()
        gen_sum[self.idx_item] += value_ranks
        sorted_indices = np.argsort(gen_sum)[::-1]
        remaining_capacity = self.capacity
        for idx in sorted_indices:
            if self.weights[idx] <= remaining_capacity:
                gen_dec[idx] = 1
                remaining_capacity -= self.weights[idx]
            else:
                gen_dec[idx] = 0

        return gen_dec