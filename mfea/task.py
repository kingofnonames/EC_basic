import numpy as np
from typing import List
from abc import abstractmethod, ABC
rng = np.random.default_rng(seed=22)
# from mfea.population import Individual
class Task(ABC):
    def __init__(self):
        self.dimension = 0
        self.capacity = 0

    def stride(self, siz_gen: int, siz_win: int) -> List[int]:
        dimension = self.dimension
        pad_width = 0
        if (siz_gen - siz_win) % (dimension - 1) == 0:
            stride = int((siz_gen - siz_win) / (dimension - 1))
        else:
            stride = (siz_gen - siz_win) // (dimension - 1) + 1
            pad_width = stride * (dimension - 1) + siz_win - siz_gen
        return stride, pad_width
    
    @abstractmethod
    def check_valid_gen(self, gen: np.ndarray) -> bool:
        pass
    
    def make_valid_gen(self, gen: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def decode(self, gen: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def load_data(self, file_name: str):
        pass

    @abstractmethod
    def compute_fitness(self, gen:np.ndarray) -> float:
        pass

class TSP(Task):
    def __init__(self):
        super().__init__()
        self.cities = []
        self.dist_matrix = None
        self.load_data()
        self.window = rng.uniform(size=2)
    
    def load_data(self, file_name: str="D:/Test/mfea/data/tsp.tsp"):
        with open(file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.cities.append(np.array(list(map(int, line.split())))[1:])
        self.dimension = len(self.cities)
        self.dist_matrix = TSP.distance_matrix(self.cities)

    def euclid_distance(city_1: np.ndarray, city_2: np.ndarray) -> float:
        return np.linalg.norm(city_1 - city_2)
    
    def distance_matrix(cities: List[np.ndarray]) -> np.ndarray:
        num_cities = len(cities)
        dist_matrix = np.zeros(shape=(num_cities, num_cities))
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                dist_matrix[i, j] = dist_matrix[j, i] = TSP.euclid_distance(cities[i], cities[j])
        return dist_matrix
    
    def check_valid_gen(self, gen: np.ndarray) -> bool:
        return True

    def make_valid_gen(self, gen: np.ndarray) -> np.ndarray:
        return gen
    # def make_valid_gen(self, gen: np.ndarray) -> np.ndarray:
    #     return gen

    def compute_fitness(self, gen: np.ndarray) -> np.ndarray:
        tour = self.decode(gen)
        N = len(tour)
        return sum(self.dist_matrix[tour[i], tour[(i + 1) % N]] for i in range(N))
    
    def decode(self, gen: np.ndarray) -> np.ndarray:
        siz_window = len(self.window)
        siz_gen  = len(gen)
        dimension = self.dimension
        stride, pad_width = self.stride(siz_gen, siz_window)
        gen_real_dec = np.empty(shape=dimension)
        if pad_width > 0:
            gen_copy = np.pad(gen, (0, pad_width), mode="constant", constant_values=0)
        else:
            gen_copy = gen.copy()
        idx = 0
        for i in range(0, siz_gen, stride):
            gen_real_dec[idx] = np.sum(gen_copy[i:i+siz_window])
            idx += 1
        gen_dec = np.argsort(gen_real_dec)
        return gen_dec

class Knapsack(Task):
    def __init__(self):
        super().__init__()
        self.load_data()
        self.window = rng.uniform(size=4)
    
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
            self.profits = self.profits[idx_sorted]
            self.idx_item = np.arange(self.dimension)[idx_sorted]   

    def compute_fitness(self, gen: np.ndarray) -> float:
        gen_dec = self.decode(gen)
        N = len(gen_dec)
        fitness = -np.sum(self.prices[i] * gen_dec[i] for i in range(N))
        return fitness
    
    def sum_weight(self, gen: np.ndarray) -> float:
        N = len(gen)
        return np.sum(self.weights[i] * gen[i] for i in range(N))
    
    def check_valid_gen(self, gen: np.ndarray) -> bool:
        gen_dec = self.decode(gen)
        weight = self.sum_weight(gen_dec)
        return weight <= self.capacity
    
    def make_valid_gen(self, gen: np.ndarray) -> np.ndarray:
        gen_dec = self.decode(gen)
        current_weight = self.sum_weight(gen_dec)
        siz_gen = len(gen)
        siz_win = len(self.window)
        if siz_gen > self.dimension:
            stride, _ = self.stride(siz_gen, siz_win)
        else:
            stride = 1
        idx = 0
        while current_weight > self.capacity and idx < self.dimension:
            item = self.idx_item[idx]
            if gen_dec[item] == 1:
                if siz_gen > self.dimension:
                    start = stride * item
                    end = min(start + siz_win, siz_gen)
                    for i in range(start, end):
                        gen[i] = 0.5 / (siz_win * max(self.window))
                else:
                    gen[item] -= 0.5

                gen_dec[item] = 0
                current_weight -= self.weights[item]
            idx += 1
        return np.clip(gen, 0, 1)
    def decode(self, gen: np.ndarray) -> np.ndarray:
        gen_copy = gen.copy()
        siz_gen = len(gen)
        siz_win = len(self.window)
        gen_dec = np.zeros(self.dimension, dtype=np.int64)

        if siz_gen > self.dimension:
            stride, pad_width = self.stride(siz_gen, siz_win)
            gen_copy = np.pad(gen_copy, (0, pad_width), mode="constant", constant_values=0)
            for idx in range(self.dimension):
                start = idx * stride
                val = np.sum(gen_copy[start:start+siz_win] * self.window)
                gen_dec[idx] = 1 if val > 0.5 else 0
        else:
            for i in range(siz_gen):
                gen_dec[i] = 1 if gen_copy[i] > 0.5 else 0

        return gen_dec
if __name__ == "__main__":
    pass