# import numpy as np
# from typing import Tuple
# import logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# np.random.seed(42)


# def neighbor_counts(labels: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     pop_eq3 = (labels == 3).astype(np.int32)
#     pop_eq2 = (labels == 2).astype(np.int32)
#     pop_ge2 = (labels >= 2).astype(np.int32)
#     count3 = pop_eq3 @ A
#     count2 = pop_eq2 @ A
#     count_ge2 = pop_ge2 @ A
#     return count3, count2, count_ge2

# def feasible_mask(labels: np.ndarray, A: np.ndarray) -> np.ndarray:
#     count3, count2, count_ge2 = neighbor_counts(labels, A)
#     is0 = (labels == 0)
#     is1 = (labels == 1)
#     ok0 = (count3 >= 1) | (count2 >= 2)
#     ok1 = (count_ge2 >= 1)
#     ok = np.ones_like(labels, dtype=bool)
#     ok &= (~is0 | ok0)
#     ok &= (~is1 | ok1)
#     return ok

# def is_feasible(labels: np.ndarray, A: np.ndarray) -> bool:
#     return feasible_mask(labels, A).all()

# def weight_of(labels: np.ndarray) -> int:
#     return int(np.sum(labels))

# def make_feasible(labels: np.ndarray, A: np.ndarray, max_iter=5) -> np.ndarray:
#     labels = labels.copy()
#     for _ in range(max_iter):
#         ok = feasible_mask(labels, A)
#         if ok.all():
#             break
#         bad = ~ok
#         labels[bad] = 2
#     return labels


# def parse_mtx_pattern_symmetric(mtx_text: str) -> np.ndarray:
#     lines = mtx_text.strip().splitlines()
#     data_lines = [ln for ln in lines if ln.strip() and not ln.startswith('%')]
#     header = data_lines[0].split()
#     nrows, ncols, nnz = map(int, header[:3])
#     if nrows != ncols:
#         raise ValueError("Matrix must be square")

#     n = nrows
#     A = np.zeros((n, n), dtype=np.int8)
#     for ln in data_lines[1:]:
#         i, j = map(int, ln.split()[:2])
#         A[i - 1, j - 1] = 1
#         A[j - 1, i - 1] = 1
#     return A


# class GraphEnvironment:
#     def __init__(self, A: np.ndarray):
#         self.A = A
#         self.n_nodes = A.shape[0]

#     def step(self, labels: np.ndarray, action: tuple):
#         node, new_label = action
#         new_labels = labels.copy()
#         new_labels[node] = new_label

#         reward = -weight_of(new_labels)
#         if not is_feasible(new_labels, self.A):
#             reward -= 100

#         done = is_feasible(new_labels, self.A)
#         return new_labels, reward, done

# class DoubleQLearning:
#     def __init__(self, env: GraphEnvironment, n_labels=4,
#                  alpha=0.1, gamma=0.9, epsilon=0.2):
#         self.env = env
#         self.n_nodes = env.n_nodes
#         self.n_labels = n_labels
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon

#         self.Q1 = np.zeros((self.n_nodes, n_labels))
#         self.Q2 = np.zeros((self.n_nodes, n_labels))

#     def choose_action(self, labels: np.ndarray, node: int):
#         if np.random.rand() < self.epsilon:
#             return np.random.randint(self.n_labels)
#         return np.argmax(self.Q1[node] + self.Q2[node])

#     def train(self, n_episodes=500, max_steps=50):
#         best_labels = None
#         best_weight = float("inf")

#         for ep in range(n_episodes):
#             labels = np.zeros(self.n_nodes, dtype=int)

#             for step_idx in range(max_steps):
#                 node = np.random.randint(self.n_nodes)
#                 action_label = self.choose_action(labels, node)
#                 action = (node, action_label)

#                 new_labels, reward, done = self.env.step(labels, action)

#                 # Double Q update
#                 if np.random.rand() < 0.5:
#                     best_next = np.argmax(self.Q1, axis=1)
#                     self.Q1[node, action_label] += self.alpha * (
#                         reward + self.gamma * self.Q2[node, best_next[node]] - self.Q1[node, action_label]
#                     )
#                 else:
#                     best_next = np.argmax(self.Q2, axis=1)
#                     self.Q2[node, action_label] += self.alpha * (
#                         reward + self.gamma * self.Q1[node, best_next[node]] - self.Q2[node, action_label]
#                     )

#                 labels = new_labels
#                 if done:
#                     break

#             if not is_feasible(labels, self.env.A):
#                 labels = make_feasible(labels, self.env.A)

#             w = weight_of(labels)
#             if w < best_weight:
#                 best_weight = w
#                 best_labels = labels.copy()
#                 logging.info(f"New best value: {best_weight} at epoch {ep}")
#             print(f"Epoch {ep}: {best_weight}")
#         return best_labels

# if __name__ == "__main__":
#     mtx_path = r"D:\Materials\EC_code\data\DROMD\ash85.mtx"
#     with open(mtx_path, "r") as f:
#         mtx_text = f.read()
#     A = parse_mtx_pattern_symmetric(mtx_text)

#     env = GraphEnvironment(A)
#     agent = DoubleQLearning(env)

#     labels = agent.train(n_episodes=500, max_steps=200)

#     print("Labels sinh ra:", labels)
#     print("Feasible?", is_feasible(labels, A))
#     print("Tổng weight:", weight_of(labels))


import numpy as np
from typing import Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

np.random.seed(42)

# =====================================================================
#  FEASIBILITY UTILITIES
# =====================================================================

def neighbor_counts(labels: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pop_eq3 = (labels == 3).astype(np.int32)
    pop_eq2 = (labels == 2).astype(np.int32)
    pop_ge2 = (labels >= 2).astype(np.int32)

    count3 = pop_eq3 @ A
    count2 = pop_eq2 @ A
    count_ge2 = pop_ge2 @ A

    return count3, count2, count_ge2

def feasible_mask(labels: np.ndarray, A: np.ndarray) -> np.ndarray:
    count3, count2, count_ge2 = neighbor_counts(labels, A)
    is0 = (labels == 0)
    is1 = (labels == 1)

    ok0 = (count3 >= 1) | (count2 >= 2)
    ok1 = (count_ge2 >= 1)

    ok = np.ones_like(labels, dtype=bool)
    ok &= (~is0 | ok0)
    ok &= (~is1 | ok1)
    return ok

def is_feasible(labels: np.ndarray, A: np.ndarray) -> bool:
    return feasible_mask(labels, A).all()

def weight_of(labels: np.ndarray) -> int:
    return int(np.sum(labels))

def make_feasible(labels: np.ndarray, A: np.ndarray, max_iter=5) -> np.ndarray:
    labels = labels.copy()
    for _ in range(max_iter):
        ok = feasible_mask(labels, A)
        if ok.all():
            break
        labels[~ok] = 2
    return labels


# =====================================================================
#  PARSE .MTX MATRIX
# =====================================================================

def parse_mtx_pattern_symmetric(mtx_text: str) -> np.ndarray:
    lines = mtx_text.strip().splitlines()
    data_lines = [ln for ln in lines if ln.strip() and not ln.startswith('%')]

    header = data_lines[0].split()
    nrows, ncols, nnz = map(int, header[:3])

    if nrows != ncols:
        raise ValueError("Matrix must be square")

    A = np.zeros((nrows, nrows), dtype=np.int8)

    for ln in data_lines[1:]:
        i, j = map(int, ln.split()[:2])
        A[i - 1, j - 1] = 1
        A[j - 1, i - 1] = 1

    return A


# =====================================================================
#  ENVIRONMENT WITH DEGREE AS STATE FEATURE
# =====================================================================

class GraphEnvironment:
    def __init__(self, A: np.ndarray):
        self.A = A
        self.n_nodes = A.shape[0]
        self.degree = np.sum(A, axis=1)   # <--- NEW: STATE FEATURE

    def step(self, labels: np.ndarray, action: tuple):
        node, new_label = action
        new_labels = labels.copy()
        new_labels[node] = new_label

        reward = -weight_of(new_labels)

        # Reward shaping theo bậc của đỉnh
        reward -= 0.1 * self.degree[node]    # <--- NEW

        # Phạt nặng nếu không hợp lệ
        if not is_feasible(new_labels, self.A):
            reward -= 100

        done = is_feasible(new_labels, self.A)
        return new_labels, reward, done


# =====================================================================
#  DOUBLE Q-LEARNING
# =====================================================================

class DoubleQLearning:
    def __init__(self, env: GraphEnvironment,
                 n_labels=4, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.env = env
        self.n_nodes = env.n_nodes
        self.n_labels = n_labels

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q1 = np.zeros((self.n_nodes, n_labels))
        self.Q2 = np.zeros((self.n_nodes, n_labels))

    # ---------------------------------------------------------------
    # ƯU TIÊN CHỌN NODE CÓ DEGREE CAO
    # ---------------------------------------------------------------
    def choose_node(self):
        if np.random.rand() < 0.8:
            # Softmax theo degree
            probs = self.env.degree / np.sum(self.env.degree)
            return np.random.choice(self.n_nodes, p=probs)
        else:
            return np.random.randint(self.n_nodes)

    # ---------------------------------------------------------------
    # CHỌN ACTION/ LABEL
    # ---------------------------------------------------------------
    def choose_action(self, labels: np.ndarray, node: int):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_labels)
        return np.argmax(self.Q1[node] + self.Q2[node])

    # ---------------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------------
    def train(self, n_episodes=500, max_steps=50):
        best_labels = None
        best_weight = float("inf")

        for ep in range(n_episodes):
            labels = np.zeros(self.n_nodes, dtype=int)

            for step in range(max_steps):

                node = self.choose_node()
                action_label = self.choose_action(labels, node)

                new_labels, reward, done = self.env.step(labels, (node, action_label))

                # Double-Q update
                if np.random.rand() < 0.5:
                    best_next = np.argmax(self.Q1, axis=1)
                    self.Q1[node, action_label] += self.alpha * (
                        reward + self.gamma * self.Q2[node, best_next[node]] - self.Q1[node, action_label]
                    )
                else:
                    best_next = np.argmax(self.Q2, axis=1)
                    self.Q2[node, action_label] += self.alpha * (
                        reward + self.gamma * self.Q1[node, best_next[node]] - self.Q2[node, action_label]
                    )

                labels = new_labels
                if done:
                    break

            # Fix cuối
            labels = make_feasible(labels, self.env.A)

            w = weight_of(labels)
            if w < best_weight:
                best_weight = w
                best_labels = labels.copy()
                logging.info(
                    f"[BEST] weight={best_weight} at epoch={ep} "
                    f"- avg_degree={np.mean(self.env.degree):.2f}, max_degree={np.max(self.env.degree)}"
                )

            print(f"Epoch {ep} → best weight: {best_weight}")

        return best_labels


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    mtx_path = r"D:\Materials\EC_code\data\DROMD\ash85.mtx"
    with open(mtx_path, "r") as f:
        mtx_text = f.read()

    A = parse_mtx_pattern_symmetric(mtx_text)

    env = GraphEnvironment(A)
    agent = DoubleQLearning(env)

    labels = agent.train(n_episodes=500, max_steps=200)

    print("Labels sinh ra:", labels)
    print("Feasible?", is_feasible(labels, A))
    print("Tổng weight:", weight_of(labels))
