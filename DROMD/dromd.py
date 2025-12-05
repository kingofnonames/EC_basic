import networkx as nx
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from scipy.io import mmread
import os

# ==============================================================================
# PHẦN 0: HÀM ĐỌC DỮ LIỆU
# ==============================================================================
def load_mtx_graph(filepath):
    """
    Đọc file .mtx và tiền xử lý cho bài toán MDR.
    """
    print(f"-> Đang đọc file: {filepath}...")
    try:
        sparse_matrix = mmread(filepath)
        G = nx.from_scipy_sparse_array(sparse_matrix)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None

    # Tiền xử lý
    if G.is_directed():
        G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    if not nx.is_connected(G):
        print("   (Cảnh báo: Đồ thị gốc không liên thông. Đang trích xuất thành phần lớn nhất...)")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
    G = nx.convert_node_labels_to_integers(G)
    print(f"-> Đồ thị đã tải: {len(G.nodes())} đỉnh, {len(G.edges())} cạnh.")
    return G

# ==============================================================================
# PHẦN 1: ĐỊNH NGHĨA 2 BÀI TOÁN (TASKS) - ĐÃ TỐI ƯU HÓA
# ==============================================================================
class MFEA_Tasks:
    def __init__(self, graph):
        self.G = graph
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)
        self.idx2node = {i: node for i, node in enumerate(self.nodes)}

    # --------------------------------------------------------------------------
    # TASK 1: DOUBLE ROMAN DOMINATION (MDR) - CẢI TIẾN
    # --------------------------------------------------------------------------
    def decode_mdr(self, genotype):
        solution = {}
        for i, val in enumerate(genotype):
            node = self.idx2node[i]
            # Quy tắc ngưỡng
            if val < 0.6: label = 0
            elif val < 0.85: label = 2
            else: label = 3
            solution[node] = label
        return solution

    def is_feasible_global(self, solution):
        """Kiểm tra nhanh tính khả thi của toàn bộ giải pháp"""
        for u in self.nodes:
            val = solution[u]
            if val == 0:
                neighbors = list(self.G.neighbors(u))
                n3 = sum(1 for v in neighbors if solution[v] == 3)
                n2 = sum(1 for v in neighbors if solution[v] == 2)
                if not (n3 >= 1 or n2 >= 2): return False
            elif val == 1:
                neighbors = list(self.G.neighbors(u))
                nge2 = sum(1 for v in neighbors if solution[v] >= 2)
                if nge2 < 1: return False
        return True

    def repair_mdr_smart(self, solution):
        """
        [NÂNG CẤP] Sửa lỗi thông minh:
        Thay vì chỉ tăng bản thân lên 2 (cost +2), thử tìm hàng xóm đang là 2 
        để tăng lên 3 (cost +1) nhằm bảo vệ mình.
        """
        repaired_sol = solution.copy()
        # Duyệt qua các đỉnh cần được bảo vệ
        for u in self.nodes:
            if repaired_sol[u] == 0:
                neighbors = list(self.G.neighbors(u))
                n3 = sum(1 for v in neighbors if repaired_sol[v] == 3)
                n2 = sum(1 for v in neighbors if repaired_sol[v] == 2)
                
                # Nếu không thỏa mãn điều kiện
                if not (n3 >= 1 or n2 >= 2):
                    # Chiến lược tối ưu:
                    # Tìm xem có hàng xóm nào đang là 2 không?
                    neighbors_2 = [v for v in neighbors if repaired_sol[v] == 2]
                    
                    if neighbors_2:
                        # Chọn hàng xóm bậc cao nhất để nâng lên 3 (lợi nhất)
                        best_nbr = max(neighbors_2, key=lambda x: self.G.degree(x))
                        repaired_sol[best_nbr] = 3
                        # Giờ u đã an toàn (có 1 hàng xóm là 3). Cost chỉ tăng 1.
                    else:
                        # Không có cách nào rẻ hơn, đành tự tăng lên 2. Cost tăng 2.
                        repaired_sol[u] = 2
        return repaired_sol

    def optimize_mdr_fast(self, solution):
        sorted_nodes = sorted(self.nodes, key=lambda x: self.G.degree(x))
        improved_sol = solution.copy()
        
        for u in sorted_nodes:
            original_val = improved_sol[u]
            if original_val == 0: continue

            # Hàm kiểm tra cục bộ (Delta Check)
            def is_safe_locally(node_u, proposed_val, current_sol):
                # 1. Kiểm tra chính node_u có an toàn không với nhãn mới?
                if proposed_val == 0:
                    nbrs = list(self.G.neighbors(node_u))
                    n3 = sum(1 for v in nbrs if current_sol[v] == 3)
                    n2 = sum(1 for v in nbrs if current_sol[v] == 2)
                    if not (n3 >= 1 or n2 >= 2): return False
                elif proposed_val == 1:
                    # (Bài toán này ít dùng nhãn 1, nhưng cứ check cho đủ)
                    nbrs = list(self.G.neighbors(node_u))
                    nge2 = sum(1 for v in nbrs if current_sol[v] >= 2)
                    if nge2 < 1: return False
                
                # 2. Kiểm tra các HÀNG XÓM của node_u
                # Việc giảm nhãn của u (ví dụ 3->0) có làm hại hàng xóm không?
                for v in self.G.neighbors(node_u):
                    val_v = current_sol[v]
                    # Chỉ quan tâm nếu hàng xóm là 0 hoặc 1 (các đỉnh cần bảo vệ)
                    if val_v == 0 or val_v == 1:
                        # Tạm thời giả lập nhãn mới của u trong calculation
                        # Check lại điều kiện an toàn cho v
                        nbrs_of_v = list(self.G.neighbors(v))
                        # Đếm lại n3, n2 của v với giả định u đã đổi nhãn
                        n3_v = 0
                        n2_v = 0
                        has_ge2 = 0
                        for w in nbrs_of_v:
                            label_w = proposed_val if w == node_u else current_sol[w]
                            if label_w == 3: n3_v += 1
                            if label_w == 2: n2_v += 1
                            if label_w >= 2: has_ge2 += 1
                        
                        if val_v == 0 and not (n3_v >= 1 or n2_v >= 2): return False
                        if val_v == 1 and has_ge2 < 1: return False
                        
                return True

            # Thử giảm về 0
            if is_safe_locally(u, 0, improved_sol):
                improved_sol[u] = 0
                continue
            
            # Thử giảm về 2 (nếu đang là 3)
            if original_val == 3:
                if is_safe_locally(u, 2, improved_sol):
                    improved_sol[u] = 2
                    continue
            
        return improved_sol

    def calculate_mdr_cost(self, genotype):
        """Quy trình tính Cost đầy đủ với các bước tối ưu"""
        # 1. Giải mã thô
        sol = self.decode_mdr(genotype)
        # 2. Sửa lỗi thông minh
        sol = self.repair_mdr_smart(sol)
        
        return sum(sol.values())

    # --------------------------------------------------------------------------
    # TASK 2: GRAPH COLORING PROBLEM (GCP)
    # --------------------------------------------------------------------------
    def decode_gcp(self, genotype):
        indices = np.argsort(genotype)[::-1]
        sorted_nodes = [self.idx2node[i] for i in indices]
        
        colors = {}
        for node in sorted_nodes:
            neighbor_colors = {colors[nbr] for nbr in self.G.neighbors(node) if nbr in colors}
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[node] = c
        return colors

    def calculate_gcp_cost(self, genotype):
        colors = self.decode_gcp(genotype)
        return len(set(colors.values()))

# ==============================================================================
# PHẦN 2: CẤU TRÚC CÁ THỂ VÀ THUẬT TOÁN MFEA (GIỮ NGUYÊN)
# ==============================================================================
class Individual:
    def __init__(self, dim):
        self.genotype = np.random.rand(dim)
        self.skill_factor = None
        self.factorial_costs = [float('inf'), float('inf')]
        self.scalar_fitness = float('inf')

class MFEA:
    def __init__(self, tasks, pop_size=100, generations=100, rmp=0.3):
        self.tasks = tasks
        self.dim = tasks.n
        self.pop_size = pop_size
        self.generations = generations
        self.rmp = rmp
        self.population = []

    def run(self):
        # KHỞI TẠO
        for _ in range(self.pop_size):
            ind = Individual(self.dim)
            cost_mdr = self.tasks.calculate_mdr_cost(ind.genotype)
            cost_gcp = self.tasks.calculate_gcp_cost(ind.genotype)
            ind.factorial_costs = [cost_mdr, cost_gcp]
            ind.skill_factor = random.choice([0, 1])
            self.population.append(ind)

        self.assign_scalar_fitness()
        history_mdr, history_gcp = [], []

        # TIẾN HÓA
        for gen in range(self.generations):
            offspring_pop = []
            random.shuffle(self.population)
            
            for i in range(0, self.pop_size, 2):
                p1 = self.population[i]
                p2 = self.population[i+1]
                c1 = Individual(self.dim)
                c2 = Individual(self.dim)
                
                can_mate = (p1.skill_factor == p2.skill_factor) or (random.random() < self.rmp)
                
                if can_mate:
                    mask = np.random.rand(self.dim) < 0.5
                    c1.genotype = np.where(mask, p1.genotype, p2.genotype)
                    c2.genotype = np.where(mask, p2.genotype, p1.genotype)
                    c1.skill_factor = random.choice([p1.skill_factor, p2.skill_factor])
                    c2.skill_factor = random.choice([p1.skill_factor, p2.skill_factor])
                else:
                    c1 = self.mutate(p1)
                    c2 = self.mutate(p2)
                
                if random.random() < 0.1: c1 = self.mutate(c1)
                if random.random() < 0.1: c2 = self.mutate(c2)
                offspring_pop.extend([c1, c2])

            # Đánh giá con cái
            for child in offspring_pop:
                if child.skill_factor == 0:
                    child.factorial_costs[0] = self.tasks.calculate_mdr_cost(child.genotype)
                else:
                    child.factorial_costs[1] = self.tasks.calculate_gcp_cost(child.genotype)

            self.population += offspring_pop
            self.assign_scalar_fitness()
            self.population.sort(key=lambda x: x.scalar_fitness, reverse=True)
            self.population = self.population[:self.pop_size]

            # Ghi nhận lịch sử
            best_mdr = min([ind.factorial_costs[0] for ind in self.population if ind.factorial_costs[0] != float('inf')])
            best_gcp = min([ind.factorial_costs[1] for ind in self.population if ind.factorial_costs[1] != float('inf')])
            history_mdr.append(best_mdr)
            history_gcp.append(best_gcp)
            
            if gen % 10 == 0:
                print(f"Gen {gen}: Best MDR Weight = {best_mdr}, Best Colors = {best_gcp}")

        # 1. Lấy cá thể tốt nhất của task MDR
        best_mdr_ind = min([ind for ind in self.population if ind.skill_factor == 0], 
                           key=lambda x: x.factorial_costs[0])
        
        # 2. Giải mã và Sửa lỗi (Repair)
        raw_sol = self.tasks.decode_mdr(best_mdr_ind.genotype)
        repaired_sol = self.tasks.repair_mdr_smart(raw_sol)
        
        # 3. [QUAN TRỌNG] Chạy Tối ưu hóa (Reduce) tại đây!
        # Chỉ chạy 1 lần duy nhất trên kết quả tốt nhất -> Không ảnh hưởng tốc độ training
        #print(">>> Đang thực hiện tối ưu hóa hậu kỳ (Reduce Solution)...")
        final_solution = self.tasks.optimize_mdr_fast(repaired_sol)
        #final_solution = repaired_sol
        final_weight = sum(final_solution.values())
        
        return final_solution, final_weight, history_mdr, history_gcp

    def mutate(self, p):
        child = Individual(self.dim)
        child.skill_factor = p.skill_factor
        child.genotype = p.genotype.copy()
        mask = np.random.rand(self.dim) < 0.1
        noise = np.random.normal(0, 0.2, self.dim)
        child.genotype[mask] += noise[mask]
        child.genotype = np.clip(child.genotype, 0, 1)
        return child

    def assign_scalar_fitness(self):
        pop_mdr = [ind for ind in self.population if ind.factorial_costs[0] != float('inf')]
        pop_gcp = [ind for ind in self.population if ind.factorial_costs[1] != float('inf')]
        pop_mdr.sort(key=lambda x: x.factorial_costs[0])
        pop_gcp.sort(key=lambda x: x.factorial_costs[1])
        
        for rank, ind in enumerate(pop_mdr): ind.scalar_fitness = 1.0 / (rank + 1)
        for rank, ind in enumerate(pop_gcp): ind.scalar_fitness = 1.0 / (rank + 1)
        for ind in self.population:
            if ind.scalar_fitness == float('inf'): ind.scalar_fitness = 0

# ==============================================================================
# PHẦN 3: MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    mtx_file = "ash85.mtx" # Đảm bảo file này nằm cùng thư mục
    
    # Kiểm tra file tồn tại
    if not os.path.exists(mtx_file):
        print(f"Không tìm thấy file {mtx_file}. Hãy tải file ash85.mtx về trước.")
    else:
        G = load_mtx_graph(mtx_file)
        
        if G is not None:
            tasks = MFEA_Tasks(G)
            # Tăng rmp lên 0.6 để tận dụng tối đa chuyển giao tri thức từ bài toán tô màu
            mfea = MFEA(tasks, pop_size=100, generations=200, rmp=0.5)
            
            print("\n>>> Bắt đầu chạy MFEA (Optimized)...")
            final_sol, final_weight, hist_mdr, hist_gcp = mfea.run()
            
            print("-" * 30)
            print(f"KẾT QUẢ TỐI ƯU TRÊN FILE {mtx_file}:")
            print(f"Double Roman Weight: {final_weight}")
            
            # Vẽ biểu đồ
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(hist_mdr, color='red')
            plt.title("MDR Weight Convergence")
            plt.xlabel("Generation")
            plt.ylabel("Weight")
            
            plt.subplot(1, 2, 2)
            plt.plot(hist_gcp, color='blue')
            plt.title("GCP Colors Convergence")
            plt.xlabel("Generation")
            plt.ylabel("Num Colors")
            plt.tight_layout()
            plt.show()