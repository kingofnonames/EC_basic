def load_mtx_graph(file_path):
    try:
        with open(file_path, 'r') as f:
            # 1. Bỏ qua comments để tìm header
            header = ""
            for line in f:
                if not line.startswith('%'):
                    header = line
                    break
            
            if not header: return []

            # 2. Parse kích thước
            # Header format: Rows Cols Entries
            rows, cols, entries = map(int, header.split())
            
            # Khởi tạo List of Lists (Nhanh hơn Dict)
            # adj[0] chứa các đỉnh kề với đỉnh 0
            adj = [[] for _ in range(rows)]

            # 3. Đọc dữ liệu
            for line in f:
                parts = line.split()
                # Bỏ qua dòng lỗi hoặc thiếu dữ liệu
                if len(parts) < 2: continue 
                
                u, v = int(parts[0]), int(parts[1])

                # LOGIC LÀM SẠCH (giống NetworkX):
                # Loại bỏ khuyên (Self-loops: cạnh nối chính nó)
                if u == v: continue

                # Chuyển về 0-based index
                u -= 1
                v -= 1

                # Thêm cạnh vào danh sách
                adj[u].append(v)
                
                # Xử lý đối xứng (Symmetric)
                # Vì MTX symmetric thường chỉ lưu 1 chiều (tam giác dưới hoặc trên)
                adj[v].append(u)

        return adj

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return None
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        return None

G = load_mtx_graph('E:/EC_basic/data/DROMD/dwt__361.mtx')
gens = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3, 0, 0, 0, 0, 3, 0]
for i, val in enumerate(gens):
    check = False
    if val == 0:
        for neighbor in G[i]:
            count = 0
            if gens[neighbor] == 3:
                check = True
                break
            elif gens[neighbor] == 2:
                count += 1
                if count == 2:
                    check = True
                    break
        if not check:
            print(f'{i} - Lỗi 0')
            for k in G[i]:
                print(k, ' : ', gens[k])
            print(check)
            break
    elif val == 1:
        for neighbor in G[i]:
            if gens[neighbor] in [2, 3]:
                check = True
                break
        if not check:
            print(f'{i} - Lỗi 1')
            for k in G[i]:
                print(k, ' : ', gens[k])
            print(check)
            break
print(check)