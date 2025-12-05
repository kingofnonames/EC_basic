import os
import numpy as np
import matplotlib.pyplot as plt
import json
from utils.load_file import list_files
def plot_result(file_path):
    if not os.path.exists(file_path):
        print(f"File không tồn tại: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = ["mean_mdr", "var_mdr", "mean_gcp", "var_gcp", "best_gcp", "best_mdr"]
    for k in required:
        if k not in data:
            print(f"File JSON thiếu trường: {k}")
            return
    
    best_mdr = int(data['best_mdr'])
    mean_mdr = np.array(data["mean_mdr"])
    std_mdr  = np.sqrt(np.array(data["var_mdr"]))
    
    best_gcp = int(data['best_gcp'])
    mean_gcp = np.array(data["mean_gcp"])
    std_gcp  = np.sqrt(np.array(data["var_gcp"]))

    x = np.arange(len(mean_mdr))

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot([], [], marker="o", color="red", linestyle="none",
             label=f"Best MDR ({best_mdr:.4f})")

    axes[1].plot([], [], marker="o", color="red", linestyle="none",
             label=f"Best GCP ({best_gcp:.4f})")

    axes[0].plot(x, mean_mdr, linewidth=2, label="Mean MDR")
    axes[0].fill_between(x, mean_mdr - std_mdr, mean_mdr + std_mdr, alpha=0.3)


    

    axes[0].set_title(f"MDR: Mean ± Std ({base_name})")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("MDR Value")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    axes[1].plot(x, mean_gcp, linewidth=2, label="Mean GCP")
    axes[1].fill_between(x, mean_gcp - std_gcp, mean_gcp + std_gcp, alpha=0.3)



    axes[1].set_title(f"GCP: Mean ± Std ({base_name})")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("GCP Value")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()

    out_name = f"./images/{base_name}_plots.png"
    plt.savefig(out_name, dpi=300)
    plt.close()

    print(f"Đã lưu hình subplot → {out_name}")

if __name__ == "__main__":
    folder = './results'
    files = list_files(folder)
    for file in files:
        plot_result(file)