import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from pathlib import Path


results_folder = Path(__file__).parent / 'results'
files = files = list(results_folder.glob('*.json'))
save_dir = results_folder / 'graph'
save_dir.mkdir(parents=True, exist_ok=True)
for file in files:
    dataset_name = Path(file.stem).stem
    with open(file, 'r') as f:
        data = json.load(f)

    mean_mdr = np.array(data['mean_mdr'])
    var_mdr = np.array(data['var_mdr'])
    std_mdr = np.sqrt(var_mdr)
    final_mean_mdr = mean_mdr[-1]

    mean_gcp = np.array(data['mean_gcp'])
    var_gcp = np.array(data['var_gcp'])
    std_gcp = np.sqrt(var_gcp)
    final_mean_gcp = mean_gcp[-1]

    generations = np.arange(len(mean_mdr))

    fig, (ax_mdr, ax_gcp) = plt.subplots(1, 2, figsize=(14, 5))

    ax_mdr.plot(generations, mean_mdr, color='#CC4F1B', linewidth=2, label='Mean Best Fitness')
    ax_mdr.fill_between(generations, 
                        mean_mdr - std_mdr, 
                        mean_mdr + std_mdr, 
                        color='#FF9848', alpha=0.5, label='Standard Deviation')
    ax_mdr.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_mdr.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_mdr.set_title(f'MDR Aggregated Convergence: {dataset_name}\nMin best mean: {final_mean_mdr:.0f}')
    ax_mdr.set_xlabel('Generation')
    ax_mdr.set_ylabel('Best Fitness')
    ax_mdr.legend()
    ax_mdr.grid(True, linestyle='--', alpha=0.6)

    ax_gcp.plot(generations, mean_gcp, color='#CC4F1B', linewidth=2, label='Mean Best Fitness')
    ax_gcp.fill_between(generations, 
                        mean_gcp - std_gcp, 
                        mean_gcp + std_gcp, 
                        color='#FF9848', alpha=0.5, label='Standard Deviation')
    ax_gcp.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_gcp.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_gcp.set_title(f'GCP Aggregated Convergence: {dataset_name}\nMin best mean: {final_mean_gcp:.0f}')
    ax_gcp.set_xlabel('Generation')
    ax_gcp.set_ylabel('Best Fitness')
    ax_gcp.legend()
    ax_gcp.grid(True, linestyle='--', alpha=0.6)

    save_path = save_dir / f'{dataset_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()