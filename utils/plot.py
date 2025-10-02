import matplotlib.pyplot as plt
import os
import numpy as np
def plot_tour(cities, tour, best_score, title=f"Best tour", filename=r"/tour.png"):
    plt.figure(figsize=(6, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c="red", s=40, label="cities")
    tour_coords = cities[tour]
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], "b-", linewidth=1.5, label="Path")
    plt.plot([tour_coords[-1, 0], tour_coords[0, 0]],
             [tour_coords[-1, 1], tour_coords[0, 1]], "b-", linewidth=1.5)
    for i, (x, y) in enumerate(cities):
        plt.text(x+0.5, y+0.5, str(i), fontsize=8, color="black")
    plt.title(title)
    plt.legend()
    plt.grid(False)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

def plot_scores(scores, title="Best score per generation", filename=r"/scores.png"):
    plt.figure(figsize=(6, 6))
    plt.plot(range(len(scores)), scores, "b-", linewidth=1.5, label="Best score")
    best_idx = np.argmin(scores)
    best_val = scores[best_idx]
    plt.scatter(best_idx, best_val, color="red", zorder=5)
    plt.text(best_idx, best_val, f"{best_val:.2f}", fontsize=8, color="red", ha="left", va="bottom")
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(False)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()