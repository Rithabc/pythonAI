#!/usr/bin/env python3
# roomba_locator_visual.py
# Robust simulation + visualization of Bayesian search with beep/no-beep sensor.
# Saves prob_step_0.png, prob_step_1.png, ... in the current folder.

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import traceback

# ------- Parameters -------
N = 5                 # grid size
alpha = 0.5           # sensor exponent parameter
MAX_STEPS = 30        # safety cutoff
SAVE_DIR = "."        # where to save PNGs

# ------- Optional: create a sample maze similar to your C++ generateGrid()
# 0 = free, 1 = blocked
def sample_maze(N, seed=None):
    # carve a single winding path (like your example)
    if seed is not None:
        random.seed(seed)
    grid = np.ones((N, N), dtype=int)
    x, y = 0, 0
    grid[x, y] = 0
    # random walk until bottom-right
    attempts = 0
    while not (x == N-1 and y == N-1) and attempts < N*N*10:
        attempts += 1
        dir = random.choice([(0,1),(0,-1),(1,0),(-1,0)])
        nx, ny = x + dir[0], y + dir[1]
        if 0 <= nx < N and 0 <= ny < N:
            grid[nx, ny] = 0
            x, y = nx, ny
    # ensure at least a few free cells
    if np.sum(grid==0) < max(3, N):
        for _ in range(N):
            rx, ry = random.randrange(N), random.randrange(N)
            grid[rx, ry] = 0
    return grid

# ------- sensor model -------
def sensor_p_beep(locator, cell, alpha=0.5):
    manh = abs(locator[0]-cell[0]) + abs(locator[1]-cell[1])
    d = max(1, manh)
    return np.exp(-alpha*(d-1))

def sense_beep(locator, roomba, alpha=0.5, stochastic=True):
    """Simulate beep/no-beep from true roomba position.
       If stochastic==False returns deterministic thresholded (p>0.5) for debugging."""
    p = sensor_p_beep(locator, roomba, alpha)
    if stochastic:
        return random.random() < p
    else:
        return p > 0.5

# ------- visualization helper -------
def save_visual(prob, grid, locator, roomba, target, step, savedir=".", show_values=False):
    plt.figure(figsize=(5,5))
    # mask blocked cells (show as gray)
    masked = np.ma.masked_where(grid != 0, prob)  # hide blocked as mask
    cmap = plt.cm.get_cmap("YlOrRd")
    im = plt.imshow(masked, cmap=cmap, origin='upper', vmin=0, vmax=prob.max() if prob.max()>0 else 1.0)
    plt.colorbar(im, label="Probability (free cells only)")
    plt.title(f"Step {step}: Locator {locator}")
    # annotate blocked cells
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i,j] == 1:
                plt.text(j, i, "X", ha="center", va="center", color="black", fontsize=12)
    # draw locator and true roomba
    plt.scatter(locator[1], locator[0], marker='o', color='blue', s=120, label='Locator')
    plt.scatter(roomba[1], roomba[0], marker='x', color='green', s=120, label='Roomba (true)')
    # draw predicted target
    plt.scatter(target[1], target[0], marker='*', color='purple', s=150, label='Target (max prob)')
    if show_values:
        for (i,j), val in np.ndenumerate(prob):
            if grid[i,j] == 0:
                plt.text(j, i+0.25, f"{val:.2f}", ha='center', va='center', fontsize=8)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    filename = os.path.join(savedir, f"prob_step_{step}.png")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

# ------- utility: move one step toward target avoiding blocked cells -------
def step_toward(src, dest, grid):
    si,sj = src
    di,dj = dest
    # candidate moves (prefer vertical then horizontal)
    candidates = []
    if di > si: candidates.append((si+1, sj))
    if di < si: candidates.append((si-1, sj))
    if dj > sj: candidates.append((si, sj+1))
    if dj < sj: candidates.append((si, sj-1))
    # fallback neighbors (in case above ones blocked)
    for r,c in [(si+1,sj),(si-1,sj),(si,sj+1),(si,sj-1)]:
        candidates.append((r,c))
    # pick first valid
    for (r,c) in candidates:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r,c] == 0:
            return (r,c)
    return src  # stuck, no valid move

# ------- main simulation -------
def run_simulation(N=5, alpha=0.5, max_steps=30, seed=None):
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        grid = sample_maze(N, seed=seed)
        free_cells = list(zip(*np.where(grid==0)))
        if len(free_cells) == 0:
            raise RuntimeError("No free cells in the grid.")
        # place locator at first free cell (matching your C++)
        locator = free_cells[0]
        # place roomba randomly on a free cell but not on locator
        roomba = random.choice(free_cells)
        while roomba == locator and len(free_cells) > 1:
            roomba = random.choice(free_cells)
        # belief uniform over free cells
        prob = np.zeros((N,N), dtype=float)
        for (i,j) in free_cells:
            prob[i,j] = 1.0 / len(free_cells)

        print("Grid (0=free, X=blocked):")
        print(grid)
        print(f"Locator at {locator}, true Roomba at {roomba}")
        step = 0
        save_visual(prob, grid, locator, roomba, (np.argmax(prob)//N, np.argmax(prob)%N), step, SAVE_DIR)

        total_actions = 0
        while step < max_steps:
            step += 1
            # 1) sense (stochastic)
            beep = sense_beep(locator, roomba, alpha, stochastic=True)
            total_actions += 1  # count sensing
            print(f"\nStep {step}: Locator {locator}, beep={beep:.0f}")

            # 2) update belief: multiply by P(z|x) then normalize (only for free cells)
            new_prob = np.zeros_like(prob)
            for i in range(N):
                for j in range(N):
                    if grid[i,j] == 1: 
                        new_prob[i,j] = 0.0
                        continue
                    p = sensor_p_beep(locator, (i,j), alpha)
                    like = p if beep else (1.0 - p)
                    new_prob[i,j] = prob[i,j] * like

            Z = new_prob.sum()
            if Z <= 0:
                # numeric rescue: reinitialize uniform over free cells
                print("Warning: belief collapsed to zero. Reinitializing uniform belief over free cells.")
                for (i,j) in free_cells:
                    new_prob[i,j] = 1.0 / len(free_cells)
            else:
                new_prob /= Z
            prob = new_prob

            # 3) choose the cell of maximal posterior probability (ties: choose one)
            max_idx = np.argmax(prob)
            target = (max_idx // N, max_idx % N)

            # 4) command: move the bot at that target one step closer to locator
            # Here we simulate by moving the *true* roomba one step toward the locator
            # (If you model that command differently, change this.)
            new_roomba = step_toward(roomba, locator, grid)
            if new_roomba != roomba:
                roomba = new_roomba
                total_actions += 1  # count each move as an action
                print(f"Roomba moved to {roomba} (toward locator).")

            # 5) optionally move the locator toward target (policy choice).
            # Here we follow your earlier policy: locator moves toward highest-prob cell.
            new_locator = step_toward(locator, target, grid)
            if new_locator != locator:
                locator = new_locator
                total_actions += 1  # moving locator counts as a move action
                print(f"Locator moved to {locator} (toward target).")

            save_visual(prob, grid, locator, roomba, target, step, SAVE_DIR)

            # termination: locator and roomba same cell = found
            if locator == roomba:
                print(f"\nFOUND: Locator and Roomba at {locator}. Total actions (senses + moves) = {total_actions}. Steps = {step}")
                return {
                    "found": True,
                    "steps": step,
                    "total_actions": total_actions,
                    "locator": locator,
                    "roomba": roomba,
                    "prob": prob
                }

        print(f"\nStopped after max_steps={max_steps}. Total actions = {total_actions}. Roomba at {roomba}")
        return {
            "found": False,
            "steps": step,
            "total_actions": total_actions,
            "locator": locator,
            "roomba": roomba,
            "prob": prob
        }

    except Exception as e:
        print("Exception during simulation:")
        traceback.print_exc()
        return None

# ------- run main -------
if __name__ == "__main__":
    # optional: set seed for reproducibility
    result = run_simulation(N=5, alpha=0.5, max_steps=30, seed=7)
    if result is not None:
        print("\nResult summary:", result)
    else:
        print("Simulation failed.")
