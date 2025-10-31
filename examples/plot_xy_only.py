#!/usr/bin/env python3
"""Simple XY trajectory plotter with obstacle"""
import sys
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    """Read double integrator CSV: time, pos_x, pos_y, vel_x, vel_y, input_x, input_y"""
    ks, px, py = [], [], []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            ks.append(int(row[0]))
            px.append(float(row[1]))
            py.append(float(row[2]))
    return ks, px, py


def plot_xy_trajectory(csv_path, out_path):
    ks, px, py = read_csv(csv_path)

    # Obstacle params
    obs_cx, obs_cy, obs_r = -5.0, 0.0, 2.0

    # Calculate min distance
    min_dist = min(math.sqrt((x - obs_cx)**2 + (y - obs_cy)**2) for x, y in zip(px, py))
    violations = sum(1 for x, y in zip(px, py) if math.sqrt((x - obs_cx)**2 + (y - obs_cy)**2) < obs_r)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot trajectory
    ax.plot(px, py, 'o-', linewidth=2, markersize=5, label='Trajectory', color='blue')
    ax.plot(px[0], py[0], 'go', markersize=15, label='Start', zorder=10)
    ax.plot(px[-1], py[-1], 'r*', markersize=20, label='End', zorder=10)
    
    # Plot obstacle
    theta = np.linspace(0, 2*np.pi, 200)
    obs_x = obs_cx + obs_r * np.cos(theta)
    obs_y = obs_cy + obs_r * np.sin(theta)
    ax.fill(obs_x, obs_y, alpha=0.3, color='red', label='Obstacle')
    ax.plot(obs_x, obs_y, 'r-', linewidth=2)
    
    # Plot goal (assume [0,0])
    ax.plot(0, 0, 'ks', markersize=12, label='Goal', zorder=10)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'XY Trajectory\nMin dist={min_dist:.3f}m, Violations={violations}/{len(ks)}, Safe={violations==0}')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"✅ Saved XY plot to {out_path}")
    print(f"📊 Min distance: {min_dist:.3f}m")
    print(f"🛡️  Violations: {violations}/{len(ks)}")
    print(f"✅ Safe: {violations == 0}")


def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path('obstacle_avoidance_sdp_lifted_trajectory.csv')

    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        sys.exit(1)

    out_path = csv_path.with_suffix('')
    out_path = Path(str(out_path) + '_xy_plot.png')
    plot_xy_trajectory(csv_path, out_path)


if __name__ == '__main__':
    main()

