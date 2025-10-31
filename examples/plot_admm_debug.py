#!/usr/bin/env python3
import sys
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path):
    ks, px, py, vx, vy, ux, uy = [], [], [], [], [], [], []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            # k, px, py, vx, vy, ux, uy
            k = int(row[0])
            ks.append(k)
            px.append(float(row[1]))
            py.append(float(row[2]))
            vx.append(float(row[3]))
            vy.append(float(row[4]))
            ux.append(float(row[5]))
            uy.append(float(row[6]))
    return ks, px, py, vx, vy, ux, uy


def plot_all(csv_path, out_path):
    ks, px, py, vx, vy, ux, uy = read_csv(csv_path)

    # Obstacle params (match C++)
    obs_cx, obs_cy, obs_r = -5.0, 0.0, 2.0

    fig = plt.figure(figsize=(10, 12))

    # 1) XY trajectory with obstacle
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(px, py, 'o-', label='trajectory')
    theta = [2*math.pi*i/200 for i in range(201)]
    obs_x = [obs_cx + obs_r*math.cos(t) for t in theta]
    obs_y = [obs_cy + obs_r*math.sin(t) for t in theta]
    ax1.fill(obs_x, obs_y, alpha=0.25, color='gray', label='obstacle')
    # Make sure obstacle is visible even when trajectory scale is large
    xmin, xmax = min(px + [obs_cx - obs_r]) , max(px + [obs_cx + obs_r])
    ymin, ymax = min(py + [obs_cy - obs_r]) , max(py + [obs_cy + obs_r])
    dx, dy = xmax - xmin, ymax - ymin
    pad = 0.05 * max(dx, dy)
    ax1.set_xlim(xmin - pad, xmax + pad)
    ax1.set_ylim(ymin - pad, ymax + pad)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('XY Trajectory with Obstacle')
    ax1.set_xlabel('px')
    ax1.set_ylabel('py')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2) States over time
    ax2 = fig.add_subplot(3, 1, 2)
    # Match Julia-style labeling for states
    ax2.plot(ks, px, label='x₁ (position x)')
    ax2.plot(ks, py, label='x₂ (position y)')
    ax2.plot(ks, vx, label='x₃ (velocity x)')
    ax2.plot(ks, vy, label='x₄ (velocity y)')
    ax2.set_title('States (x)')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('State Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=4)

    # 3) Inputs over time (length N-1, but last row has zeros)
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.step(ks, ux, where='post', label='u₁ (acceleration x)')
    ax3.step(ks, uy, where='post', label='u₂ (acceleration y)')
    ax3.set_title('Controls (u)')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Control Value')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    # Stick to the rollout trajectory CSV by default, matching the prior behavior
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path('obstacle_avoidance_sdp_lifted_trajectory.csv')

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    out_path = csv_path.with_suffix('')
    out_path = Path(str(out_path) + '_plots.png')
    plot_all(csv_path, out_path)


if __name__ == '__main__':
    main()
