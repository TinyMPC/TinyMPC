#!/usr/bin/env python3
import sys
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_quadrotor_csv(path):
    """Read quadrotor CSV: time, x, y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi, u1, u2, u3, u4"""
    data = {
        'k': [], 'x': [], 'y': [], 'z': [],
        'phi': [], 'theta': [], 'psi': [],
        'dx': [], 'dy': [], 'dz': [],
        'dphi': [], 'dtheta': [], 'dpsi': [],
        'u1': [], 'u2': [], 'u3': [], 'u4': []
    }
    
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            data['k'].append(int(row[0]))
            data['x'].append(float(row[1]))
            data['y'].append(float(row[2]))
            data['z'].append(float(row[3]))
            data['phi'].append(float(row[4]))
            data['theta'].append(float(row[5]))
            data['psi'].append(float(row[6]))
            data['dx'].append(float(row[7]))
            data['dy'].append(float(row[8]))
            data['dz'].append(float(row[9]))
            data['dphi'].append(float(row[10]))
            data['dtheta'].append(float(row[11]))
            data['dpsi'].append(float(row[12]))
            data['u1'].append(float(row[13]))
            data['u2'].append(float(row[14]))
            data['u3'].append(float(row[15]))
            data['u4'].append(float(row[16]))
    
    return data


def plot_quadrotor(csv_path, out_path):
    data = read_quadrotor_csv(csv_path)

    # Obstacle params (XY plane only)
    obs_cx, obs_cy, obs_r = -5.0, 0.0, 2.0

    fig = plt.figure(figsize=(12, 14))

    # 1) XY trajectory with obstacle (top view)
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(data['x'], data['y'], 'o-', label='XY trajectory', markersize=4)
    ax1.plot(data['x'][0], data['y'][0], 'go', markersize=10, label='Start')
    ax1.plot(data['x'][-1], data['y'][-1], 'r*', markersize=15, label='End')
    
    # Obstacle circle
    theta = np.linspace(0, 2*np.pi, 200)
    obs_x = obs_cx + obs_r * np.cos(theta)
    obs_y = obs_cy + obs_r * np.sin(theta)
    ax1.fill(obs_x, obs_y, alpha=0.3, color='red', label='Obstacle')
    ax1.plot(obs_x, obs_y, 'r-', linewidth=2)
    
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('XY Trajectory with Obstacle (Top View)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2) Position states over time
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(data['k'], data['x'], label='x', linewidth=2)
    ax2.plot(data['k'], data['y'], label='y', linewidth=2)
    ax2.plot(data['k'], data['z'], label='z', linewidth=2)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Target altitude')
    ax2.set_title('Position States')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position (m)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3) Velocity states over time
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(data['k'], data['dx'], label='dx', linewidth=2)
    ax3.plot(data['k'], data['dy'], label='dy', linewidth=2)
    ax3.plot(data['k'], data['dz'], label='dz', linewidth=2)
    ax3.set_title('Velocity States')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4) Control inputs over time
    ax4 = fig.add_subplot(4, 1, 4)
    ax4.step(data['k'], data['u1'], where='post', label='u₁', linewidth=2)
    ax4.step(data['k'], data['u2'], where='post', label='u₂', linewidth=2)
    ax4.step(data['k'], data['u3'], where='post', label='u₃', linewidth=2)
    ax4.step(data['k'], data['u4'], where='post', label='u₄', linewidth=2)
    ax4.axhline(y=0.4, color='r', linestyle='--', alpha=0.3, label='Bounds')
    ax4.axhline(y=-0.4, color='r', linestyle='--', alpha=0.3)
    ax4.set_title('Control Inputs')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Control Value')
    ax4.grid(True, alpha=0.3)
    ax4.legend(ncol=5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"✅ Saved plot to {out_path}")


def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path('quadrotor_obstacle_avoidance_sdp_lifted.csv')

    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        sys.exit(1)

    out_path = csv_path.with_suffix('')
    out_path = Path(str(out_path) + '_plots.png')
    plot_quadrotor(csv_path, out_path)


if __name__ == '__main__':
    main()

