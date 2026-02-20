import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from numpy.linalg import norm
import pyvista as pv

def compute_q_criterion(points, velocities, k=20):
    """
    Compute Q-criterion using local velocity gradients estimated via least squares.
    """
    tree = cKDTree(points)
    N = points.shape[0]
    Q = np.zeros(N)

    for i in range(N):
        _, idx = tree.query(points[i], k=k+1)  # include self
        neighbor_points = points[idx[1:]]  # exclude self
        neighbor_vels = velocities[idx[1:]]

        # Fit linear model: v = A·x + b
        A = np.linalg.lstsq(neighbor_points - points[i], neighbor_vels - velocities[i], rcond=None)[0].T  # (3,3)

        # Velocity gradient tensor ∇v
        grad_v = A

        S = 0.5 * (grad_v + grad_v.T)
        Ω = 0.5 * (grad_v - grad_v.T)

        Q[i] = 0.5 * (np.sum(Ω**2) - np.sum(S**2))

    return Q

# -----------------------------
# 1. Load CSV Data
# -----------------------------
# CSV should contain: X, Y, Z, U, V, W columns
csv_path = "transient_results/velp_step30.csv"
output_vtp = "step30_output_with_q.vtp"
df = pd.read_csv(csv_path)

# Extract coordinates and velocity components
points = df[['x', 'y', 'z']].to_numpy()
points = points * 0.001
U = df['u'].to_numpy()
V = df['v'].to_numpy()
W = df['w'].to_numpy()
vel = np.stack([U, V, W], axis=1)

Q = compute_q_criterion(points, vel, k=20)

# Create VTK PolyData
pdata = pv.PolyData(points)
# pdata["velocity"] = vel
pdata["Q"] = Q

# Save as .vtp
pdata.save(output_vtp)
print(f"[✓] Saved VTP file with Q-criterion: {output_vtp}")
