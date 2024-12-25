# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np

from src.solver import IrlsHornSolver, IrlsLinearSolver, GncHornSolver, GncLinearSolver


CLOUD_SRC_PATH = "data/cloud_bin_0.txt"
CLOUD_DST_PATH = "data/cloud_bin_1.txt"
GT_PATH = "data/gt.txt"


def get_registration_test_data():
    src = np.loadtxt(CLOUD_SRC_PATH)
    dst = np.loadtxt(CLOUD_DST_PATH)

    gt = np.loadtxt(GT_PATH)

    return src, dst, gt


c = 1
noise_bound = 0.1
max_iteration = 1000
rel_tol = 1e-6

src_reg, dst_reg, gt_reg = get_registration_test_data()
print("Ground truth:")
print(gt_reg, end="\n\n")


solver = IrlsHornSolver(max_iteration, rel_tol, c, robust_type="TLS")
rot, t = solver.solve(src_reg, dst_reg, noise_bound)
est_reg = np.eye(4)
est_reg[:3, :3] = rot
est_reg[:3, 3] = t
print("IRLS-TLS-Horn:")
print(est_reg, end="\n\n")

solver = IrlsHornSolver(max_iteration, rel_tol, c, robust_type="GM")
rot, t = solver.solve(src_reg, dst_reg, noise_bound)
est_reg = np.eye(4)
est_reg[:3, :3] = rot
est_reg[:3, 3] = t
print("IRLS-GM-Horn:")
print(est_reg, end="\n\n")

solver = IrlsLinearSolver(max_iteration, rel_tol, c, robust_type="TLS")
rot, t = solver.solve(src_reg, dst_reg, noise_bound)
est_reg = np.eye(4)
est_reg[:3, :3] = rot
est_reg[:3, 3] = t
print("IRLS-TLS-Linear:")
print(est_reg, end="\n\n")

solver = IrlsLinearSolver(max_iteration, rel_tol, c, robust_type="GM")
rot, t = solver.solve(src_reg, dst_reg, noise_bound)
est_reg = np.eye(4)
est_reg[:3, :3] = rot
est_reg[:3, 3] = t
print("IRLS-GM-Linear:")
print(est_reg, end="\n\n")

solver = GncHornSolver(max_iteration, rel_tol, c, robust_type="TLS")
rot, t = solver.solve(src_reg, dst_reg, noise_bound)
est_reg = np.eye(4)
est_reg[:3, :3] = rot
est_reg[:3, 3] = t
print("GNC-TLS-Horn:")
print(est_reg, end="\n\n")

solver = GncHornSolver(max_iteration, rel_tol, c, robust_type="GM")
rot, t = solver.solve(src_reg, dst_reg, noise_bound)
est_reg = np.eye(4)
est_reg[:3, :3] = rot
est_reg[:3, 3] = t
print("GNC-GM-Horn:")
print(est_reg, end="\n\n")

solver = GncLinearSolver(max_iteration, rel_tol, c, robust_type="TLS")
rot, t = solver.solve(src_reg, dst_reg, noise_bound)
est_reg = np.eye(4)
est_reg[:3, :3] = rot
est_reg[:3, 3] = t
print("GNC-TLS-Linear:")
print(est_reg, end="\n\n")

solver = GncLinearSolver(max_iteration, rel_tol, c, robust_type="GM")
rot, t = solver.solve(src_reg, dst_reg, noise_bound)
est_reg = np.eye(4)
est_reg[:3, :3] = rot
est_reg[:3, 3] = t
print("GNC-GM-Linear:")
print(est_reg, end="\n\n")
