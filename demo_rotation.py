# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np

from src.RotationSolver import (
    IrlsSvdSolver,
    IrlsLinearSolver,
    GncSvdSolver,
    GncLinearSolver,
)


CLOUD_SRC_PATH = "data/rotation/cloud_src.txt"
CLOUD_DST_PATH = "data/rotation/cloud_dst.txt"
GT_PATH = "data/rotation/gt.txt"


def get_toy_data():
    src = np.loadtxt(CLOUD_SRC_PATH)
    dst = np.loadtxt(CLOUD_DST_PATH)
    gt = np.loadtxt(GT_PATH)
    return src, dst, gt


def print_solution(rot, t):
    sol = np.eye(4)
    sol[:3, :3] = rot
    sol[:3, 3] = t
    print(sol, end="\n\n")


c = 1
noise_bound = 0.1
max_iteration = 1000
rel_tol = 1e-6


src_reg, dst_reg, gt_reg = get_toy_data()
print("Ground truth:")
print(gt_reg, end="\n\n")

solver = IrlsSvdSolver(max_iteration, rel_tol, c, robust_type="TLS")
rot = solver.solve(src_reg, dst_reg, noise_bound)
print("IRLS-TLS-SVD:")
print(rot, end="\n\n")

solver = IrlsSvdSolver(max_iteration, rel_tol, c, robust_type="GM")
rot = solver.solve(src_reg, dst_reg, noise_bound)
print("IRLS-GM-SVD:")
print(rot, end="\n\n")

solver = IrlsLinearSolver(max_iteration, rel_tol, c, robust_type="TLS")
rot = solver.solve(src_reg, dst_reg, noise_bound)
print("IRLS-TLS-Linear:")
print(rot, end="\n\n")

solver = IrlsLinearSolver(max_iteration, rel_tol, c, robust_type="GM")
rot = solver.solve(src_reg, dst_reg, noise_bound)
print("IRLS-GM-Linear:")
print(rot, end="\n\n")

solver = GncSvdSolver(max_iteration, rel_tol, c, robust_type="TLS")
rot = solver.solve(src_reg, dst_reg, noise_bound)
print("GNC-TLS-SVD:")
print(rot, end="\n\n")

solver = GncSvdSolver(max_iteration, rel_tol, c, robust_type="GM")
rot = solver.solve(src_reg, dst_reg, noise_bound)
print("GNC-GM-SVD:")
print(rot, end="\n\n")

solver = GncLinearSolver(max_iteration, rel_tol, c, robust_type="TLS")
rot = solver.solve(src_reg, dst_reg, noise_bound)
print("GNC-TLS-Linear:")
print(rot, end="\n\n")

solver = GncLinearSolver(max_iteration, rel_tol, c, robust_type="GM")
rot = solver.solve(src_reg, dst_reg, noise_bound)
print("GNC-GM-Linear:")
print(rot, end="\n\n")
