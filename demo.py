# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np

from src.RegistrationSolver import (
    IrlsHornSolver,
    IrlsLinearSolver,
    GncHornSolver,
    GncLinearSolver,
    FracgmLinearSolver,
)


CLOUD_SRC_PATH = "data/registration/cloud_bin_0.txt"
CLOUD_DST_PATH = "data/registration/cloud_bin_1.txt"
GT_PATH = "data/registration/gt.txt"


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
tolerance = 1e-6


src_reg, dst_reg, gt_reg = get_toy_data()
print("Ground truth:")
print(gt_reg, end="\n\n")


solver = IrlsHornSolver(max_iteration, tolerance, c, robust_type="GM")
rot, t = solver.solve(src_reg, dst_reg, noise_bound)
print("IRLS-GM-Horn:")
print_solution(rot, t)

solver = IrlsLinearSolver(max_iteration, tolerance, c, robust_type="GM")
rot, t = solver.solve(src_reg, dst_reg, noise_bound)
print("IRLS-GM-Linear:")
print_solution(rot, t)

solver = GncHornSolver(max_iteration, tolerance, c, robust_type="GM")
rot, t = solver.solve(src_reg, dst_reg, noise_bound)
print("GNC-GM-Horn:")
print_solution(rot, t)

solver = GncLinearSolver(max_iteration, tolerance, c, robust_type="GM")
rot, t = solver.solve(src_reg, dst_reg, noise_bound)
print("GNC-GM-Linear:")
print_solution(rot, t)

solver = FracgmLinearSolver(max_iteration, tolerance, c)
rot, t = solver.solve(src_reg, dst_reg, noise_bound)
print("FracGM-Linear:")
print_solution(rot, t)
