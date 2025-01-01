# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np

from src.rotation.solver import AbstractSolver
from src.rotation.linear import LinearRelaxationSolver
from src.rotation.gnc import GncSolver
from src.rotation.utils import rot_to_vec, vec_to_rot
from src.utils import project, rotationSVD


class IrlsSvdSolver(AbstractSolver):
    def __init__(self, max_iteration=1000, tolerance=1e-6, c=1, robust_type="GM"):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Singular Value Decomposition
        rot = rotationSVD(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        for _ in range(self.max_iter):
            # weight update
            vec_w = self.compute_weights(res, self.c, self.robust_type)
            # variable update
            rot = rotationSVD(pcd1, pcd2, weight=vec_w)
            res = self.compute_residuals(pcd1, pcd2, rot, noise_bound)
            # stopping criteria
            cost = np.sum(res * res)
            if self.check_cost_convergence(cost, prev_cost):
                break
            prev_cost = cost

        return rot


class IrlsLinearSolver(LinearRelaxationSolver, AbstractSolver):
    def __init__(self, max_iteration=1000, tolerance=1e-6, c=1, robust_type="GM"):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Singular Value Decomposition
        rot = rotationSVD(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        terms = self.compute_terms(pcd1, pcd2, noise_bound)
        x = rot_to_vec(rot)

        for _ in range(self.max_iter):
            # weight update
            mat_w, vec_w = self.compute_weighted_term(
                terms, x, self.c, self.robust_type
            )

            # NOTE: In extreme outlier cases, TLS sometimes get zero matrix as the quadratic term.
            # This is because all the weights are zero, i.e., all the data are outliers.
            if sum(vec_w) == 0:
                break

            # variable update
            x = self.solve_quadratic_program(mat_w)
            # stopping criteria
            cost = x.T @ mat_w @ x
            if self.check_cost_convergence(cost, prev_cost):
                break
            prev_cost = cost

        rot = project(vec_to_rot(x))

        return rot


class GncSvdSolver(GncSolver, AbstractSolver):
    def __init__(
        self,
        max_iteration=1000,
        tolerance=1e-6,
        c=1,
        robust_type="GM",
        gnc_factor=1.4,
        weight_tolerace=1e-4,
    ):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type
        self.gnc_factor = gnc_factor
        self.weight_tol = weight_tolerace

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Singular Value Decomposition
        rot = rotationSVD(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        # initial surrogate parameter
        mu = self.compute_initial_mu(res, self.c, self.robust_type)

        for _ in range(self.max_iter):
            # weight update
            vec_w = self.compute_weights(res, self.c, self.robust_type, mu)
            # variable update
            rot = rotationSVD(pcd1, pcd2, weight=vec_w)
            res = self.compute_residuals(pcd1, pcd2, rot, noise_bound)
            # stopping criteria
            cost = np.sum(res * res)
            if (
                self.check_cost_convergence(cost, prev_cost)
                or self.check_mu_convergence(mu, self.robust_type)
                or self.check_weight_convergence(vec_w, self.robust_type)
            ):
                break
            prev_cost = cost
            # surrogate update
            mu = self.update_mu(mu, self.robust_type, self.gnc_factor)

        return rot


class GncLinearSolver(GncSolver, LinearRelaxationSolver, AbstractSolver):
    def __init__(
        self,
        max_iteration=1000,
        tolerance=1e-6,
        c=1,
        robust_type="GM",
        gnc_factor=1.4,
        weight_tolerace=1e-4,
    ):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type
        self.gnc_factor = gnc_factor
        self.weight_tol = weight_tolerace

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot = rotationSVD(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        terms = self.compute_terms(pcd1, pcd2, noise_bound)
        x = rot_to_vec(rot)

        # initial surrogate parameter
        mu = self.compute_initial_mu(res, self.c, self.robust_type)

        for _ in range(self.max_iter):
            # weight update
            mat_w, vec_w = self.compute_weighted_term(
                terms, x, self.c, self.robust_type, mu
            )

            # NOTE: In extreme outlier cases, TLS sometimes get zero matrix as the quadratic term.
            # This is because all the weights are zero, i.e., all the data are outliers.
            if sum(vec_w) == 0:
                break

            # variable update
            x = self.solve_quadratic_program(mat_w)
            # stopping criteria
            cost = x.T @ mat_w @ x
            if (
                self.check_cost_convergence(cost, prev_cost)
                or self.check_mu_convergence(mu, self.robust_type)
                or self.check_weight_convergence(vec_w, self.robust_type)
            ):
                break
            prev_cost = cost
            # surrogate update
            mu = self.update_mu(mu, self.robust_type, self.gnc_factor)

        rot = project(vec_to_rot(x))

        return rot
