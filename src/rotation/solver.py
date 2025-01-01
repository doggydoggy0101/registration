# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np
from src.utils import rotationSVD


class AbstractSolver:
    def __init__(self):
        """
        Robust rotation estimation solver.

        Arguments

        - `max_iter`    - Maximum number of iterations.
        - `tol`         - Cost tolerance for early stop.
        - `c`           - Threshold c in the robust function.
        - `robust_type` - Robust function: Truncated Least Squares (TLS) or Geman-McClure (GM).
        """
        self.max_iter = 1000
        self.tol = 1e-6
        self.c = 1
        self.robust_type = "GM"

    def solve(self, pcd1, pcd2, noise_bound):
        """
        Solve the robust rotation estimation problem.

        Arguments

        - `pcd1`        - Source point cloud.
        - `pcd2`        - Target point cloud.
        - `noise_bound` - Noise bound (sigma) of the residual.

        Returns
        - `rot` - Rotation matrix.
        """
        rot = rotationSVD(pcd1, pcd2)
        return rot

    def compute_residuals(self, pcd1, pcd2, rot, noise_bound):
        """
        Compute a residual vector based on rotation, translation, and noise bound.

        Arguments

        - `pcd1`        - Source point cloud.
        - `pcd2`        - Target point cloud.
        - `rot`         - Rotation matrix.
        - `noise_bound` - Noise bound (sigma) of the residual.

        Returns

        - Residual vector.
        """
        return np.linalg.norm(pcd2 - (rot @ pcd1.T).T, axis=1, ord=2) / noise_bound

    def compute_weights(self, res, c, robust_type):
        """
        Compute a weight vector based on residuals.

        Arguments

        - `res`         - Residuals of the previous iteration.
        - `c`           - Square of threshold c.
        - `robust_type` - Robust function: Truncated Least Squares (TLS) or Geman-McClure (GM).

        Returns

        - `vec` - Weight vector.
        """
        n = len(res)
        vec = np.zeros(n)
        if robust_type == "TLS":
            for i in range(n):
                if res[i] * res[i] <= c * c:
                    vec[i] = 1
        if robust_type == "GM":
            for i in range(n):
                vec[i] = 1 / pow(res[i] * res[i] + c * c, 2)
        return vec

    def compute_terms(self, pcd1, pcd2, noise_bound):
        """
        Compute the quadratic terms of each residual.

        Arguments

        - `pcd1`         - Source point cloud.
        - `pcd2`         - Target point cloud.
        - `noise_bound`  - Noise bound (sigma) of the residual.

        Returns

        - `terms` - List of quadratic terms.
        """
        terms = []
        id3 = np.eye(3)
        for i in range(pcd1.shape[0]):
            temp = np.zeros((3, 10))
            temp[:, :9] = np.kron(pcd1[i], id3)
            temp[:, 9] = -pcd2[i]
            terms.append((temp.T @ temp) / (noise_bound * noise_bound))
        return terms

    def compute_weighted_term(self, terms, x, c, robust_type):
        """
        Compute a weighted quadratic term based on x.

        Arguments

        - `term`        - List of orginal quadratic terms.
        - `x`           - Vectorized solution of the previous iteration.
        - `c`           - Square of threshold c.
        - `robust_type` - Robust function: Truncated Least Squares (TLS) or Geman-McClure (GM).

        Returns

        - `mat` - Weighted quadratic term.
        - `vec` - Weight vector.
        """
        mat = np.zeros((10, 10))
        vec = np.zeros(len(terms))
        if robust_type == "TLS":
            for i, mat_i in enumerate(terms):
                if x.T @ mat_i @ x <= c * c:
                    mat += mat_i
                    vec[i] = 1
        if robust_type == "GM":
            for i, mat_i in enumerate(terms):
                w_i = 1 / pow(x.T @ mat_i @ x + c * c, 2)
                mat += w_i * mat_i
                vec[i] = w_i
        return mat, vec

    # Check convergence of relative cost difference.
    def check_cost_convergence(self, prev_cost, curr_cost):
        costConverge = False
        if abs(curr_cost - prev_cost) / max(prev_cost, 1e-7) < self.tol:
            costConverge = True
        return costConverge
