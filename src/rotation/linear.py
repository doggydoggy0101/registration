# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np
import scipy as sp


class LinearRelaxationSolver:
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

    def solve_quadratic_program(self, mat):
        """
        Closed-form solution for the quadratic program:

            min  x.T @ mat @ x
            s.t. e.T @ x = 1.

        This has a closed-form solution:

            x = (inv(mat) @ e) / (e.T @ inv(mat) @ e)

        NOTE: This requires mat to be invertible/singular/full-rank.
        We use the pseudo-inverse if the matrix is non-singular.

        Arguments

        - `mat` - Quadratic term of the objective.

        Arguments

        - `mat` - Quadratic term of the objective.

        Returns

        - Solution of the quadratic program.
        """
        e = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        if np.linalg.matrix_rank(mat) == mat.shape[0]:
            lu, piv = sp.linalg.lu_factor(mat)
            temp = sp.linalg.lu_solve((lu, piv), e)
        else:
            mat_pseudo_inv = np.linalg.pinv(mat)
            temp = mat_pseudo_inv @ e
        return (1 / (e @ temp)) * temp
