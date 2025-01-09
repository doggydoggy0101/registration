# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np


# A structure that can be used to compute the quadratic form associated
# with a matrix, and keep track of the most recently computed value.
class R2Sym:
    def __init__(self, mat, cache=0.0):
        self.mat = mat
        self.cache = cache

    # Compute the quadratic form associated with self and x.
    def call(self, x):
        return x.T @ self.mat @ x

    # Update the cached value of the quadratic form associated with self.
    def update_cache(self, x):
        self.cache = self.call(x)


# A structure to represent a fractional term $f(x)/h(x)$ in the
# Geman-McClure-based objective function.
class Fractional:
    def __init__(self, r2, c2):
        self.r2 = r2
        self.c2 = c2

    # Updates the cache of the square of residual.
    def update_cache(self, x):
        self.r2.update_cache(x)

    # Computes the numerator $f(x)$.
    def f(self):
        return self.c2 * self.r2.cache

    # Computes the denominator $h(x)$.
    def h(self):
        return self.r2.cache + self.c2

    # Get the matrix associated with the numerator.
    def f_mat(self):
        return self.c2 * self.r2.mat

    # Get the matrix associated with the denominator.
    def h_mat(self):
        return self.r2.mat


class FracgmSolver:
    def compute_fractional_terms(self, pcd1, pcd2, noise_bound, c):
        """
        Compute structurized quadratic terms of each residual (FracGM) .

        Arguments

        - `pcd1`         - Source point cloud.
        - `pcd2`         - Target point cloud.
        - `noise_bound`  - Noise bound (sigma) of the residual.
        - `c`            - Threshold c.

        Returns

        - `terms` - List of structurized quadratic terms (FracGM).
        """
        terms = []
        id3 = np.eye(3)
        for i in range(pcd1.shape[0]):
            temp = np.zeros((3, 13))
            temp[:, :9] = np.kron(pcd1[i], id3)
            temp[:, 9:12] = id3
            temp[:, 12] = -pcd2[i]
            terms.append(
                Fractional(R2Sym((temp.T @ temp) / (noise_bound * noise_bound)), c * c)
            )
        return terms

    def compute_weighted_fractional_term(self, terms, beta, mu):
        """
        Compute a weighted quadratic term based on beta and mu (FracGM).

        Arguments

        - `terms` - List of structurized quadratic terms (FracGM).
        - `beta`  - Auxiliary parameter beta (FracGM).
        - `mu`    - Auxiliary parameter mu (FracGM).

        Returns

        - `mat` - Weighted quadratic term.
        """
        mat = np.zeros((13, 13))
        for i in range(len(terms)):
            mat += mu[i] * terms[i].f_mat() - mu[i] * beta[i] * terms[i].h_mat()
        return mat

    def update_auxiliary_variables(self, terms):
        """
        Update the auxiliary variables based on structurized quadratic terms (FracGM).

        Arguments

        - `terms` - List of structurized quadratic terms (FracGM).

        Returns

        - `beta` - Auxiliary parameter beta (FracGM).
        - `mu`   - Auxiliary parameter mu (FracGM).
        """
        beta = [term.f() / term.h() for term in terms]
        mu = [1 / term.h() for term in terms]

        return beta, mu

    # Update the cache of the structurized quadratic terms (FracGM).
    def update_fractional_terms_cache(self, terms, vec):
        for term in terms:
            term.update_cache(vec)

    # Compute the norm of psi function.
    def compute_psi_norm(self, terms, beta, mu):
        loss = 0.0
        for i, term in enumerate(terms):
            a = -term.f() + beta[i] * term.h()
            b = -1.0 + mu[i] * term.h()
            loss += a * a + b * b
        return np.sqrt(loss)
