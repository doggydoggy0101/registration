# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np


class GncSolver:
    def __init__(self):
        """
        Graduated non-convexity solver.

        Arguments

        - `gnc_factor` - Surrogate parameter's update step size.
        - `weight_tol` - Stopping critera for weights being binary.
        """
        self.gnc_factor = 1.4
        self.weight_tol = 1e-4

    def compute_initial_mu(self, res, c, robust_type):
        """
        Compute the initial value for surrogate paramter mu.

        Arguments

        - `res`         - Residuals of the previous iteration.
        - `c`           - Square of threshold c.
        - `robust_type` - Robust function: Truncated Least Squares (TLS) or Geman-McClure (GM).

        Returns

        - `mu` - Surrogate parameter.
        """
        max_res = np.max(res)
        if robust_type == "TLS":
            mu = (c * c) / (2 * max_res * max_res - c * c)
            # if the residual is very large, set threshold of 1e-6 to avoid mu = 0
            if mu >= 0 and mu < 1e-6:
                mu = 1e-6
        if robust_type == "GM":
            mu = (2 * max_res * max_res) / (c * c)
        return mu

    def update_mu(self, mu, robust_type, gnc_factor):
        """
        Update surrogate parameter to gradually increase nonconvexity.

        Arguments & Returns

        - `mu`          - Surrogate parameter.
        - `robust_type` - Robust function: Truncated Least Squares (TLS) or Geman-McClure (GM).

        Returns

        - `mu` - Surrogate parameter.
        """
        if robust_type == "TLS":
            mu *= gnc_factor
        if robust_type == "GM":
            mu /= gnc_factor
            mu = max(1.0, mu)  # saturate at 1
        return mu

    def compute_weights(self, res, c, robust_type, mu):
        """
        Compute a weight vector based on residuals and surrogate parameter mu.

        Arguments

        - `res`         - Residuals of the previous iteration.
        - `c`           - Square of threshold c.
        - `robust_type` - Robust function: Truncated Least Squares (TLS) or Geman-McClure (GM).
        - `mu`          - Surrogate parameter mu.

        Returns

        - `vec` - Weight vector.
        """
        n = len(res)
        vec = np.zeros(n)
        if robust_type == "TLS":
            for i in range(n):
                if res[i] * res[i] <= (mu / (mu + 1)) * c * c:
                    vec[i] = 1
                elif res[i] * res[i] <= ((mu + 1) / mu) * c * c:
                    vec[i] = (c / res[i]) * np.sqrt(mu * (mu + 1)) - mu
        if robust_type == "GM":
            for i in range(n):
                vec[i] = pow((mu * c * c) / (res[i] * res[i] + mu * c * c), 2)
        return vec

    def compute_weighted_term(self, terms, x, c, robust_type, mu):
        """
        Compute a weighted quadratic term based on x and surrogate parameter mu.

        Arguments

        - `term`        - List of orginal quadratic terms.
        - `x`           - Vectorized solution of the previous iteration.
        - `c`           - Square of threshold c.
        - `robust_type` - Robust function: Truncated Least Squares (TLS) or Geman-McClure (GM).

        Returns

        - `mat` - Weighted quadratic term.
        """
        mat = np.zeros((13, 13))
        if robust_type == "TLS":
            for mat_i in terms:
                if x.T @ mat_i @ x <= (mu / (mu + 1)) * c * c:
                    mat += mat_i
                elif x.T @ mat_i @ x <= ((mu + 1) / mu) * c * c:
                    mat += mat_i * (c * np.sqrt(mu * (mu + 1) / (x.T @ mat_i @ x)) - mu)
        if robust_type == "GM":
            for mat_i in terms:
                mat += mat_i * pow((mu * c * c) / (x @ mat_i @ x + mu * c * c), 2)
        return mat

    def check_mu_convergence(self, mu, robust_type):
        # Check if surrogate function approximates the original robust function (for GM).
        muConverge = False
        if robust_type == "GM" and abs(mu - 1.0) < 1e-9:
            muConverge = True
        return muConverge

    def check_weight_convergence(self, weight, robust_type):
        # Check convergence of weights to binary values (for TLS).
        weightConverge = False
        if robust_type == "TLS":
            weightConverge = True
            for w_i in weight:
                if abs(w_i - round(w_i)) > self.weight_tol:
                    weightConverge = False
                    break
        return weightConverge
