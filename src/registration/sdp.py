# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np
import cvxpy as cp

from src.registration.utils import sdp_constraints


class SdpSolver:
    def solve_semidefinite_program(self, mat, initial=None):
        """
        Solve SDP by CVXPY and MOSEK.

        Arguments

        - `mat`     - Quadratic term of the objective.
        - `initial` - Initial guess for the semidefinite program.

        Returns

        - Solution of the semidefinite program.
        """
        var = cp.Variable((13, 13), PSD=True, value=np.outer(initial, initial))
        constraints = [cp.trace(sdp_constraints(22) @ var) == 1]
        for i in range(21):
            constraints += [cp.trace(sdp_constraints(i + 1) @ var) == 0]

        prob = cp.Problem(cp.Minimize(cp.trace(mat @ var)), constraints)
        prob.solve(solver=cp.MOSEK)
        mat_z = var.value

        U, d, _ = np.linalg.svd(mat_z)
        x = U[:, np.where(d == np.max(d))[0]].reshape(
            13
        )  # eigenvector associated with the maximum eigenvalue
        x /= x[-1]  # de-homogeneize

        return x
