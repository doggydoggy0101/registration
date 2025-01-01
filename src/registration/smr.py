# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np
import cvxpy as cp


class SmrSolver:
    def solve_stiefel_manifold(self, mat, initial):
        """
        Solve least squares by Stiefel manifold relaxation.

        Arguments

        - `mat`           - Quadratic term of the objective.
        - `x`             - Current solution vector x.

        Returns

        - Solution of the Stiefel manifold relaxation.
        """
        var = cp.Variable(13, value=initial)

        rot = cp.vstack(
            [
                cp.hstack([var[0], var[3], var[6]]),
                cp.hstack([var[1], var[4], var[7]]),
                cp.hstack([var[2], var[5], var[8]]),
            ]
        )

        i3 = cp.Parameter((3, 3))
        i3.value = np.eye(3)
        # semidefinite constraint
        sdc_mat = cp.vstack([cp.hstack([i3, rot]), cp.hstack([rot.T, i3])])

        constraints = [var[-1] == 1]
        constraints += [sdc_mat >> 0]

        prob = cp.Problem(cp.Minimize(cp.quad_form(var, mat)), constraints)
        prob.solve(solver=cp.MOSEK)

        return var.value
