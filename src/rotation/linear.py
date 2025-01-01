# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np
import scipy as sp


class LinearRelaxationSolver:
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
