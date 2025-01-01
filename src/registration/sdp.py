# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np
import cvxpy as cp


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


# SDP constraints of SO(3) in registration problems (from 1 to 22)
def sdp_constraints(idx):
    assert idx >= 1 and idx <= 22, "index must be 1-22"

    mat_a = np.zeros((13, 13))

    if idx == 1:
        for i in range(3):
            mat_a[i][i] = 1
        mat_a[-1, -1] = -1

    if idx == 2:
        for i in range(3):
            mat_a[i + 3][i + 3] = 1
        mat_a[-1, -1] = -1

    if idx == 3:
        for i in range(3):
            mat_a[i + 6][i + 6] = 1
        mat_a[-1, -1] = -1

    if idx == 4:
        for i in range(3):
            mat_a[i, i + 3] = 1 / 2
        mat_a += mat_a.T

    if idx == 5:
        for i in range(3):
            mat_a[i, i + 6] = 1 / 2
        mat_a += mat_a.T

    if idx == 6:
        for i in range(3):
            mat_a[i + 3, i + 6] = 1 / 2
        mat_a += mat_a.T

    if idx == 7:
        for i in range(3):
            mat_a[3 * i, 3 * i] = 1
        mat_a[-1, -1] = -1

    if idx == 8:
        for i in range(3):
            mat_a[3 * i + 1, 3 * i + 1] = 1
        mat_a[-1, -1] = -1

    if idx == 9:
        for i in range(3):
            mat_a[3 * i + 2, 3 * i + 2] = 1
        mat_a[-1, -1] = -1

    if idx == 10:
        for i in range(3):
            mat_a[3 * i, 3 * i + 1] = 1 / 2
        mat_a += mat_a.T

    if idx == 11:
        for i in range(3):
            mat_a[3 * i, 3 * i + 2] = 1 / 2
        mat_a += mat_a.T

    if idx == 12:
        for i in range(3):
            mat_a[3 * i + 1, 3 * i + 2] = 1 / 2
        mat_a += mat_a.T

    if idx == 13:
        mat_a[4, 8], mat_a[5, 7], mat_a[0, -1] = 1 / 2, -1 / 2, -1 / 2
        mat_a += mat_a.T

    if idx == 14:
        mat_a[5, 6], mat_a[3, 8], mat_a[1, -1] = 1 / 2, -1 / 2, -1 / 2
        mat_a += mat_a.T

    if idx == 15:
        mat_a[3, 7], mat_a[4, 6], mat_a[2, -1] = 1 / 2, -1 / 2, -1 / 2
        mat_a += mat_a.T

    if idx == 16:
        mat_a[2, 7], mat_a[1, 8], mat_a[3, -1] = 1 / 2, -1 / 2, -1 / 2
        mat_a += mat_a.T

    if idx == 17:
        mat_a[0, 8], mat_a[2, 6], mat_a[4, -1] = 1 / 2, -1 / 2, -1 / 2
        mat_a += mat_a.T

    if idx == 18:
        mat_a[1, 6], mat_a[0, 7], mat_a[5, -1] = 1 / 2, -1 / 2, -1 / 2
        mat_a += mat_a.T

    if idx == 19:
        mat_a[1, 5], mat_a[2, 4], mat_a[6, -1] = 1 / 2, -1 / 2, -1 / 2
        mat_a += mat_a.T

    if idx == 20:
        mat_a[2, 3], mat_a[0, 5], mat_a[7, -1] = 1 / 2, -1 / 2, -1 / 2
        mat_a += mat_a.T

    if idx == 21:
        mat_a[0, 4], mat_a[1, 3], mat_a[8, -1] = 1 / 2, -1 / 2, -1 / 2
        mat_a += mat_a.T

    if idx == 22:
        mat_a[-1, -1] = 1

    return mat_a
