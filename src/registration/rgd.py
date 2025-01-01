# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np

from src.line_search import BackTrackingLineSearch
from src.utils import project


class RgdSolver:
    def solve_riemannian_gradient_descent(
        self, mat, x, max_iteration=1000, min_grad_norm=1e-6, min_step_size=1e-10
    ):
        """
        Solve least squares by Riemannian gradient descent.

        Arguments

        - `mat`           - Quadratic term of the objective.
        - `x`             - Current solution vector x.
        - `max_iteration` - Maximum number of iterations.
        - `min_grad_norm` - Gradient norm tolerance for early stop.
        - `min_step_size` - Step size tolerance for early stop.

        Returns

        - Solution of the Riemannian gradient descent.
        """

        def objective(x):
            return x.T @ mat @ x

        line_search = BackTrackingLineSearch()

        for i in range(max_iteration):
            gradEuclidean = 2 * mat @ x
            gradRiemannian = se3_projection(
                gradEuclidean, x
            )  # projection to the tangent space of se3_x
            grad_norm = se3_norm(gradRiemannian)

            x, step_size = line_search.search(
                objective,
                se3_retraction,
                se3_norm,
                x,
                -gradRiemannian,
                objective(x),
                -(grad_norm**2),
            )

            if grad_norm < min_grad_norm or step_size < min_step_size:
                break
        return x


# Projection of SE(3).
def se3_projection(vec, x):
    w_x = (
        x[6] * vec[3]
        + x[7] * vec[4]
        + x[8] * vec[5]
        - x[3] * vec[6]
        - x[4] * vec[7]
        - x[5] * vec[8]
    )
    w_y = (
        x[0] * vec[6]
        + x[1] * vec[7]
        + x[2] * vec[8]
        - x[6] * vec[0]
        - x[7] * vec[1]
        - x[8] * vec[2]
    )
    w_z = (
        x[3] * vec[0]
        + x[4] * vec[1]
        + x[5] * vec[2]
        - x[0] * vec[3]
        - x[1] * vec[4]
        - x[2] * vec[5]
    )

    return np.array(
        [
            x[3] * w_z - x[6] * w_y,
            x[4] * w_z - x[7] * w_y,
            x[5] * w_z - x[8] * w_y,
            x[6] * w_x - x[0] * w_z,
            x[7] * w_x - x[1] * w_z,
            x[8] * w_x - x[2] * w_z,
            x[0] * w_y - x[3] * w_x,
            x[1] * w_y - x[4] * w_x,
            x[2] * w_y - x[5] * w_x,
            vec[9],
            vec[10],
            vec[11],
            0.0,
        ]
    )


# Retraction of SE(3).
def se3_retraction(vec, x):
    rot = project(
        np.array(
            [
                [x[0] + vec[0], x[3] + vec[3], x[6] + vec[6]],
                [x[1] + vec[1], x[4] + vec[4], x[7] + vec[7]],
                [x[2] + vec[2], x[5] + vec[5], x[8] + vec[8]],
            ]
        )
    )

    return np.array(
        [
            rot[0, 0],
            rot[1, 0],
            rot[2, 0],
            rot[0, 1],
            rot[1, 1],
            rot[2, 1],
            rot[0, 2],
            rot[1, 2],
            rot[2, 2],
            x[9] + vec[9],
            x[10] + vec[10],
            x[11] + vec[11],
            1,
        ]
    )


def se3_norm(vec):
    return np.linalg.norm(
        np.array(
            [
                [vec[0], vec[3], vec[6]],
                [vec[1], vec[4], vec[7]],
                [vec[2], vec[5], vec[8]],
            ]
        )
    ) + np.linalg.norm(vec[9:12])
