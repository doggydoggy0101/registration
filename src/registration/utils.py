# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np


def rot_and_t_to_vec(rot, t):
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
            t[0],
            t[1],
            t[2],
            1.0,
        ]
    )


def vec_to_rot_and_t(vec):
    rot = np.array(
        [[vec[0], vec[3], vec[6]], [vec[1], vec[4], vec[7]], [vec[2], vec[5], vec[8]]]
    )
    t = np.array([vec[9], vec[10], vec[11]])
    return rot, t


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
