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
