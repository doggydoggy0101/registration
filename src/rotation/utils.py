# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np


def rot_to_vec(rot):
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
            1.0,
        ]
    )


def vec_to_rot(vec):
    rot = np.array(
        [[vec[0], vec[3], vec[6]], [vec[1], vec[4], vec[7]], [vec[2], vec[5], vec[8]]]
    )
    return rot
