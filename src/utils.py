# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np


def get_zero_mean_point_cloud(pcd):
    """
    Gets the zero mean point cloud.

    Arguments

    - `pcd` - Input point cloud.

    Returns

    - `zm_pcd` - Point cloud with zero mean.
    - `mean`   - Mean of the original point cloud.
    """
    mean = np.mean(pcd, axis=0)
    zm_pcd = pcd - mean

    return zm_pcd, mean


def project(mat):
    """
    Closed-form solution for the Orthogonal Procrustes problem.
    This function finds the nearest rotation of a given matrix.

    Arguments

    - `mat` - Input matrix.

    Returns

    - `rot` - Rotation matrix.
    """
    U, _, Vt = np.linalg.svd(mat)
    rot = U @ Vt

    # reflection case
    if np.linalg.det(rot) < 0:
        D = np.diag(np.array([1.0, 1.0, -1.0]))
        rot = U @ D @ Vt

    return rot


def rotationSVD(pcd1, pcd2, weight=None):
    """
    Closed-form solution for the Wahba problem/least sqaures of rotation estimation.

    Arguments

    - `pcd1`   - Source point cloud.
    - `pcd2`   - Target point cloud.
    - `weight` - Weights if solving a weighted least squares.

    Returns

    - `rot` - Rotation matrix.
    """

    zmpcd1, _ = get_zero_mean_point_cloud(pcd1)
    zmpcd2, _ = get_zero_mean_point_cloud(pcd2)

    if weight is None:
        rot = project(zmpcd2.T @ zmpcd1)
    else:
        rot = project(zmpcd2.T @ np.diag(weight) @ zmpcd1)

    return rot


def registrationHorn(pcd1, pcd2, weight=None):
    """
    Closed-form solution for registration by Horn.

    Arguments

    - `pcd1`   - Source point cloud.
    - `pcd2`   - Target point cloud.
    - `weight` - Weights if solving a weighted least squares.

    Returns

    - `rot` - Rotation matrix.
    - `t`   - Translation vector.
    """
    zmpcd1, mean1 = get_zero_mean_point_cloud(pcd1)
    zmpcd2, mean2 = get_zero_mean_point_cloud(pcd2)

    if weight is None:
        rot = project(zmpcd2.T @ zmpcd1)
    else:
        rot = project(zmpcd2.T @ np.diag(weight) @ zmpcd1)

    t = mean2 - rot @ mean1

    return rot, t
