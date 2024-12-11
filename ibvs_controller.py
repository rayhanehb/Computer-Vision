import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian


def ibvs_controller(K, pts_des, pts_obs, zs, gain):
    """
    A simple proportional controller for IBVS.

    Implementation of a simple proportional controller for image-based
    visual servoing. The error is the difference between the desired and
    observed image plane points. Note that the number of points, n, may
    be greater than three. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K       - 3x3 np.array, camera intrinsic calibration matrix.
    pts_des - 2xn np.array, desired (target) image plane points.
    pts_obs - 2xn np.array, observed (current) image plane points.
    zs      - nx0 np.array, points depth values (may be estimated).
    gain    - Controller gain (lambda).

    Returns:
    --------
    v  - 6x1 np.array, desired tx, ty, tz, wx, wy, wz camera velocities.
    """

    # Initialize the velocity and stacked Jacobian matrix
    v = np.zeros((6, 1))
    J = np.zeros((2 * pts_obs.shape[1], 6))

    # Fill the Jacobian matrix
    for i in range(pts_obs.shape[1]):
        pt = np.array([[pts_obs[0, i]], [pts_obs[1, i]]])  # Corrected indexing
        J_part = ibvs_jacobian(K, pt, zs[i])
        J[2*i:2*i+2, :] = J_part

    # reshape the desired and observed points fortran 
    pded = pts_des.reshape(-1, order='F')
    pobs = pts_obs.reshape(-1, order='F')
    error = pded - pobs

    # Controller with pseudo-inverse

    J_T = J.T
    J_inv = inv(J_T @ J) @ J_T  # Using pseudo-inverse for numerical stability
    v = gain * J_inv @ error.T
    v = v.reshape((6, 1))

    # Ensure the correct shape and type for v
    if not (isinstance(v, np.ndarray) and v.dtype == np.float64 and v.shape == (6, 1)):
        raise TypeError("Wrong type or size returned!")

    return v
