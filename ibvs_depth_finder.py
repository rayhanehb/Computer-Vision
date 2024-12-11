import numpy as np

def ibvs_depth_finder(K, pts_obs, pts_prev, v_cam):
    """
    Compute estimated 

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K        - 3x3 np.array, camera intrinsic calibration matrix.
    pts_obs  - 2xn np.array, observed (current) image plane points.
    pts_prev - 2xn np.array, observed (previous) image plane points.
    v_cam    - 6x1 np.array, camera velocity (last commmanded).

    Returns:
    --------
    zs_est - nx0 np.array, updated, estimated depth values for each point.
    """
    n = pts_obs.shape[1]
    f = K[0, 0]  # Focal length
    u0, v0 = K[0, 2], K[1, 2]  # Principal point offsets
    zs_est = np.zeros(n, dtype=np.float64)

    for i in range(n):
        # Compute temporal change in image plane coordinates
        u_dot = pts_obs[0, i] - pts_prev[0, i]
        v_dot = pts_obs[1, i] - pts_prev[1, i]
        u_v_dot = np.array([[u_dot], [v_dot]])

        # Extract observed pixel coordinates
        u, v = pts_obs[0, i], pts_obs[1, i]

        # Construct translation Jacobian (J_t)
        J_t = np.zeros((2, 3))
        J_t[0, 0] = -f
        J_t[1, 1] = -f
        J_t[0, 2] = u - u0
        J_t[1, 2] = v - v0

        # Construct rotational Jacobian (J_w)
        J_w = np.zeros((2, 3))
        J_w[0, 0] = (u - u0) * (v - v0) / f
        J_w[0, 1] = -(f**2 + (u - u0)**2) / f
        J_w[0, 2] = v - v0
        J_w[1, 0] = (f**2 + (v - v0)**2) / f
        J_w[1, 1] = -(u - u0) * (v - v0) / f
        J_w[1, 2] = -(u - u0)

        # Compute A and b for the least-squares estimation
        A = J_t @ v_cam[:3].reshape(3, 1)
        b = u_v_dot - J_w @ v_cam[3:6].reshape(3, 1)

        # Solve for 1/Z using least-squares
        theta = np.linalg.lstsq(A, b, rcond=None)[0]

        # Compute estimated depth
        zs_est[i] = 1.0 / theta[0]

    return zs_est
