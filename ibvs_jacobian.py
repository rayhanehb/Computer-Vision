import numpy as np

def ibvs_jacobian(K, pt, z):
    """
    Determine the Jacobian for IBVS.

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K  - 3x3 np.array, camera intrinsic calibration matrix.
    pt - 2x1 np.array, image plane point. 
    z  - Scalar depth value (estimated).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian. The matrix must contain float64 values.
    """

    #--- FILL ME IN ---
    #initialize the Jacobian
    J = np.zeros((2, 6))
    #define f
    f = K[0, 0]
    #deifne u and v
    u = pt[0, 0]
    v = pt[1, 0]
    #compute the Jacobian as numpy array
    J[0, 0] = -f/z
    J[0, 1] = 0
    J[0, 2] = (u - K[0, 2])/z
    J[0, 3] = ((u - K[0, 2])*(v - K[1, 2]))/f
    J[0, 4] = -(f**2+(u - K[0, 2])**2)/f
    J[0, 5] = (v-K[1, 2])
    J[1, 0] = 0
    J[1, 1] = -f/z
    J[1, 2] = (v - K[1, 2])/z
    J[1, 3] = (f**2+(v-K[1, 2])**2)/f
    J[1, 4] = -(u - K[0, 2])*(v - K[1, 2])/f
    J[1, 5] = -(u-K[0, 2])

    
    
    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J