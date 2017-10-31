"""Functions for manipulating SO3 quantities (rigid body rotations).

--- OVERVIEW
The functions in this module are general purpose SO3 operations. These
functions can be used to abstract SO3 operations so that the user can
think in terms of simple vector math. For example, "minus" can be used
to find the difference between two orientations.

Functional and syntactic documentation is provided in the function docstrings.
In contrast to oritools, these functions are generally representation
independent. Acceptable SO3 representations are rotation matrices, affine
(homogeneous) matrices, quaternions, and rotation vectors (rotvecs).

A rotvec is the product of the minimum angle
of rotation (radians, 0 to pi) and axis of rotation
(unit 3-vector). The set of all rotvecs is the tangent space
to the SO3 manifold; it is a lie algebra. When expressed
as a skew-symmetric matrix, the matrix exponential of the
rotvec is equal to its associated rotation matrix.

--- USAGE
Inputted array quantities should be np ndarrays, but may also be lists or tuples.
Outputted array quantities will be np ndarrays.
Inputted angles must be in radians, any value.
Outputted angles will be in radians, between 0 and pi.
Quaternions are ordered as [w, xi, yj, zk] where w is the real part.

--- REFERENCE
https://en.wikipedia.org/wiki/Rotation_(mathematics)
https://en.wikipedia.org/wiki/Axis-angle_representation
http://www.euclideanspace.com/maths/geometry/rotations/
https://en.wikipedia.org/wiki/Versor
http://arxiv.org/pdf/1107.1119.pdf

--- AUTHOR
Jason Nezvadovitz

"""
################################################# IMPORTS

# standard
from __future__ import division
# 3rd party
import numpy as np
import numpy.linalg as npl
import transformations as trns

################################################# MAIN

def rep(x):
    """Returns a string naming the SO3 representation of the given
    array x by examining its shape and testing for orthonormality.
    If the quantity is not a rotation vector, unit quaternion, rotation
    matrix, or homogeneous transformation matrix, 'notSO3' is returned.

    Output formats: 'rotvec', 'quaternion', 'rotmat', 'tfmat', or 'notSO3'.

    Note that this function is also useful for checking if an SO3 quantity
    is remaining on SO3 during operations that accrue error.

    >>> M = trns.random_rotation_matrix()
    >>> print(rep(M))
    tfmat
    >>> print(rep(M[:3, :3]))
    rotmat
    >>> M[-1, -1] = 2  # breaks homogeneity
    >>> print(rep(M))
    notSO3
    >>> print(rep([5, 1, 0]))  # arbitrary vector
    notSO3
    >>> print(rep([0, np.pi/2, 0]))
    rotvec

    """
    x = np.array(x)
    shape = x.shape
    if shape == (3, ):
        if (npl.norm(x) >= 0) and (npl.norm(x) <= np.pi):
            return 'rotvec'  # norm of a proper rotvec is always between 0 and pi (inclusive)
    elif shape == (4, ):
        if np.isclose(npl.norm(x), 1):
            return 'quaternion'  # here, quaternion refers specifically to unit quaternions (versors)
    elif shape == (3, 3):
        if np.isclose(npl.det(x), 1) and np.allclose(x.dot(x.T), np.eye(3)):
            return 'rotmat'  # a proper orthonormal matrix must have a determinant of +1 and an inverse equal to its transpose
    elif shape == (4, 4):
        if np.isclose(npl.det(x[:3, :3]), 1) and np.allclose(x[:3, :3].dot(x[:3, :3].T), np.eye(3)) and np.allclose(x[3, :], [0, 0, 0, 1]):
            return 'tfmat'  # both orthonormality and homogeneity (bottom row of [0, 0, 0, 1]) are required
    return 'notSO3'


def apply_to_points(x, P, t=None):
    """Applies the rotation given by the SO3 quantity x to a set of
    points P, and returns the newly transformed set of points. The points
    must be in 3-space and fitted into P as columns:

        [x1 x2 x3 ...]
    P = [y1 y2 y3 ...]
        [z1 z2 z3 ...]

    If t is set to a 3 element vector, then a translation by that specific vector
    will also be applied. Even if you pass in an affine transform (tfmat), only
    the rotation part will be carried out. However, if you intend to pass in
    a tfmat, do  apply_to_points(x, P, 'tfmat')  instead of apply_to_points(x, P, x[:3, 3])
    and the extraction of x[:3, 3] will be inherent.

    >>> P = [[1, 0, 1], 
    ...      [0, 1, 1], 
    ...      [0, 0, 1]]
    >>> r = [0, 0, np.pi/2]  # rotation of 90 deg about +z axis, rotvec form
    >>> P2 = apply_to_points(r, P)
    >>> print(np.round(P2, 2))
    [[ 0. -1. -1.]
     [ 1.  0.  1.]
     [ 0.  0.  1.]]
    >>> P3 = apply_to_points(r, P, [1, 2, 3])
    >>> print(np.round(P3, 2))
    [[ 1.  0.  0.]
     [ 3.  2.  3.]
     [ 3.  3.  4.]]
    >>> M = make_affine(matrix_from_rotvec(r), [1, 2, 3])
    >>> P4 = apply_to_points(M, P)
    >>> P5 = apply_to_points(M, P, 'tfmat')
    >>> print(np.allclose(P2, P4))
    True
    >>> print(np.allclose(P3, P5))
    True

    """
    # Check shape of P:
    P = np.array(P)
    shape = P.shape
    if shape[0] != 3:
        raise ValueError("P must be 3 by n.")
    # Check representation of x:
    xrep = rep(x)
    # Get rotation matrix if not rotmat:
    if xrep == 'rotvec':
        x = matrix_from_rotvec(x)
    elif xrep == 'quaternion':
        x = trns.quaternion_matrix(x)[:3, :3]
    elif xrep == 'tfmat':
        if t == 'tfmat':
            t = x[:3, 3]
        x = x[:3, :3]
    elif xrep != 'rotmat':
        raise ValueError("Quantity to apply is not on SO3.")
    # Apply rotation matrix to each point:
    newP = x.dot(P)
    # Apply translation as desired:
    if t is not None:
        # Allow the user to have used a 1D array for a single point:
        if shape == (3, ):
            newP = newP + t
        else:
            newP = newP + np.array([t]).T
    # Finally:
    return newP


def apply_to_matrix(x, A):
    """Applies the change of basis given by the SO3 quantity
    x to the 3by3 matrix M.

    To be clear, this performs:
    R * A * transpose(R)
    where R is the rotation matrix form of x.

    >>> A = np.random.rand(3, 3)
    >>> q = trns.random_quaternion()
    >>> R = trns.quaternion_matrix(q)
    >>> B1 = R[:3, :3].dot(A).dot(R[:3, :3].T)
    >>> B2 = apply_to_matrix(q, A)
    >>> print(np.allclose(B1, B2))
    True

    """
    A = np.array(A)
    if A.shape != (3, 3):
        raise ValueError("A must be 3 by 3.")
    xrep = rep(x)
    # Get rotation matrix if not rotmat:
    if xrep == 'rotvec':
        x = matrix_from_rotvec(x)
    elif xrep == 'quaternion':
        x = trns.quaternion_matrix(x)[:3, :3]
    elif xrep == 'tfmat':
        x = x[:3, :3]
    elif xrep != 'rotmat':
        raise ValueError("Quantity to apply is not on SO3.")
    # Apply change of basis to A:
    return x.dot(A).dot(x.T)


def plus(x1, x2):
    """Returns the SO3 quantity that represents first rotating
    by x1 and then by x2 (i.e. the composition of x2 on x1, call
    it "x1+x2"). The output will be in the same representation as the
    inputs, but the inputs must be in the same representation as each other.

    >>> R1 = trns.random_rotation_matrix()[:3, :3]
    >>> R2 = trns.random_rotation_matrix()[:3, :3]
    >>> R3 = R2.dot(R1)  # rotating vector v by R3 is R2*R1*v
    >>> R = plus(R1, R2)
    >>> print(np.allclose(R, R3))
    True
    >>> M1 = make_affine(R1, [4, 4, 4])
    >>> M2 = make_affine(R2, [5, 5, 5])
    >>> M3 = make_affine(R2.dot(R1), R2.dot([4, 4, 4]) + [5, 5, 5])
    >>> M = plus(M1, M2)
    >>> print(np.allclose(M, M3))
    True
    >>> print(np.allclose(M[:3, :3], R))
    True
    >>> q1 = trns.quaternion_from_matrix(R1)
    >>> q2 = trns.quaternion_from_matrix(R2)
    >>> q3 = trns.quaternion_multiply(q2, q1)
    >>> q = plus(q1, q2)
    >>> print(np.allclose(q, q3))
    True
    >>> print(np.allclose(trns.quaternion_matrix(q)[:3, :3], R))
    True
    >>> R1plusR2plusR3 = reduce(plus, [R1, R2, R3])  # using plus in succession
    >>> print(rep(R1plusR2plusR3))
    rotmat

    """
    xrep = rep(x1)
    xrep2 = rep(x2)
    if xrep != xrep2:
        raise ValueError('Values to "add" are not in the same SO3 representation.')
    if xrep == 'quaternion':
        return normalize(trns.quaternion_multiply(x2, x1))
    elif xrep in ['rotmat', 'tfmat']:
        return x2.dot(x1)
    elif xrep == 'rotvec':
        # Adding rotvecs is only valid for small perturbations...
        # Perform operation on actual SO3 manifold instead:
        x1 = matrix_from_rotvec(x1)
        x2 = matrix_from_rotvec(x2)
        return get_rotvec(x2.dot(x1))
    else:
        raise ValueError('Values to "add" are not on SO3.')


def minus(x1, x2):
    """Returns the SO3 quantity representing the minimal rotation from
    orientation x1 to orientation x2 (i.e. the inverse composition of x2 on x1, 
    call it "x2-x1"). The output will be in the same representation as the
    inputs, but the inputs must be in the same representation as each other.

    >>> A = trns.rotation_matrix(.2, [0, 1, 0])  # rotation of 0.2 rad about +y axis
    >>> B = trns.rotation_matrix(-.2, [0, 1, 0])  # rotation of -0.2 rad about +y axis
    >>> AtoB = minus(A, B)  # this is "B - A", the rotation from A to B
    >>> angle, axis = angle_axis(AtoB)  # should be a rotation of (-0.2)-(0.2) = -0.4 rad about +y axis
    >>> print(angle, axis)
    (0.39999999999999997, array([-0., -1., -0.]))
    >>> # The above is 0.4 rad about -y axis, which is equivalent to the expected -0.4 rad about the +y axis.
    >>> # As usual, the angle is always returned between 0 and pi, with the accordingly correct axis.
    >>> # The rotations need not be about the same axis of course:
    >>> NtoA = trns.random_quaternion()  # some rotation from frame N to frame A
    >>> NtoB = trns.random_quaternion()  # some rotation from frame N to frame B
    >>> AtoB = minus(NtoA, NtoB)  # We say "AtoB = NtoB - NtoA"
    >>> NtoAtoB = plus(NtoA, AtoB)  # NtoAtoB == NtoB and we say "NtoAtoB = NtoA + AtoB"
    >>> print(np.allclose(NtoAtoB, NtoB))
    True
    >>> # Evidently, plus and minus are inverse operations.
    >>> # "x1 + (x2 - x1) = x2"  reads as "(N to x1) plus (x1 to x2) = (N to x2)"
    >>> A[:3, 3] = trns.random_vector(3)
    >>> B[:3, 3] = trns.random_vector(3)
    >>> C1 = B.dot(npl.inv(A))
    >>> C2 = minus(A, B)
    >>> print(np.allclose(C1, C2))
    True

    """
    xrep = rep(x1)
    xrep2 = rep(x2)
    if xrep != xrep2:
        raise ValueError('Values to "subtract" are not in the same SO3 representation.')
    if xrep == 'quaternion':
        return normalize(trns.quaternion_multiply(x2, trns.quaternion_inverse(x1)))
    elif xrep == 'rotmat':
        return x2.dot(x1.T)
    elif xrep == 'tfmat':
        return x2.dot(npl.inv(x1))  # inverse of a tfmat is not its transpose
    elif xrep == 'rotvec':
        # Subtracting rotvecs is only valid for small perturbations...
        # Perform operation on actual SO3 manifold instead:
        x1 = matrix_from_rotvec(x1)
        x2 = matrix_from_rotvec(x2)
        return get_rotvec(x2.dot(x1.T))
    else:
        raise ValueError('Values to "subtract" are not on SO3.')


def error(current, target):
    """Returns a vector representing the difference in orientation between target
    and current. Picture the SO3 manifold as some surface, and now put two points
    on it; one called target and one called current. Draw a curve along the manifold
    from current to target. That curve is minus(current, target). At the curve's
    midpoint, draw the tangent vector to the curve. Make that tangent vector's
    magnitude equal to the arclength of the curve. This vector is the rotvec. If current
    and target are very close to each other, the rotvec becomes equivalent to the curve.
    However, even for long distances, it still tells you instantaneously which way to rotate
    to get from current to target, and tells you approximately by how much. As current
    gets closer to target, these approximations become exact, hence why rotvecs work.

    >>> current = trns.random_quaternion()
    >>> rotvec_current = get_rotvec(current)
    >>>
    >>> current_to_target_small = trns.quaternion_about_axis(0.001, [1, -2, 3])  # a small rotation of 0.001 rad about some axis
    >>> current_to_target_large = trns.quaternion_about_axis(3, [1, -2, 3])  # a large rotation of 3 rad about the same axis
    >>>
    >>> target_near = plus(current, current_to_target_small)  # current + small change = target_near
    >>> rotvec_target_near = get_rotvec(target_near)  # rotvec form of target_near
    >>>
    >>> target_far = plus(current, current_to_target_large)  # current + large change = target_far
    >>> rotvec_target_far = get_rotvec(target_far)  # rotvec form of target_far
    >>>
    >>> err_near = error(current, target_near)  # rotvec form of current_to_target_small
    >>> err_far = error(current, target_far)  # rotvec form of current_to_target_large
    >>>
    >>> # In the limit, the error becomes exactly the difference between the two rotvecs:
    >>> print(np.allclose(err_near, rotvec_target_near - rotvec_current, atol=1e-02))
    True
    >>> print(np.allclose(err_far, rotvec_target_far - rotvec_current, atol=1e-02))
    False
    >>> q1 = trns.random_quaternion()
    >>> q2 = trns.random_quaternion()
    >>> print(np.allclose(error(q1, q2), -error(q2, q1)))
    True

    """
    return get_rotvec(minus(current, target))


def slerp(x1, x2, fraction):
    """Spherical Linear intERPolation. Returns an SO3 quantity representing an orientation
    between x1 and x2. The fraction of the path from x1 to x2 is the fraction input, and it
    must be between 0 and 1. The output will be in the same representation as the inputs, but
    the inputs must be in the same representation as each other.

    >>> x1 = make_affine(trns.random_rotation_matrix()[:3, :3], [1, 1, 1])
    >>> x2 = make_affine(trns.random_rotation_matrix()[:3, :3], [2, 2, 2])
    >>> nogo = slerp(x1, x2, 0)
    >>> print(np.allclose(nogo, x1))
    True
    >>> allgo = slerp(x1, x2, 1)
    >>> print(np.allclose(allgo, x2))
    True
    >>> first_25 = slerp(x1, x2, 0.25)  # 0 to 25 percent from x1 to x2
    >>> last_75 = minus(first_25, x2)  # 25 to 75 percent from x1 to x2
    >>> wholeway = plus(first_25, last_75)
    >>> print(np.allclose(wholeway, x2))
    True
    >>> x1 = trns.quaternion_from_matrix(x1)
    >>> x2 = trns.quaternion_from_matrix(x2)
    >>> mine = slerp(x1, x2, 0.3)
    >>> his = trns.quaternion_slerp(x1, x2, 0.3)  # only works on quaternions
    >>> print(np.allclose(mine, his))
    True

    """
    xrep = rep(x1)
    xrep2 = rep(x2)
    if xrep != xrep2:
        raise ValueError('Values to slerp between are not in the same SO3 representation.')
    if xrep == 'quaternion':
        r12 = get_rotvec(trns.quaternion_multiply(x2, trns.quaternion_inverse(x1)))
        x12 = normalize(quaternion_from_rotvec(fraction * r12))
    elif xrep == 'rotmat':
        r12 = get_rotvec(x2.dot(x1.T))
        x12 = matrix_from_rotvec(fraction * r12)
    elif xrep == 'tfmat':
        M12 = x2.dot(npl.inv(x1))
        r12 = get_rotvec(M12)
        x12 = make_affine(matrix_from_rotvec(fraction * r12), fraction * M12[:3, 3])
    elif xrep == 'rotvec':
        R1 = matrix_from_rotvec(x1)
        R2 = matrix_from_rotvec(x2)
        x12 = fraction * get_rotvec(R2.dot(R1.T))
    else:
        raise ValueError("One or both values to slerp between are not on SO3.")
    return plus(x1, x12)


def get_rotvec(x):
    """Returns the rotvec equivalent to the given SO3 quantity x.

    >>> q = trns.random_quaternion()
    >>> r = get_rotvec(q)
    >>> M = trns.rotation_matrix(npl.norm(r), r/npl.norm(r))
    >>> print(trns.is_same_transform(M, trns.quaternion_matrix(q)))
    True

    """
    # sorry Mario, but the function you are looking for is in another castle
    angle, axis = angle_axis(x)
    return angle * axis


def angle_axis(x):
    """Returns the extracted angle and axis from a valid SO3 quantity x.
    This is equivalent to separating the rotvec form of x into its
    magnitude and direction. The angle will always be between 0 and pi, 
    inclusive, and the axis will always be a unit vector.

    >>> yourAngle, yourAxis = -1337, [-2, 0, 7]
    >>> yourMat = trns.rotation_matrix(yourAngle, yourAxis)
    >>> myAngle, myAxis = angle_axis(yourMat)
    >>> print(np.isclose(myAngle, np.mod(yourAngle, 2*np.pi)))
    True
    >>> print(np.allclose(myAxis, normalize(yourAxis)))
    True
    >>> myAngle2, myAxis2 = angle_axis(trns.quaternion_from_matrix(yourMat))
    >>> print(np.allclose(myAngle, myAngle2), np.allclose(myAxis, myAxis2))
    (True, True)

    """
    xrep = rep(x)
    # In the matrix case, transformations.py already has a nice implementation:
    if xrep in ['rotmat', 'tfmat']:
        if xrep == 'rotmat':
            x = make_affine(x)
        angle, axis, point = trns.rotation_from_matrix(x)
        # But to be consistent with rotvecs, we only use positive angles:
        if angle < 0:
            angle, axis = -angle, -axis
        # And we map the identity rotation to an axis of [0, 0, 0]:
        elif np.isclose(angle, 0):
            axis = np.array([0, 0, 0])
    # In the quaternion case, we carry out the following routine:
    elif xrep == 'quaternion':
        # Renormalize for accuracy:
        x = normalize(x)
        # If "amount of rotation" is negative, flip quaternion:
        if x[0] < 0:
            x = -x
        # Extract axis:
        axis = normalize(x[1:])
        # Extract angle:
        angle = 2*np.arccos(x[0])
    # In the rotvec case, the routine is trivial:
    elif xrep == 'rotvec':
        angle = npl.norm(x)
        if not np.isclose(angle, 0):
            axis = x / angle
        else:
            axis = np.array([0, 0, 0])
    # If not SO3:
    else:
        raise ValueError("Quantity to extract angle and axis from must be on SO3.")
    # Finally:
    return (angle, axis)


def quaternion_from_rotvec(r):
    """Returns the quaternion equivalent to the given rotvec r.

    >>> myRotvec = np.pi * np.array([0, 1, 0])
    >>> q = quaternion_from_rotvec(myRotvec)
    >>> M1 = trns.quaternion_matrix(q)
    >>> M2 = trns.rotation_matrix(np.pi, [0, 1, 0])
    >>> print(trns.is_same_transform(M1, M2))
    True

    """
    angle = np.mod(npl.norm(r), 2*np.pi)
    if not np.isclose(angle, 0):
        return trns.quaternion_about_axis(angle, r / angle)
    else:
        return np.array([1, 0, 0, 0])  # unit real number is identity quaternion


def matrix_from_rotvec(r):
    """Returns the rotation matrix equivalent to the given rotvec r.

    >>> angle, axis = 45*(np.pi/180), normalize([0, 0, 1])
    >>> myRotvec = angle*axis
    >>> myMat = matrix_from_rotvec(myRotvec)
    >>> print(np.round(myMat, decimals=3))
    [[ 0.707 -0.707  0.   ]
     [ 0.707  0.707  0.   ]
     [ 0.     0.     1.   ]]

    """
    angle = np.mod(npl.norm(r), 2*np.pi)
    if not np.isclose(angle, 0):
        rotvecmat = crossmat(r / angle)
        return np.eye(3) + (np.sin(angle) * rotvecmat) + ((1 - np.cos(angle)) * rotvecmat.dot(rotvecmat))
    else:
        return np.eye(3)


def get_a2b(a, b, rep_out='rotmat'):
    """Returns an SO3 quantity that will align vector a with vector b.
    The output will be in the representation selected by rep_out
    ('rotmat', 'quaternion', or 'rotvec').

    >>> a = trns.random_vector(3)
    >>> b = trns.random_vector(3)
    >>> R = get_a2b(a, b)
    >>> p1 = R.dot(a)
    >>> print(np.allclose(np.cross(p1, b), [0, 0, 0]))
    True
    >>> p1.dot(b) > 0
    True
    >>> q = get_a2b(a, b, 'quaternion')
    >>> p2 = apply_to_points(q, a)
    >>> print(np.allclose(np.cross(p2, b), [0, 0, 0]))
    True
    >>> r = get_a2b(a, b, 'rotvec')
    >>> p3 = apply_to_points(r, a)
    >>> print(np.allclose(np.cross(p3, b), [0, 0, 0]))
    True

    """
    a = normalize(a)
    b = normalize(b)
    cosine = a.dot(b)
    if np.isclose(abs(cosine), 1):
        R = np.eye(3)
    else:
        axis = np.cross(a, b)
        angle = np.arctan2(npl.norm(axis), cosine)
        R = trns.rotation_matrix(angle, axis)
    if rep_out == 'rotmat':
        return R[:3, :3]
    elif rep_out == 'quaternion':
        return trns.quaternion_from_matrix(R)
    elif rep_out == 'rotvec':
        return get_rotvec(R)
    else:
        raise ValueError("Invalid rep_out. Choose 'rotmat', 'quaternion', or 'rotvec'.")


def make_affine(x, t=None):
    """Returns the affine space form of a given array x. If x is
    a vector, it appends a new element with value 1 (creating a
    homogeneous coordinate). If x is a square matrix, it appends
    the row [0, 0, 0...] and the column [t, 1].T representing a
    translation of vector t. If t is None (default) then the
    translation is set to the zero vector.
    See: https://en.wikipedia.org/wiki/Affine_transformation#Augmented_matrix

    >>> v = np.array([-4, 5, 2])
    >>> print(make_affine(v))
    [-4  5  2  1]
    >>> R = np.array([[np.cos(1.5), -np.sin(1.5)], [np.sin(1.5), np.cos(1.5)]])
    >>> print(make_affine(R, [3, -4]))
    [[ 0.0707372  -0.99749499  3.        ]
     [ 0.99749499  0.0707372  -4.        ]
     [ 0.          0.          1.        ]]
    >>> M = trns.random_rotation_matrix()  # fyi this function creates a tfmat
    >>> print(np.allclose(M, make_affine(M[:3, :3])))
    True

    """
    x = np.array(x)
    shape = x.shape
    dim = len(shape)
    if dim == 1:
        return np.append(x, 1)
    elif dim == 2 and shape[0] == shape[1]:
        cast = np.zeros((shape[0] + 1, shape[1] + 1))
        cast[-1, -1] = 1
        cast[:shape[0], :shape[1]] = x
        if t is not None:
            cast[:-1, -1] = t
        return cast
    else:
        raise ValueError("Array to make affine must be a row or a square.")


def crossmat(v):
    """Returns the skew-symmetric matrix form of a three element vector.
    See: https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication

    >>> a = [1, 2, -3]
    >>> b = [4, -5, 6]
    >>> print(np.cross(a, b) == crossmat(a).dot(b))
    [ True  True  True]

    """
    return np.array([[ 0   , -v[2]  ,  v[1] ], 
                    [ v[2] ,   0    , -v[0] ], 
                    [-v[1] ,  v[0]  ,   0   ]])


def crossvec(W):
    """Returns the vector form of the given skew-symmetric matrix W.
    This is exactly the inverse operation of crossmat(v).

    >>> v = trns.random_vector(3)
    >>> u = crossvec(crossmat(v))
    >>> print(np.allclose(u, v))
    True

    """
    v = np.array([W[2, 1], W[0, 2], W[1, 0]])
    if not np.allclose(W, crossmat(v)):
        raise ValueError("Cross-product matrix must be skew-symmetric.")
    return v


def random_rotvec():
    """Returns a randomly generated valid rotvec.

    >>> r = random_rotvec()
    >>> print(rep(r))
    rotvec

    """
    angle = np.pi * np.random.rand(1)
    axis = normalize(2 * (np.random.rand(3) - 0.5))
    return angle * axis


def normalize(v):
    """Returns v unitized by its 2norm.
    (This is a simpler version of trns.unit_vector).

    >>> v0 = np.random.random(3)
    >>> v1 = normalize(v0)
    >>> np.allclose(v1, v0 / npl.norm(v0))
    True

    """
    mag = npl.norm(v)
    if np.isclose(mag, 0):
        return np.zeros_like(v)
    else:
        return np.array(v) / npl.norm(v)


# Module unit test:
if __name__ == "__main__":
    import doctest
    doctest.testmod()
