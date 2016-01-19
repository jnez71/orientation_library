"""Functions to help with orientation state manipulation.

--- OVERVIEW
For orientations represented as unit quaternions,
these functions can be used to abstract quaternion
operations so that the user can think in terms of
simple vector math. For example, "minus" can be used
to find the difference between two orientations.

Functional and syntactic documentation is provided
in the function docstrings below. The associated script
demo_PDcontrol exemplifies using this module.

In contrast to so3tools, the functions in this module are
designed for speed and do not have the overhead required to be
representation independent. They are all quaternion based.

Conversions to and from rotation vectors (rotvecs)
are available. A rotvec is the product of the minimum angle
of rotation (radians, 0 to pi) and axis of rotation
(unit 3-vector). The set of all rotvecs is the tangent space
to the SO3 manifold; it is a lie algebra. When expressed
as a skew-symmetric matrix, the matrix exponential of the
rotvec is equal to its associated rotation matrix.

--- USAGE
Quaternions are ordered as [w,xi,yj,zk] where w is the real part.
Inputted array quantities should be numpy ndarrays, but may also be lists or tuples.
Outputted array quantities will be numpy ndarrays.
Inputted angles must be in radians, any value.
Outputted angles will be in radians, between 0 and pi.

--- REFERENCE
https://en.wikipedia.org/wiki/Rotation_(mathematics)
https://en.wikipedia.org/wiki/Axis-angle_representation
http://www.euclideanspace.com/maths/geometry/rotations/
https://en.wikipedia.org/wiki/Versor
http://arxiv.org/pdf/1107.1119.pdf

"""
################################################# IMPORTS

# standard
from __future__ import division
# 3rd party
import numpy as np
import numpy.linalg as npl
import transformations as trns

################################################# MAIN

def plus(q1, q2):
    """Returns the quaternion that represents first rotating
    by q1 and then by q2 (i.e. the composition of q2 on q1,
    call it "q1+q2").

    >>> q1 = trns.random_quaternion()
    >>> q2 = trns.random_quaternion()
    >>> q12 = plus(q1, q2)
    >>>
    >>> p1 = trns.random_vector(3)
    >>> p2 = qapply_points(q1, p1)
    >>> p3a = qapply_points(q2, p2)
    >>>
    >>> p3b = qapply_points(q12, p1)
    >>> print(np.allclose(p3a, p3b))
    True

    """
    return normalize(trns.quaternion_multiply(q2, q1))


def minus(q1, q2):
    """Returns a quaternion representing the minimal rotation from
    orientation q1 to orientation q2 (i.e. the inverse composition
    of q2 on q1, call it "q2-q1").

    >>> NtoA = trns.random_quaternion()  # some rotation from frame N to frame A
    >>> NtoB = trns.random_quaternion()  # some rotation from frame N to frame B
    >>> AtoB = minus(NtoA, NtoB)  # We say "AtoB = NtoB - NtoA"
    >>> NtoAtoB = plus(NtoA, AtoB)  # NtoAtoB == NtoB and we say "NtoAtoB = NtoA + AtoB"
    >>> print(np.allclose(NtoAtoB, NtoB))
    True
    >>> # Evidently, plus and minus are inverse operations.
    >>> # "q1 + (q2 - q1) = q2"  reads as "(N to q1) plus (q1 to q2) = (N to q2)"
    >>> q = plus(NtoA, minus(NtoA, NtoB))
    >>> print(np.allclose(q, NtoB))
    True

    """
    return normalize(trns.quaternion_multiply(q2, trns.quaternion_inverse(q1)))


def error(q_current, q_target):
    """Returns the rotation vector representing the difference in
    orientation between q_target and q_current, which are quaternions.

    Picture the SO3 manifold as some surface, and now put two points
    on it; one called target and one called current. Draw a curve along the manifold
    from current to target. That curve is minus(current, target). At the curve's
    midpoint, draw the tangent vector to the curve. Make that tangent vector's
    magnitude equal to the arclength of the curve. This is the rotation vector. It
    tells you instantaneously which way to rotate to get from current to target, and
    tells you approximately by how much. As current gets closer to target, the
    approximation become exact.

    >>> current = trns.random_quaternion()
    >>> r_current = rotvec_from_quaternion(current)
    >>>
    >>> current_to_target_small = trns.quaternion_about_axis(0.001, [1,-2,3])  # a small rotation of 0.001 rad about some axis
    >>> current_to_target_large = trns.quaternion_about_axis(3, [1,-2,3])  # a large rotation of 3 rad about the same axis
    >>>
    >>> target_near = plus(current, current_to_target_small)  # current + small change = target_near
    >>> r_target_near = rotvec_from_quaternion(target_near)  # rotvec form of target_near
    >>>
    >>> target_far = plus(current, current_to_target_large)  # current + large change = target_far
    >>> r_target_far = rotvec_from_quaternion(target_far)  # rotvec form of target_far
    >>>
    >>> err_near = error(current, target_near)  # rotvec form of current_to_target_small
    >>> err_far = error(current, target_far)  # rotvec form of current_to_target_large
    >>>
    >>> # In the limit, the error becomes exactly the difference between the two rotvecs:
    >>> print(np.allclose(err_near, r_target_near - r_current, atol=1e-02))
    True
    >>> print(np.allclose(err_far, r_target_far - r_current, atol=1e-02))
    False

    """
    return rotvec_from_quaternion(minus(q_current, q_target))


def qapply_points(q, P, t=None):
    """Applies the rotation given quaternion q to a set of
    points P, and returns the newly transformed set of points.
    The points must be in 3-space and fitted into P as columns:

        [x1 x2 x3 ...]
    P = [y1 y2 y3 ...]
        [z1 z2 z3 ...]

    If t is set to a 3 element vector, then a translation by that
    specific vector will also be applied.

    >>> P = [[1, 0, 1], 
    ...      [0, 1, 1], 
    ...      [0, 0, 1]]
    >>> q = quaternion_from_rotvec([0, 0, np.pi/2])  # rotation of 90 deg about +z axis
    >>> P2 = qapply_points(q, P)
    >>> print(np.round(P2, 2))
    [[ 0. -1. -1.]
     [ 1.  0.  1.]
     [ 0.  0.  1.]]
    >>> P3 = qapply_points(q, P, [1, 2, 3])
    >>> print(np.abs(np.round(P3, 2)))  # abs is because -0 = 0
    [[ 1.  0.  0.]
     [ 3.  2.  3.]
     [ 3.  3.  4.]]

    """
    R = trns.quaternion_matrix(q)[:3, :3]
    P = R.dot(P)
    if t is not None:
        if P.shape == (3, ):
            P = P + t
        else:
            P = P + np.array([t]).T
    return P


def qapply_matrix(q, A):
    """Applies the change of basis given by the quaternion
    q to the 3by3 matrix A.

    Performs
    R * A * transpose(R)
    where R is the rotation matrix form of q.

    >>> A = np.random.rand(3, 3)
    >>> q = trns.random_quaternion()
    >>> R = trns.quaternion_matrix(q)
    >>> B1 = R[:3, :3].dot(A).dot(R[:3, :3].T)
    >>> B2 = qapply_matrix(q, A)
    >>> print(np.allclose(B1, B2))
    True

    """
    R = trns.quaternion_matrix(q)[:3, :3]
    return R.dot(A).dot(R.T)


def angle_axis_from_quaternion(q):
    """Returns the extracted angle and axis from the quaternion q.
    The angle will always be between 0 and pi, inclusive, and the
    axis will always be a unit vector.

    >>> yourAngle, yourAxis = -1337, [-2, 0, 7]
    >>> yourMat = trns.rotation_matrix(yourAngle, yourAxis)
    >>> myAngle, myAxis = angle_axis_from_quaternion(trns.quaternion_from_matrix(yourMat))
    >>> print(np.isclose(myAngle, np.mod(yourAngle, 2*np.pi)))
    True
    >>> print(np.allclose(myAxis, normalize(yourAxis)))
    True

    """
    # Renormalize for accuracy:
    q = normalize(q)
    # If "amount of rotation" is negative, flip quaternion:
    if q[0] < 0:
        q = -q
    # Extract axis:
    imMax = max(map(abs, q[1:]))
    if not np.isclose(imMax, 0):
        axis = normalize(q[1:] / (imMax * npl.norm(q[1:])))
    else:
        axis = np.array([0, 0, 0])
    # Extract angle:
    angle = 2 * np.arccos(q[0])
    # Finally:
    return (angle, axis)


def rotvec_from_quaternion(q):
    """Returns the rotation vector corresponding to
    the quaternion q. A rotation vector is the product
    of the angle of rotation and axis of rotation of
    an SO3 quantity like a quaternion.

    >>> q = trns.random_quaternion()
    >>> r = rotvec_from_quaternion(q)
    >>> M = trns.rotation_matrix(npl.norm(r), r/npl.norm(r))
    >>> print(trns.is_same_transform(M, trns.quaternion_matrix(q)))
    True

    """
    angle, axis = angle_axis_from_quaternion(q)
    return angle * axis


def quaternion_from_rotvec(r):
    """Returns the quaternion equivalent to the given rotation vector r.

    >>> r = np.pi * np.array([0,1,0])
    >>> q = quaternion_from_rotvec(r)
    >>> M1 = trns.quaternion_matrix(q)
    >>> M2 = trns.rotation_matrix(np.pi, [0,1,0])
    >>> print(trns.is_same_transform(M1, M2))
    True

    """
    angle = np.mod(npl.norm(r), 2 * np.pi)
    if not np.isclose(angle, 0):
        return trns.quaternion_about_axis(angle, r / angle)
    else:
        return np.array([1, 0, 0, 0])  # unit real number is identity quaternion


def get_a2b(a, b, rep_out='rotmat'):
    """Returns an SO3 quantity that will align vector a with vector b.
    The output will be in the representation selected by rep_out:
    'rotmat' (rotation matrix) OR 'quaternion'.

    >>> a = trns.random_vector(3)
    >>> b = trns.random_vector(3)
    >>> R = get_a2b(a, b)
    >>> p1 = R.dot(a)
    >>> print(np.allclose(np.cross(p1, b), [0,0,0]))
    True
    >>> p1.dot(b) > 0
    True
    >>> q = get_a2b(a, b, 'quaternion')
    >>> p2 = qapply_points(q, a)
    >>> print(np.allclose(np.cross(p2, b), [0,0,0]))
    True
    >>> p2.dot(b) > 0
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
    else:
        raise ValueError("Invalid rep_out. Choose 'rotmat' or 'quaternion'.")


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


def normalize(v):
    """Returns v unitized by its 2norm.
    (This is a simpler version of trns.unit_vector).

    >>> v0 = np.random.random(3)
    >>> v1 = normalize(v0)
    >>> np.allclose(v1, v0 / npl.norm(v0))
    True

    """
    return np.array(v) / npl.norm(v)


# Module unit test:
if __name__ == "__main__":
    import doctest
    doctest.testmod()
