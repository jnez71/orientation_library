This is a library for dealing with rigid body orientation math.

**so3tools**

Representation independent functions for manipulating SO3 quantities
like rotation matrices, affine (homogeneous) matrices, quaternions, 
and rotation vectors (sometimes called Euler vectors).
These functions can be used to abstract SO3 operations so that
the user can think in terms of simple vector math. For example,
"minus" can be used to find the difference between two orientations.

**oritools**

A simplified version of so3tools that is only designed to
work with the quaternion representation. It is advisable to
use oritools and to keep your orientation states as quaternions
in practice. This is faster and cleaner.

**demo_PDcontroller**

Demo of using oritools in an attitude dynamics simulation
and PD controller. The demo could use so3tools just as well,
but to reinforce good practice, a quaternion representation
and oritools are used.

**transformations**

The latest version of Christoph Gohlke's notorious transformations
module. This version is newer than the one used in ROS tf. When using
anything from this library, this transformations file will be imported,
not the old one from tf.
Current Version: 2015.07.18

**test_oritools**

An all-cases unit test for oritools. This is in addition to the
doctest built into oritools (and so3tools for that matter).

**(dependencies)**

- numpy
- matplotlib (only for demo_PDcontroller)

**(references)**

- https://en.wikipedia.org/wiki/Rotation_(mathematics)
- https://en.wikipedia.org/wiki/Axis-angle_representation
- http://www.euclideanspace.com/maths/geometry/rotations/
- https://en.wikipedia.org/wiki/Versor
- http://arxiv.org/pdf/1107.1119.pdf
