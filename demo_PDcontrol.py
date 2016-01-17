"""Demo of using oritools in an attitude dynamics simulation and PD controller.

--- DETAILS
The script below is divided into SETUP, SIMULATE, and DISPLAY. In the SETUP
section you can adjust simulation length, timestep, initial conditions, desired
state, mass properties, controller gains, and the display framerate.

In the SIMULATE section, you will find a verbose description of exactly what is done
to carry out the simulation. In short, it is an Euler forward integration, and
the controller applied is a simple PD controller. The simulation and controller
code is super short and neat thanks to oritools.

Once the simulation results have been stored, the script then goes into DISPLAY where
the terror that is matplotlib is used to animate the results. You will notice that the
animation is not real time - this is because matplotlib has issues. Remember that it is
only displaying results; the simulation has already been completed.

In the animation, rigid body orientation is represented as 3 perpendicular lines. Dotted
lines show the initial orientation, and dashed lines show the desired orientation. The
animation loops when it reaches the end of the data, which depends on what you set the
duration to in SETUP. Remember to keep the display window aspect ratio square so it doesn't
distort the perpendicularity of the lines. You can change your view by click&dragging the
plot. You probably want to do this a little or else the lines stop appearing 3D to the eye.
Ignore the future_warning that MatPlotLib prints.

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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
# 1st party
import oritools as ori

################################################# SETUP

# Define time:
dt = 0.01  # global time step (s)
T = 10  # simulation duration (s)
t_arr = np.arange(0, T, dt)  # time values array (s)

# Define body inertia:
I = np.array([[ 1   , 0.01  , 0.02],
              [0.01 ,  2    , 0.03],
              [0.02 , 0.03  ,  3 ]])  # inertia matrix in body frame (kg*m^2)
# I = np.diag([1, 2, 3])  # cool to compare to diagonal inertia matrix case (which would imply body frame is principal frame)
invI = npl.inv(I)  # store inverse for future use

# Define state and initial conditions:
q = trns.random_quaternion()  # orientation state quaternion representing a conversion ***from body frame to world frame***
w = 10 * (np.random.rand(3) - 0.5)  # angular velocity state (rad/s) in world frame
torque = np.array([0, 0, 0]) # initial control input torque, need only be initialized for programming purposes
print('\nInitial Orientation: {}'.format(q))
print('Initial Ang. Velocity: {}'.format(np.rad2deg(w)))

# Controller setup (set gains to 0 if you want to test torque-free precession):
q_des = trns.random_quaternion()  # desired orientation state
w_des = np.array([0, 0, 0])  # desired angular velocity state ()
kp = np.array([150, 150, 150])  # proportional gain (body frame roll, pitch, yaw)
kd = np.array([170, 170, 170])  # derivative gain (body frame rolling, pitching, yawing)
print('Desired Orientation: {}'.format(q_des))
print('Desired Ang. Velocity: {}'.format(np.rad2deg(w_des)))
print('Proportional Gains: {}'.format(kp))
print('Derivative Gains: {}'.format(kd))

# Animation setup:
showplots = True  # Show angle plots before animating? (close plot window to proceed to animation)
framerate = 20  # Despite the display not being real-time, this controls something like a framerate

# Visually represent the body as three perpendicular lines, each defined by its endpoints:
linlen = 0.5
xline = np.array([[0, linlen], [0, 0], [0, 0]])
yline = np.array([[0, 0], [0, linlen], [0, 0]])
zline = np.array([[0, 0], [0, 0], [0, linlen]])
body = np.concatenate((xline, yline, zline), axis=1)  # each column is a point in body

# Initialize histories for recording the simulation:
q_history = np.zeros((len(t_arr), 4))
roll_history, pitch_history, yaw_history = np.zeros(len(t_arr)), np.zeros(len(t_arr)), np.zeros(len(t_arr))  # for plotting something understandable
body_world_history = np.zeros((body.shape[0], body.shape[1], len(t_arr)))  # will store all the body points expressed in world frame at each instant
w_history = np.zeros((len(t_arr), 3))
torque_history = np.zeros((len(t_arr), 3))

################################################# SIMULATE

# Solve the rigid body rotation ODE with first-order integration:
# ---
# Equation 1:  Hnext = Hlast + torque*dt
# ---
# where in this case torque is the controller output and H is the 
# body's angular momentum in world frame,  H = I_world * w   ==>   w = inv(I_world)*H
# In differential equation form, this is the classic law  Hdot = torque  iff
# the origin of the body frame is at the center of mass, which is typically done.
# ---
# Equation 2:  qnext = dq "oriplus" q
# ---
# where dq represents ***nextbody to lastbody*** and q represents ***lastbody to world*** so that
# their orisum is ***nextbody to world*** which then overwrites ***lastbody to world*** as the new q.
# The key here is that w*dt is the rotvec representing ***nextbody to lastbody***,
# and it has an equivalent quaternion expression dq. In differential equation form, the equation
# is  qdot = f(w,dt) = ori.quaternion_from_rotvec(w*dt)/dt, but it is crucial to understand that
# integrating this equation requires use of ori.plus, so it cannot be easily fed into a standard solver.
# ---
# Simulate over t_arr:
for i, t in enumerate(t_arr):
    # Record current state:
    q_history[i, :] = np.copy(q)
    roll_history[i], pitch_history[i], yaw_history[i] = trns.euler_from_quaternion(q, 'rxyz')
    body_world_history[:, :, i] = ori.qapply_points(q, body)
    w_history[i, :] = np.copy(w)
    torque_history[i, :] = np.copy(torque)
    # Current values needed to compute next state:
    I_world = ori.qapply_matrix(q, I)  # current inertia matrix in world frame
    H = I_world.dot(w)  # current angular momentum
    wb = ori.qapply_points(trns.quaternion_inverse(q), w)  # w in body frame
    dq = ori.quaternion_from_rotvec(wb * dt)  # change in orientation for this timestep instant
    # PD controller:
    q_err = ori.error(q, q_des)  # q_err is a rotvec
    w_err = w_des - w
    kpW = np.diag(ori.qapply_matrix(q, np.diag(kp))) # world frame kp gains
    kdW = np.diag(ori.qapply_matrix(q, np.diag(kd))) # world frame kd gains
    torque = (kpW * q_err) + (kdW * w_err)
    # Compute next state:
    q = ori.plus(dq, q)  # new orientation computed using dq and old q
    I_world = ori.qapply_matrix(q, I)  # new I_world computed using new q
    H = H + (torque * dt)  # new H from old H and torque
    w = npl.inv(I_world).dot(H)  # new angular velocity computed using new I_world and new H

################################################# DISPLAY

if showplots:
    fig1 = plt.figure()
    fig1.suptitle('Orientation State Evolution', fontsize=24)

    # Plot roll:
    ax1 = fig1.add_subplot(3, 3, 1)
    ax1.plot(t_arr, np.rad2deg(roll_history))
    ax1.set_ylabel('roll (deg)', fontsize=16)
    ax1.grid(True)

    # Plot pitch:
    ax2 = fig1.add_subplot(3, 3, 4)
    ax2.plot(t_arr, np.rad2deg(pitch_history))
    ax2.set_ylabel('pitch (deg)', fontsize=16)
    ax2.grid(True)

    # Plot yaw:
    ax3 = fig1.add_subplot(3, 3, 7)
    ax3.plot(t_arr, np.rad2deg(yaw_history))
    ax3.set_xlabel('time (s)', fontsize=16)
    ax3.set_ylabel('yaw (deg)', fontsize=16)
    ax3.grid(True)

    # Plot rolling:
    ax1 = fig1.add_subplot(3, 3, 2)
    ax1.plot(t_arr, np.rad2deg(w_history[:, 0]))
    ax1.set_ylabel('w_x (deg/s)', fontsize=16)
    ax1.grid(True)

    # Plot pitching:
    ax2 = fig1.add_subplot(3, 3, 5)
    ax2.plot(t_arr, np.rad2deg(w_history[:, 1]))
    ax2.set_ylabel('w_y (deg/s)', fontsize=16)
    ax2.grid(True)

    # Plot yawing:
    ax3 = fig1.add_subplot(3, 3, 8)
    ax3.plot(t_arr, np.rad2deg(w_history[:, 2]))
    ax3.set_ylabel('w_z (deg/s)', fontsize=16)
    ax3.set_xlabel('time (s)', fontsize=16)
    ax3.grid(True)

    # Plot torque_x:
    ax1 = fig1.add_subplot(3, 3, 3)
    ax1.plot(t_arr, torque_history[:, 0])
    ax1.set_ylabel('T_x (N*m)', fontsize=16)
    ax1.grid(True)

    # Plot torque_y:
    ax2 = fig1.add_subplot(3, 3, 6)
    ax2.plot(t_arr, torque_history[:, 1])
    ax2.set_ylabel('T_y (N*m)', fontsize=16)
    ax2.grid(True)

    # Plot torque_z:
    ax3 = fig1.add_subplot(3, 3, 9)
    ax3.plot(t_arr, torque_history[:, 2])
    ax3.set_ylabel('T_z (N*m)', fontsize=16)
    ax3.set_xlabel('time (s)', fontsize=16)
    ax3.grid(True)

    plt.show()

fig2 = plt.figure()
fig2.suptitle('Orientation State Evolution', fontsize=24)

ax4 = p3.Axes3D(fig2)
ax4.set_xlim3d([-1, 1])
ax4.set_ylim3d([-1, 1])
ax4.set_zlim3d([-1, 1])
ax4.set_xlabel('- World X +')
ax4.set_ylabel('- World Y +')
ax4.set_zlabel('- World Z +')
ax4.grid(True)

# Plot desired:
body_des = 2 * ori.qapply_points(q_des, body)
ax4.plot(body_des[0, :2], body_des[1, :2], body_des[2, :2], color='red', ls='--', linewidth=0.8)
ax4.plot(body_des[0, 2:4], body_des[1, 2:4], body_des[2, 2:4], color='green', ls='--', linewidth=0.8)
ax4.plot(body_des[0, 4:6], body_des[1, 4:6], body_des[2, 4:6], color='blue', ls='--', linewidth=0.8)

# Plot initial:
body_world_history_init = 2 * body_world_history[:, :, 0]
ax4.plot(body_world_history_init[0, :2], body_world_history_init[1, :2], body_world_history_init[2, :2], color='red', ls=':', linewidth=0.8)
ax4.plot(body_world_history_init[0, 2:4], body_world_history_init[1, 2:4], body_world_history_init[2, 2:4], color='green', ls=':', linewidth=0.8)
ax4.plot(body_world_history_init[0, 4:6], body_world_history_init[1, 4:6], body_world_history_init[2, 4:6], color='blue', ls=':', linewidth=0.8)

# Create drawing objects:
x = ax4.plot(body_world_history[0, :2, 0], body_world_history[1, :2, 0], body_world_history[2, :2, 0], color='red', linewidth=4)
y = ax4.plot(body_world_history[0, 2:4, 0], body_world_history[1, 2:4, 0], body_world_history[2, 2:4, 0], color='green', linewidth=4)
z = ax4.plot(body_world_history[0, 4:6, 0], body_world_history[1, 4:6, 0], body_world_history[2, 4:6, 0], color='blue', linewidth=4)

def update(arg, ii=[0]):
    i = ii[0]
    if np.isclose(t_arr[i], np.around(t_arr[i], 1)):
        fig2.suptitle('Orientation State Evolution (Time: {})'.format(t_arr[i]), fontsize=24)
    x[0].set_data(body_world_history[0, :2, i], body_world_history[1, :2, i])
    x[0].set_3d_properties(body_world_history[2, :2, i])
    y[0].set_data(body_world_history[0, 2:4, i], body_world_history[1, 2:4, i])
    y[0].set_3d_properties(body_world_history[2, 2:4, i])
    z[0].set_data(body_world_history[0, 4:6, i], body_world_history[1, 4:6, i])
    z[0].set_3d_properties(body_world_history[2, 4:6, i])
    ii[0] += int(1 / (dt * framerate))
    if ii[0] >= len(t_arr):
        ii[0] = 0
    return [x, y, z]

ani = animation.FuncAnimation(fig2, func=update, interval=dt*1000)
print('Remember to keep the diplay window aspect ratio square!')
print('')
plt.show()
