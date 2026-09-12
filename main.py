import numpy as np
from simulator import Simulator, centerline

sim = Simulator()

prev_error = 0

def controller(x):
    global prev_error

    """controller for a car

    Args:
        x (ndarray): numpy array of shape (5,) containing [x, y, heading, velocity, steering angle]

    Returns:
        ndarray: numpy array of shape (2,) containing [fwd acceleration, steering rate]
    """
    xpos   = x[0]                   # current x position
    ypos   = x[1]                   # current y position
    phi    = np.mod(x[2], 2*np.pi)  # current heading (radians)
    v      = x[3]                   # current velocity
    theta   = x[4]                  # current steering angle

    dt = 0.01

    kp = 10.5 # proportional for PID
    kd = 1 # derivate for PID

    wheel_l = 1.58 / 2

    steps = np.arange(0, 1200, 1)
    dist = centerline(steps)
    dist_error = np.sqrt((dist[:, 0] - xpos)**2 + (dist[:, 1] - ypos)**2)
    cur_dist = steps[np.argmin(dist_error)]

    forward_dist = cur_dist + 6
    forward = centerline(forward_dist)

    desired_heading = np.arctan2(forward[1] - ypos, forward[0] - xpos)
    theta_error = np.arctan2(np.sin(desired_heading - phi), np.cos(desired_heading - phi))
    desired_theta = np.clip(theta_error, -0.7, 0.7)

    error = desired_theta - theta

    P = kp * error
    D = kd * ((error - prev_error) / dt )

    str_rate = np.clip(P + D, -1, 1)

    prev_error = error

    near_curve = cur_dist + np.arange(2, 20, 2) 
    far_curve = cur_dist + np.arange(20, 40, 10)

    curve_ahead = np.append(near_curve, far_curve)

    step = 2

    before_curve = centerline(curve_ahead - step)
    at_curve = centerline(curve_ahead)
    after_curve = centerline(curve_ahead + step)

    x_prime = (after_curve[:, 0] - before_curve[:, 0]) / step * 2
    y_prime = (after_curve[:, 1] - before_curve[:, 1]) / step * 2

    x_double_prime = (after_curve[:, 0] - 2*at_curve[:, 0] + before_curve[:, 0]) / step**2
    y_double_prime = (after_curve[:, 1] - 2*at_curve[:, 1] + before_curve[:, 1]) / step**2

    num = np.abs((x_prime * y_double_prime) - (y_prime * x_double_prime))
    denom = (x_prime**2 + y_prime**2)**1.5 + 1e-9
    curvature = num / denom

    if len(curvature):
        max_c = np.max(curvature)
    else:
        max_c = float("infinity")

    a_brake_limit = 12
    dist = curve_ahead - cur_dist

    v_corner = np.sqrt(12 / max_c)
    v_safe = np.sqrt(v_corner**2 + a_brake_limit * dist)
    v_target = np.min(v_safe)
    v_final = np.clip(v_target, 5, 11)

    if v < (v_final - 0.1):
        a = 4
    else:
        a = 3 * (v_final - v)

    a_y = v**2 * max_c

    if a_y >= 11.5:
        a_max = 0
    else:
        a_max = np.sqrt(max(0, 12**2 - a_y**2))

    final_a = np.clip(a, max(-10, -a_max), min(4, a_max))

    return np.array([final_a, str_rate])



sim.set_controller(controller)
sim.run()
sim.animate()
sim.plot()

print(f"total time: {np.max(sim.controller_times)*1000:.3f}")