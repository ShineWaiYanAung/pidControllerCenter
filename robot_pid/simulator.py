import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from .pid import PID
from .robot import Robot
from .planner import generate_waypoints

def angle_wrap(a):
    """Wrap angle between [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi

def compute_cross_track_error(traj_x, traj_y, waypoints):
    """
    Compute average and max cross-track error (minimum distance to nearest waypoint).
    """
    ctes = []
    for tx, ty in zip(traj_x, traj_y):
        min_dist = float('inf')
        for wx, wy in waypoints:
            dist = math.sqrt((tx - wx)**2 + (ty - wy)**2)
            min_dist = min(min_dist, dist)
        ctes.append(min_dist)
    return np.mean(ctes) if ctes else 0.0, max(ctes) if ctes else 0.0

def compute_path_length(traj_x, traj_y):
    """Compute total path length."""
    points = np.array(list(zip(traj_x, traj_y)))
    diffs = np.diff(points, axis=0)
    return np.sum(np.sqrt(np.sum(diffs**2, axis=1))) if len(points) > 1 else 0.0

def run_simulation(kp=2.0, ki=0.1, kd=0.3, show_plot=True, animate=True):
    """
    Simulate robot following path using PID heading and velocity control with optional animation.
    
    Args:
        kp, ki, kd (float): PID gains.
        show_plot (bool): Show static plot at the end.
        animate (bool): Show real-time animation of robot movement.
    
    Returns:
        tuple: (time_to_goal, path_length, avg_cte, max_cte)
    """
    robot = Robot(0, 0, 0)  # Start pose
    waypoints = generate_waypoints((0, 0), (18, 18), 20)  # Generate waypoints with A*
    heading_pid = PID(kp, ki, kd, output_limits=(-2, 2))
    v_pid = PID(1.0, 0.1, 0.1, output_limits=(0.1, 2.0))  # Velocity PID
    v_cmd_base = 1.0  # Base forward speed
    dt = 0.1  # Time step
    waypoint_tolerance = 1.0  # Tolerance for convergence

    traj_x, traj_y, headings = [], [], []
    total_steps = 0

    # Store states for animation
    states = []

    for i, (wx, wy) in enumerate(waypoints):
        iter_count = 0
        while True:
            dx = wx - robot.x
            dy = wy - robot.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < waypoint_tolerance:
                break

            # Desired heading and heading error
            desired_heading = math.atan2(dy, dx)
            heading_error = angle_wrap(desired_heading - robot.heading)
            omega_cmd = heading_pid.update(heading_error, dt)

            # Velocity control based on distance to waypoint
            v_error = dist - waypoint_tolerance
            v_cmd = v_pid.update(v_error, dt) if dist > waypoint_tolerance else 0.1
            v_cmd = max(0.1, min(v_cmd_base, v_cmd))  # Limit velocity

            # Update robot pose
            robot.step(v_cmd, omega_cmd, dt)

            # Record trajectory and state
            traj_x.append(robot.x)
            traj_y.append(robot.y)
            headings.append(robot.heading)
            states.append((robot.x, robot.y, robot.heading))
            total_steps += 1
            iter_count += 1

            # Prevent infinite loop
            if iter_count > 2000:
                print(f"Warning: Max iterations reached for waypoint {i+1}. Skipping.")
                break

    # Compute metrics
    time_to_goal = total_steps * dt
    path_length = compute_path_length(traj_x, traj_y)
    avg_cte, max_cte = compute_cross_track_error(traj_x, traj_y, waypoints)

    if animate:
        # Set up the figure for animation
        fig, ax = plt.subplots()
        ax.plot([p[0] for p in waypoints], [p[1] for p in waypoints], "ro--", label="Planned path")
        trajectory, = ax.plot([], [], "b-", label="Robot path")
        robot_marker, = ax.plot([], [], "g^", label="Robot", markersize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"PID Run (Kp={kp}, Ki={ki}, Kd={kd})")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")

        # Animation update function
        def update(frame):
            x, y, heading = states[frame]
            trajectory.set_data(traj_x[:frame+1], traj_y[:frame+1])
            robot_marker.set_data([x], [y])
            # Update robot orientation (triangle marker rotates with heading)
            robot_marker.set_marker((3, 0, np.degrees(heading)))
            return trajectory, robot_marker

        # Create animation
        ani = FuncAnimation(fig, update, frames=len(states), interval=dt*1000, blit=True)
        plt.show()

    if show_plot:
        plt.figure()
        plt.plot([p[0] for p in waypoints], [p[1] for p in waypoints], "ro--", label="Planned path")
        plt.plot(traj_x, traj_y, "b-", label="Robot path")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"PID Run (Kp={kp}, Ki={ki}, Kd={kd})")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()

    return time_to_goal, path_length, avg_cte, max_cte