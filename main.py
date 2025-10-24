from robot_pid.simulator import run_simulation

if __name__ == "__main__":
    pid_sets = [
        (0.8, 0.1, 0.2),  # Under-damped
        (2.0, 0.1, 0.3),  # Well-tuned
        (4.0, 0.1, 0.5)   # Over-damped
    ]

    print("PID Tuning | Time (s) | Path Len (m) | Avg CTE (m) | Max CTE (m)")
    print("-" * 50)

    for kp, ki, kd in pid_sets:
        print(f"Running simulation with Kp={kp}, Ki={ki}, Kd={kd}")
        metrics = run_simulation(kp=kp, ki=ki, kd=kd, show_plot=True, animate=True)
        time_to_goal, path_length, avg_cte, max_cte = metrics
        print(f"{kp},{ki},{kd} | {time_to_goal:.2f} | {path_length:.2f} | {avg_cte:.3f} | {max_cte:.3f}")