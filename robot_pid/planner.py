import numpy as np
import heapq

def generate_waypoints(start, goal, num_points=20, grid_size=(20, 20), obstacles=None):
    """
    Generate waypoints from start to goal using A* algorithm on a 2D grid.
    
    Args:
        start (tuple): (x, y) start position in meters.
        goal (tuple): (x, y) goal position in meters.
        num_points (int): Target number of waypoints.
        grid_size (tuple): (width, height) of the grid in cells.
        obstacles (list): List of (x, y) grid cells that are obstacles.
    
    Returns:
        list: List of (x, y) waypoints in meters.
    """
    if obstacles is None:
        obstacles = [(5, 5), (5, 6), (6, 5), (6, 6),  # Example obstacle
                     (10, 10), (10, 11), (11, 10), (11, 11)]  # Another obstacle

    # Convert to grid indices
    grid_width, grid_height = grid_size
    start_grid = (int(start[0]), int(start[1]))
    goal_grid = (int(goal[0]), int(goal[1]))
    start_grid = (max(0, min(grid_width - 1, start_grid[0])), max(0, min(grid_height - 1, start_grid[1])))
    goal_grid = (max(0, min(grid_width - 1, goal_grid[0])), max(0, min(grid_height - 1, goal_grid[1])))

    # Create grid (0: free, 1: obstacle)
    grid = np.zeros(grid_size, dtype=int)
    for obs in obstacles:
        if 0 <= obs[0] < grid_width and 0 <= obs[1] < grid_height:
            grid[obs[0], obs[1]] = 1

    # A* heuristic
    def heuristic(a, b):
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    # Movements (4 directions)
    movements = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # A* setup
    open_set = [(0, start_grid)]
    heapq.heapify(open_set)
    came_from = {}
    g_score = {start_grid: 0}
    f_score = {start_grid: heuristic(start_grid, goal_grid)}

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == goal_grid:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_grid)
            path.reverse()

            # Convert to continuous waypoints (cell centers)
            waypoints = [(x + 0.5, y + 0.5) for x, y in path]

            # Adjust to target num_points
            if len(waypoints) > num_points:
                indices = np.linspace(0, len(waypoints) - 1, num_points, dtype=int)
                waypoints = [waypoints[i] for i in indices]
            elif len(waypoints) < num_points:
                new_waypoints = []
                for i in range(len(waypoints) - 1):
                    x0, y0 = waypoints[i]
                    x1, y1 = waypoints[i + 1]
                    n = int(np.ceil(num_points / (len(waypoints) - 1)))
                    t = np.linspace(0, 1, n, endpoint=(i == len(waypoints) - 2))
                    for ti in t:
                        new_waypoints.append((x0 + ti * (x1 - x0), y0 + ti * (y1 - y0)))
                waypoints = new_waypoints[:num_points]

            return waypoints

        for dx, dy in movements:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height and grid[neighbor[0], neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # Fallback to linear if no path
    print("Warning: A* failed to find a path. Falling back to linear waypoints.")
    x_points = np.linspace(start[0], goal[0], num_points)
    y_points = np.linspace(start[1], goal[1], num_points)
    return list(zip(x_points, y_points))