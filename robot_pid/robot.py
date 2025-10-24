import math

class Robot:
    """
    Simple 2D robot model using differential drive kinematics.
    
    Args:
        x (float): Initial x position.
        y (float): Initial y position.
        heading (float): Initial heading (radians).
    """
    def __init__(self, x=0.0, y=0.0, heading=0.0):
        self.x = x
        self.y = y
        self.heading = heading  # in radians

    def step(self, v, omega, dt):
        """
        Update robot pose.
        
        Args:
            v (float): Linear velocity (m/s).
            omega (float): Angular velocity (rad/s).
            dt (float): Time step (s).
        """
        self.x += v * math.cos(self.heading) * dt
        self.y += v * math.sin(self.heading) * dt
        self.heading += omega * dt