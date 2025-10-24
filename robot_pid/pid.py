class PID:
    """
    PID Controller class for heading or velocity control.
    
    Args:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        output_limits (tuple): (min, max) output limits.
    """
    def __init__(self, kp, ki, kd, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output, self.max_output = output_limits

        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        """
        Update PID output.
        
        Args:
            error (float): Current error.
            dt (float): Time step.
        
        Returns:
            float: PID output.
        """
        # Proportional term
        p = self.kp * error

        # Integral term
        self.integral += error * dt
        i = self.ki * self.integral

        # Derivative term
        d = self.kd * (error - self.prev_error) / dt if dt > 0 else 0.0

        # PID output
        output = p + i + d

        # Update previous error
        self.prev_error = error

        # Clamp output
        if self.min_output is not None and output < self.min_output:
            output = self.min_output
        if self.max_output is not None and output > self.max_output:
            output = self.max_output

        return output