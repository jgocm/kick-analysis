import numpy as np
import matplotlib.pyplot as plt

class BallDetector:
    def __init__(self, data):
        self.balls, self.kick_loads, self.timestamps = data.get_parsed_data()
    
    def get_kick_events(self):
        kick_events = []
        last_ball = False
        for ball in self.balls:
            if (ball and last_ball): kick_events.append(True)
            else: kick_events.append(False)
            last_ball = ball
        return np.array(kick_events)

    def get_filtered_kick_loads(self, weight=0.1):
        filtered_loads = []
        current_load = 0
        for load in self.kick_loads:
            current_load = weight*load + (1-weight)*current_load
            filtered_loads.append(current_load)
        return np.array(filtered_loads)

    def plot(self):
        # plot data
        plt.plot(self.timestamps, self.get_kick_events(), label='Kick Event')
        plt.plot(self.timestamps, self.get_filtered_kick_loads(), label='Capacitor Load')

        # set axis labels and title
        plt.xlabel('Timestamps')
        plt.ylabel('Value')
        plt.title('Kicks vs Time(s)')

        # set legend
        plt.legend()

        # display plot
        plt.show()

class BallTracker:
    def __init__(self, data):
        self.title = data.label
        self.positions = data.get_positions()
        self.velocities = data.get_velocities()
        self.directions = []
        self.distances = []
        self.speeds = []
        self.calculate_tracking()
        self.max_speed = max(self.speeds)
        self.kick_strength = float(data.label[:4])/1000

    def smallest_angle_diff(self, theta1, theta2):
        diff = theta2 - theta1
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    def get_directions(self):
        position_diffs = self.positions[1:] - self.positions[:-2]
        for diff in position_diffs:
            direction = np.arctan2(diff[1], diff[0])
            self.directions.append(direction)
    
    def limit_trajectory(self):
        #TODO: implement method for checking if the ball has hit the wall
        return True
    
    def calculate_tracking(self):
        for (position, velocity) in zip(self.positions, self.velocities):
            distance = np.linalg.norm(position-self.positions[0])
            speed = np.linalg.norm(velocity)
            self.distances.append(distance)
            self.speeds.append(speed)
            if position[0]>4.1 or position[1]>2.9:
                break
        self.end_point = position
    
    def plot(self):
        _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        ax1.plot(self.distances, self.speeds)
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Speed (m/s)')
        ax1.set_title('Speed vs Distance')
        ax1.set_xlim(0, 8)
        ax1.set_ylim(0, 5)

        ax2.plot(self.positions[:,0], self.positions[:,1])
        ax2.set_xlabel('Position X (m)')
        ax2.set_ylabel('Position Y (m)')
        ax2.set_title('Trajectory')
        ax2.set_xlim(-0.3, 4.2)
        ax2.set_ylim(-3, 3)

        # Highlight initial and final positions
        ax2.plot(self.positions[0][0], self.positions[0][1], marker='o', markersize=5, color='red', label='start')
        ax2.plot(self.end_point[0], self.end_point[1], marker='o', markersize=5, color='black', label='end')
        ax2.legend()

        # Set window title
        plt.suptitle(self.title)
        plt.show()



