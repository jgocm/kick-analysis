import csv
import numpy as np

class VisionKicksReader:
    def __init__(self, file_path, label):
        self.label = label
        self.file_path = file_path
        self.data = []
        with open(self.file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if not [float(row['position_x']),float(row['position_y']),float(row['velocity_x']),float(row['velocity_y'])] == [0,0,0,0]:
                    self.data.append(row)

    def get_positions(self):
        positions = []
        for row in self.data:
            positions.append((float(row['position_x']), float(row['position_y'])))
        return np.array(positions)/1000
    
    def get_velocities(self):
        velocities = []
        for row in self.data:
            velocities.append((float(row['velocity_x']), float(row['velocity_y'])))
        return np.array(velocities)/1000

class RobotInfoReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        with open(self.file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.data.append(row)
    
    def get_parsed_data(self):
        balls = []
        kick_loads = []
        timestamps = []
        for row in self.data:
            balls.append(True if row['HAS_BALL']=='True' else False)
            kick_loads.append(float(row['KICK_LOAD']))
            timestamps.append(float(row['TIMESTAMP']))
        return np.array(balls), np.array(kick_loads), np.array(timestamps)

if __name__ == "__main__":
    kick_event_nr = 0
    csv_dir = f'/home/rc-blackout/Documents/RoboCIn-Embarcados/kicks/kicksEvents'
    kicks_reader = VisionKicksReader(csv_dir + f'/kickEvent{kick_event_nr}.csv')
    positions = kicks_reader.get_positions()

    robot_reader = RobotInfoReader(csv_dir + f'/log.csv')
    balls, kick_loads, timestamps = robot_reader.get_parsed_data()
    print(timestamps)
    