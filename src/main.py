from libs import Read, Track
import os

# Get path to csv files' directory
csvs_dir = os.getcwd() + '/kicksEvents'

# Get the list of CSV files in the folder and sort them in ascending order
csv_files = sorted([filename for filename in os.listdir(csvs_dir) if filename.endswith('.csv')])

trackers = []
for filename in csv_files:
    file_path = os.path.join(csvs_dir, filename)
    file_name_without_ext = os.path.splitext(filename)[0]  # Remove the extension
    #print(f'Reading new data: {file_name_without_ext}')
    data = Read.VisionKicksReader(file_path, file_name_without_ext)
    tracker = Track.BallTracker(data)
    #tracker.plot()
    trackers.append(tracker)


# Plot multiple kicks
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size):
    # Pad the data at the beginning and end with zeros
    padded_data = np.pad(data, (window_size//2, window_size//2), mode='edge')
    
    # Construct the moving average window
    window = np.ones(window_size) / window_size
    
    # Apply the moving average filter
    smoothed_data = np.convolve(padded_data, window, mode='valid')
    
    return smoothed_data

for tracker in trackers:
    # Filtering data
    tracker.distances = moving_average(tracker.distances, 3)
    tracker.speeds = moving_average(tracker.speeds, 3)

    # Highlight max speed point
    plt.plot(tracker.distances[np.argmax(tracker.speeds)], 
             tracker.speeds[np.argmax(tracker.speeds)], 
             marker='o', 
             markersize=3, 
             color='red')
    plt.plot(moving_average(tracker.distances, 3), 
             moving_average(tracker.speeds, 3), 
             label=tracker.title)

plt.xlabel('Distance (m)')
plt.ylabel('Speed (m/s)')
plt.title('Ball Kicks')
plt.legend()
plt.show()

# TODO: quebrar o problema em duas partes oara resolver a regressão:
# 1-baseado na velocidade máxima, qual a trajetória?
# 2-baseado na força, qual a velocidade máxima?
from sklearn.linear_model import RANSACRegressor

# Estimate ball deacceleration based on the maxium velocity
As = []
Bs = []
max_speeds = []
for tracker in trackers[2:]:
    plt.plot(tracker.distances[np.argmax(tracker.speeds):], 
             tracker.speeds[np.argmax(tracker.speeds):])
    
    # Compute linear regression
    model = RANSACRegressor()
    X = tracker.distances[np.argmax(tracker.speeds):].reshape(-1, 1)
    y = tracker.speeds[np.argmax(tracker.speeds):].reshape(-1, 1)
    model.fit(X, y)
    A, B = model.estimator_.coef_[0], model.estimator_.intercept_
    As.append(A)
    Bs.append(B)
    max_speeds.append(tracker.max_speed)
    y_predict = A*X + B
    plt.plot(tracker.distances[np.argmax(tracker.speeds):], 
             y_predict)

plt.xlabel('Distance (m)')
plt.ylabel('Speed (m/s)')
plt.title('Ball Kicks')
plt.show()


# Find B from max_speed

# Compute linear regression
model = RANSACRegressor()
X = np.array(max_speeds).reshape(-1, 1)
y = np.array(Bs).reshape(-1, 1)
model.fit(X, y)
y_predict = model.estimator_.coef_[0]*X+model.estimator_.intercept_
print(f'A: {np.mean(As)} , ab: {model.estimator_.coef_[0]}, bb: {model.estimator_.intercept_}')

plt.scatter(max_speeds, y, label='original')
plt.scatter(max_speeds, y_predict, label='regression')
plt.xlabel('Max Speed (m/s)')
plt.ylabel('B')
plt.title('Ball Kicks')
plt.legend()
plt.show()
plt.clf()


'''
Coefficients can be expressed as:
A = np.mean(As)
B = model.estimator_.coef_[0]*max_speed+model.estimator_.intercept_
'''

# Determining distance for achieving max_speed from the kick strength (discharge time in ms)
from scipy.optimize import curve_fit

def func(X, A, B):
    return A * (1 - np.exp(-B * (X-1)))

As = []
Bs = []
dists = []
kick_strengths = []
for tracker in trackers:
    dists.append(tracker.distances[np.argmax(tracker.speeds)])
    kick_strengths.append(tracker.kick_strength)

X = np.array(kick_strengths)
y = np.array(dists)
popt, pcov = curve_fit(func, X, y)
A = popt[0]
B = popt[1]
y_predict = func(X, A, B)

plt.scatter(kick_strengths, dists, label='original')
plt.scatter(kick_strengths, y_predict, label='regression')
plt.xlim(0, 7)
plt.ylim(0, 3.5)
plt.xlabel('Discharge Time (ms)')
plt.ylabel('Distance for Maximum Speed (m)')
plt.title('Ball Kicks')
plt.legend()
plt.show()
   
# Plot results for testing
