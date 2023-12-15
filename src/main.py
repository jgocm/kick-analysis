from libs import Read, Track
import os

# Get path to csv files' directory
csvs_dir = os.getcwd() + '/kicksEvents'

# Get the list of CSV files in the folder and sort them in ascending order
csv_files = sorted([filename for filename in os.listdir(csvs_dir) if filename.endswith('.csv')])

# Checking if data were cut correctly
'''
First, we exclude the data after the kick has ended and plot it to check
'''
trackers = []
for filename in csv_files:
    file_path = os.path.join(csvs_dir, filename)
    file_name_without_ext = os.path.splitext(filename)[0]  # Remove the extension
    #print(f'Reading new data: {file_name_without_ext}')
    data = Read.VisionKicksReader(file_path, file_name_without_ext)
    tracker = Track.BallTracker(data)
    tracker.plot()
    trackers.append(tracker)


# Plot multiple kicks
'''
Here, we apply some filtering to the data and plot the kicks
just for visualizing their behaviors
'''
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
plt.title('Ball Kicks and Detected Max Speeds')
plt.legend()
plt.show()

# 1-baseado na velocidade máxima, qual a trajetória?
# 2-baseado na força, qual a velocidade máxima?
from sklearn.linear_model import RANSACRegressor

# Estimate ball deacceleration based on the maxium velocity
'''
Here, we assume the ball's speed to decrease linearly with the
distance from the max speed until the end and find the best lines
that approximate this part of the movement
'''
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
plt.title('Linear Regression for Each Kick Force')
plt.show()


# Find B from max_speed
'''
The lines from the previous step have very similar angular coefficients,
so we assume this parameter to be constant and try to find a relation
between the linear coefficients (B) and the ball's max speeds
'''
y = np.array(Bs).reshape(-1, 1)
plt.scatter(max_speeds, y, label='original')
plt.xlabel('Max Speed (m/s)')
plt.ylabel('B')
plt.title('Ball Kicks')
plt.legend()
plt.show()
plt.clf()

# Compute linear regression
'''
From the plot, the linear coefficients (B) and the ball's max speeds seem
to have a linear relation, so we find a linear regression for it and
compare our results with the original ones
'''
model = RANSACRegressor()
X = np.array(max_speeds).reshape(-1, 1)
y = np.array(Bs).reshape(-1, 1)
model.fit(X, y)
y_predict = model.estimator_.coef_[0]*X+model.estimator_.intercept_
Av = np.mean(As)
Ab = model.estimator_.coef_[0]
Bb = model.estimator_.intercept_

plt.scatter(max_speeds, y, label='original')
plt.scatter(max_speeds, y_predict, label='regression')
plt.xlabel('Max Speed (m/s)')
plt.ylabel('B')
plt.title('Ball Kicks')
plt.legend()
plt.show()
plt.clf()

# Plot results for testing
'''
Here, we plot the lines generated from our regression against
the original data, to check how well they match
'''
for tracker in trackers[2:]:
    Vmax = tracker.max_speed
    d = tracker.distances
    v = Av*d + Ab*Vmax + Bb
    plt.plot(d, v)
    plt.plot(moving_average(tracker.distances, 3), 
             moving_average(tracker.speeds, 3), 
             label=tracker.title)

plt.xlabel('Distance (m)')
plt.ylabel('Speed (m/s)')
plt.title('Our Regression vs Original')
plt.show()

# Determining max_speed from the kick strength (discharge time in ms)
'''
So far, we could find a relation between the ball's maximum speed and
it's trajectory. Now we will try to find a relation between the maximum
speed and the kick strength, so that we can merge these formulas and 
estimate the whole ball's trajectory based on the kick strength and, thus,
the inverse too, allowing us to choose which strength we want to apply,
based on the trajectory we want the ball to execute.
'''

# Analyzing the data
'''
First, we analyze the ball's max speed vs the kick strength, 
which is actually given by the discharge time in milliseconds
'''
As = []
Bs = []
dists = []
kick_strengths = []
max_speeds = []
for tracker in trackers:
    dists.append(tracker.distances[np.argmax(tracker.speeds)])
    kick_strengths.append(tracker.kick_strength)
    max_speeds.append(tracker.max_speed)

X = np.array(kick_strengths)
plt.scatter(kick_strengths, max_speeds, label='original')
plt.xlim(0, 7)
plt.ylim(0, 3.5)
plt.xlabel('Discharge Time (ms)')
plt.ylabel('Maximum Speed (m/s)')
plt.title('Ball Kicks')
plt.legend()
plt.show()

# Regression
'''
We assume the previous behavior to be exponential, since it
is directly related to the capacitor's voltage discharge on the
solenoid. Thus, it should be in the form:

    v = A*(1 - e^(-B*(F-Fmin)))

where v is the ball's speed, F is the kick strength (discharge time),
and Fmin is a parameter that relates to the minimum kick strength, i.e.,
the minimum discharge time that makes the ball move.

For finding A and B that best fit our equation and data, we will use
an optimizer from the scipy library and plot the results against our
original data
'''
from scipy.optimize import curve_fit
Fmin = 0.9
def func(X, A, B):
    return A * (1 - np.exp(-B * (X-Fmin)))

X = np.array(kick_strengths)
y = np.array(max_speeds)
popt, pcov = curve_fit(func, X, y)
Ad = popt[0]
Bd = popt[1]
y_predict = func(X, Ad, Bd)

plt.scatter(kick_strengths, max_speeds, label='original')
plt.scatter(kick_strengths, y_predict, label='regression')
plt.xlim(0, 7)
plt.ylim(0, 3.5)
plt.xlabel('Discharge Time (ms)')
plt.ylabel('Maximum Speed (m/s)')
plt.title('Ball Kicks')
plt.legend()
plt.show()
   
# Test complete equation
'''
Here, we merge our estimated regressions for ploting the results
of our complete equation against the original data
'''
for tracker in trackers[2:]:
    distances = tracker.distances
    speeds_measured = tracker.speeds
    v_max_estimated = func(tracker.kick_strength, Ad, Bd)
    speeds_estimated = Av*distances + Ab*v_max_estimated + Bb
    plt.plot(distances, speeds_estimated)
    plt.plot(distances, speeds_measured)

    # Highlight max speed point
    plt.plot(distances[np.argmax(tracker.speeds)], 
             speeds_measured[np.argmax(tracker.speeds)], 
             marker='o', 
             markersize=3, 
             color='red')
    v_max_estimated_index = np.argmin(np.abs(speeds_estimated-v_max_estimated))
    plt.plot(distances[v_max_estimated_index], 
             speeds_estimated[v_max_estimated_index], 
             marker='o', 
             markersize=3, 
             color='red')
    
plt.xlabel('Distance (m)')
plt.ylabel('Speed (m/s)')
plt.title('Ball Kicks')
plt.show()
plt.clf()

# Checking coefficients
Ab = Ab[0]
Bb = Bb[0]
print(f'Av: {Av} , Ab: {Ab}, Bb: {Bb}, Ad: {Ad}, Bd: {Bd}, Fmin: {Fmin}')

# Test final equation
'''
The equation for determining the ball speed by the traveled distance 
for a given kick strength (discharge time in ms), can be expressed as:
    v = Av*d + Ab*Ad*(1-exp(-Bd(F-Fmin))) + Bb

We invert this equation as:
    exp(-Bb(F-min)) = 1 - (v - Av*d - Bb)/(Ab*Ad)

We express 'gamma' as the term:
    gamma = (v - Av*d - Bb)/(Ab*Ad)

Finally, F can be determined from:
    F = Fmin - Log(1 - gamma)/Bd
'''

robot_distances = [1, 2, 3, 4, 5, 6, 7, 8]
desired_ball_speed = 2
Fmax = 6
Fs = []

for d in robot_distances:
    v = desired_ball_speed
    gamma = (v - Av*d - Bb) / (Ab*Ad)
    if gamma>1: gamma=0.999
    F = Fmin - np.log(1-gamma)/Bd
    if F<Fmin: F = Fmin
    elif F>Fmax: F = Fmax
    Fs.append(F)

plt.scatter(robot_distances, Fs)
plt.xlim(0,9)
plt.ylim(0,7)
plt.xlabel('Robot Distance (m)')
plt.ylabel('Discharge Time (ms)')
plt.title('Ball Kicks')
plt.show()