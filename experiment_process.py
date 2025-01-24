from scipy.stats import binom
import os
import pandas as pd
from experiment_plot import read_csv_value 

n = 3 * 3 * 4      # Number of trials: 3 test scenes, 3 speeds, 4 bitrates
p0 = 0.95   # Probability of success, computed from csv
cp = [0.025, 0.975]    # Cumulative probability threshold, 95% confidence interval

folder_path = r'C:\Users\15142\new\Falcor\Source\Samples\EncodeDecode\experimentResult'
x_positions = read_csv_value(folder_path)
# print(f'x_positions {x_positions}')
bitrates = [1000, 2000, 4000, 8000]
speeds = [0.5, 1, 2]
nested_dict = {}
for bitrate, speed, score in x_positions:
    if bitrate not in nested_dict:
        nested_dict[bitrate] = {}  # Initialize sub-dictionary for this bitrate
    if speed not in nested_dict[bitrate]:
        nested_dict[bitrate][speed] = []  # Initialize an empty list for this speed
    nested_dict[bitrate][speed].append(score)  # Append the score to the list for this speed

print(f'nested_dict {nested_dict}')
y_range = {}
for bitrate in bitrates:
    for speed in speeds:
        y_range[speed] = [] if speed not in y_range else y_range[speed]
        print(f'===== bitrate {bitrate}, speed {speed} =====')
        if bitrate in nested_dict and speed in nested_dict[bitrate]:
            p = sum(nested_dict[bitrate][speed]) / len(nested_dict[bitrate][speed])
            k = binom.ppf(cp, n, p)
            print(f'p {p}, k {k}')
            y_range[speed].append(tuple(k))
print(y_range)
# {8000.0: {0.5: 1.0, 2: 1}, 1000.0: {0.5: 1.0, 2: 1}, 4000.0: {0.5: 1.0, 2: 1},}