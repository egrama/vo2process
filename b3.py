import sys
import numpy as np
from scipy.signal import find_peaks
import argparse
import pandas as pd
from io import StringIO
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from vo2process import *


# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/sample_fb.csv'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/sample2.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/4-130bpm-25c-963hpa-77hum.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/salavlad.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/vlad_sala_1.csv'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/maxlung1_emil.csv'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/vs/xaa'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/labirou1_edited.csv'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/esala2_p2.csv'
parser = argparse.ArgumentParser()
parser.add_argument('csv_file', nargs='?', default=default_csv_file)
parser.add_argument('--log', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
											help='Set the logging level (default: info)')
args = parser.parse_args()
csv_file = args.csv_file
_, part2 = split_csv(csv_file)
csv_data = StringIO(''.join(part2))



def find_breath_limits(df, sg_win_lenght=11, sg_polyorder=2):
  savgol_filtered = savgol_filter(df['oneDp'],
                                  window_length=sg_win_lenght, polyorder=sg_polyorder)
  # Find indexes where the line changes sign and crosses the x-axis
  indexes = [0]
  indexes_ms = [df.index[0]]
  for i in range(len(savgol_filtered)-1):
    # only consider sign changes that have enough pressure (>40) 
    # and are at least 20 samples apart (close to max there are valid breaths with less samples!)
    # and the sum of the pressure has a different sign from the previous one
    dpSum = df.loc[indexes_ms[-1]:].head(i-indexes[-1])['oneDp'].sum()
    if (savgol_filtered[i] * savgol_filtered[i+1] < 0) and \
       (abs(dpSum) > 40 ) and \
       ((i - indexes[-1] > 20) or abs(dpSum) > 400) and \
       (dpSum * df.loc[indexes_ms[-1], 'dpSum'] <= 0):
      df.loc[df.index[i], 'dpSum'] = dpSum
      indexes.append(i)
      indexes_ms.append(df.index[i])
  return indexes, indexes_ms


def sanitize_data():
   # Set to 0 negative pressure values and values lower than sensor threshold
  df.loc[df['dpIn'] < flow_sensor_threshold, 'dpIn'] = 0
  df.loc[df['dpOut'] < flow_sensor_threshold, 'dpOut'] = 0


df = pd.read_csv(csv_data, 
                  names=['ts', 'type', 'millis', 'dpIn', 'dpOut', 'o2'],
                  dtype={'millis': np.float64, 'dpIn': np.float64, 'dpOut': np.float64, 'o2': np.float64})
df.set_index('millis', inplace=True)
sanitize_data()

df['millis_diff'] = df.index.to_series().diff()
df['oneDp'] = df['dpIn'] - df['dpOut'] # signed diff pressure
# rolling_avg = df['oneDp'].rolling(window=10).mean()
df['dpSum'] = 0
savgol_filtered = savgol_filter(df['oneDp'], window_length=11, polyorder=2)

plt.figure(figsize=(12,6))
# plt.plot(range(len(df)), df['o2'], 'r', label='Original')
plt.plot(df.index, df['oneDp'], 'r', label='OneDP')
# plt.plot(range(len(df)), rolling_avg, 'g', label='Rolling Average')
plt.plot(df.index, savgol_filtered, 'b', label='Savitzky-Golay')
plt.axhline(y=0, color='black', linestyle='-')  # Add horizontal line at y=0
plt.xlabel('Position in DataFrame')
plt.ylabel('dpIn - dpOut')
# plt.xticks(range(0, len(df), 20), rotation=45)
plt.legend()


for index in find_breath_limits(df)[1]:
	plt.axvline(x=index, color='black', linestyle='-')
plt.show()
print('the end')

















































































# going with the sign change strategy - leaving this as backup


# def find_valleys(signal, distance=30, height=-1, width=10):
#   """Finds valleys in a signal."""
#   inverted_signal = -signal
#   peaks, _ = find_peaks(inverted_signal, distance=distance, height=height, width=width)
#   return peaks



# for a in ['dpIn', 'dpOut']:
# 	signal = df[a].to_numpy()
# 	filtered_signal = savgol_filter(signal, window_length=11, polyorder=2)
# 	p = find_valleys(filtered_signal)
# 	print(p)
# 	print(len(p))

# 	plt.figure(figsize=(12,6))
# 	plt.plot(range(len(df)), df[a], 'r', label='dpIn')
# 	plt.xlabel('Position in DataFrame')
# 	plt.ylabel('dpIn')
# 	plt.xticks(range(0, len(df), 20), rotation=45)

# 	# Add vertical lines for each value in p
# 	for peak in p:
# 		plt.axvline(x=peak, color='black', linestyle='-')

# 	plt.show()