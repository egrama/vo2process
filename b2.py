import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def analyze_breathing(df, window_size=5, threshold=0.1):
    # Combine dpIn and dpOut into a single column
    df['pressure'] = df['dpIn'] - df['dpOut']
    
    # Apply smoothing using a rolling average
    df['smooth_pressure'] = df['pressure'].rolling(window=window_size, center=True).mean()
    
    # Fill NaN values at the edges with the nearest valid value
    df['smooth_pressure'] = df['smooth_pressure'].fillna(method='bfill').fillna(method='ffill')
    
    # Calculate the derivative of the smoothed pressure
    df['pressure_derivative'] = df['smooth_pressure'].diff()
    
    # Find peaks in the absolute derivative (both positive and negative)
    peaks, _ = find_peaks(np.abs(df['pressure_derivative']), height=threshold)
    
    # Initialize lists to store results
    inspirations = []
    expirations = []
    
    # Analyze each breath cycle
    for i in range(0, len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        
        # Determine if it's inspiration or expiration
        if end_idx > start_idx:
            segment = df.iloc[start_idx:end_idx+1]
            mid_point = segment['smooth_pressure'].idxmax()
            mid_idx = segment.index.get_loc(mid_point)
            
            if segment['smooth_pressure'].iloc[0] < segment['smooth_pressure'].iloc[mid_idx]:
                inspirations.append({
                    'start': segment.index[0],
                    'end': segment.index[mid_idx]
                })
                expirations.append({
                    'start': segment.index[mid_idx],
                    'end': segment.index[-1]
                })
            else:
                expirations.append({
                    'start': segment.index[0],
                    'end': segment.index[mid_idx]
                })
                inspirations.append({
                    'start': segment.index[mid_idx],
                    'end': segment.index[-1]
                })
    
    # Create DataFrames for inspirations and expirations
    inspiration_df = pd.DataFrame(inspirations)
    expiration_df = pd.DataFrame(expirations)
    
    return inspiration_df, expiration_df





default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/salavlad.csv'


csv_file = sys.argv[1] if len(sys.argv) > 1 else default_csv_file
df = pd.read_csv(csv_file, names=['ts', 'type', 'millis', 'dpIn','dpOut', 'o2'])
df.set_index('millis', inplace=True)
# Calculate the time difference between each row
df['millis_diff'] = df.index.to_series().diff()


# Analyze the breathing data
inspiration_df, expiration_df = analyze_breathing(df)

# Print the results
print("Inspirations:")
print(inspiration_df)
print("\nExpirations:")
print(expiration_df)

inspiration_df['duration'] = inspiration_df['end'] - inspiration_df['start']
expiration_df['duration'] = expiration_df['end'] - expiration_df['start']


plt.figure(figsize=(12,6))
plt.plot(inspiration_df.index, inspiration_df['duration'], 'r', label='Inspiration')
# plt.plot(expiration_df.index, expiration_df['duration'], 'b', label='Expiration')
plt.legend()
plt.show()

