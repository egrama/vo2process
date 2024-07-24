import sys
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/salavlad.csv'


csv_file = sys.argv[1] if len(sys.argv) > 1 else default_csv_file
df = pd.read_csv(csv_file, names=['ts', 'type', 'millis', 'dpIn','dpOut', 'o2'])
df.set_index('millis', inplace=True)
# Calculate the time difference between each row
df['millis_diff'] = df.index.to_series().diff()



def process_breaths(df):
    # Create boolean masks for in and out breaths
    in_breath = df['dpIn'] > 0
    out_breath = df['dpOut'] > 0

    # Create groups for consecutive breaths
    in_groups = (in_breath != in_breath.shift()).cumsum()
    out_groups = (out_breath != out_breath.shift()).cumsum()

    # Function to process each breath group
    def process_group(group):
        if len(group) > 3 and (group['dpIn'].sum() > 0 or group['dpOut'].sum() > 0):
            return pd.Series({
                'start': group.index[0],
                'stop': group.index[-1],
                'duration': len(group),
                'type': 'in' if group['dpIn'].sum() > 0 else 'out'
            })
        return None

    # Process in breaths
    in_results = df.groupby(in_groups).apply(process_group).dropna()
    in_results = in_results[in_results['type'] == 'in']

    # Process out breaths
    out_results = df.groupby(out_groups).apply(process_group).dropna()
    out_results = out_results[out_results['type'] == 'out']

    # Combine results
    all_breaths = pd.concat([in_results, out_results])
    
    # Reset index to remove the multi-level index
    all_breaths = all_breaths.reset_index(drop=True)
    
    # Sort by the 'start' column
    all_breaths = all_breaths.sort_values('start')

    return all_breaths

# Apply the function to your DataFrame
breath_data = process_breaths(df)

breath_data.set_index('start', inplace=True)
print(breath_data)


# plt.figure(figsize=(12,6))
# plt.scatter(breath_data.index, (breath_data['stop'] - breath_data.index ) / 1000, label='duration')
# plt.show()

# Calculate the breath duration
breath_data['breath_duration'] = breath_data['stop'] - breath_data.index

# Separate the 'in' and 'out' breaths
in_breaths = breath_data[breath_data['type'] == 'in']
out_breaths = breath_data[breath_data['type'] == 'out']

# Create the scatter plot
plt.figure(figsize=(12, 6))

# Plot 'in' breaths
plt.scatter(in_breaths.index, in_breaths['breath_duration'], color='blue', label='In', alpha=0.6)

# Plot 'out' breaths
plt.scatter(out_breaths.index, out_breaths['breath_duration'], color='red', label='Out', alpha=0.6)

# Customize the plot
plt.xlabel('Start Time')
plt.ylabel('Breath Duration')
plt.title('Breath Duration Over Time')
plt.legend()

# If your index is datetime, you might want to rotate the x-axis labels
plt.xticks(rotation=45)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Show the plot
plt.show()