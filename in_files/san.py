import pandas as pd
import matplotlib.pyplot as plt 
from io import StringIO
import sys 
import numpy as np

# Read the CSV data
df = pd.read_csv(sys.argv[1], header=None, names=['Timestamp', 'Type', 'Field3', 'Field4', 'Field5', 'Field6'])

# df['Timestamp'] = df['Timestamp'].str.strip()
# Convert timestamp to datetime
# df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')

# Function to detect outliers using IQR method
def detect_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

# Detect and print outliers for Fields 3, 4, 5, and 6
fields_to_check = ['Field3', 'Field4', 'Field5', 'Field6']

for field in fields_to_check:
    outliers = detect_outliers(df[field])
    if outliers.any():
        print(f"\nOutliers detected in {field}:")
        print(df[outliers][['Timestamp', field]])

# Create subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))

# Plot Field3
ax1.plot(df['Timestamp'], df['Field3'], 'o-')
ax1.set_title('Field 3 Values Over Time')
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Field 3 Value')

# Plot Field4
ax2.plot(df['Timestamp'], df['Field4'], 'o-')
ax2.set_title('Field 4 Values Over Time')
ax2.set_xlabel('Timestamp')
ax2.set_ylabel('Field 4 Value')

# Plot Field5
ax3.plot(df['Timestamp'], df['Field5'], 'o-')
ax3.set_title('Field 5 Values Over Time')
ax3.set_xlabel('Timestamp')
ax3.set_ylabel('Field 5 Value')

# Plot Field6
ax4.plot(df['Timestamp'], df['Field6'], 'o-')
ax4.set_title('Field 6 Values Over Time')
ax4.set_xlabel('Timestamp')
ax4.set_ylabel('Field 6 Value')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
