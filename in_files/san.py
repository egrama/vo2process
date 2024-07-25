import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import sys

# Your CSV data
csv_data = """14:49:00.060 , Vol+O2, 699616.00, 0.00, 0.00, 15.08
14:49:00.091 , Vol+O2, 699616.00, 4.12, 0.00, 15.08
14:49:00.122 , Vol+O2, 699653.00, 3.53, 0.00, 15.08
14:49:00.122 , Vol+O2, 699690.00, 3.70, 0.00, 15.08"""


# Read the CSV data
#df = pd.read_csv(StringIO(csv_data), header=None, names=['Timestamp', 'Type', 'Field3', 'Field4', 'Field5', 'Field6'])
df = pd.read_csv(sys.argv[1], header=None, names=['Timestamp', 'Type', 'Field3', 'Field4', 'Field5', 'Field6'])

df['Timestamp'] = df['Timestamp'].str.strip()
# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S.%f')

# Create subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

# Plot Field4
ax1.plot(df['Timestamp'], df['Field4'], 'o-')
ax1.set_title('Field 4 Values Over Time')
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Field 4 Value')

# Plot Field5
ax2.plot(df['Timestamp'], df['Field5'], 'o-')
ax2.set_title('Field 5 Values Over Time')
ax2.set_xlabel('Timestamp')
ax2.set_ylabel('Field 5 Value')

# Plot Field6
ax3.plot(df['Timestamp'], df['Field6'], 'o-')
ax3.set_title('Field 6 Values Over Time')
ax3.set_xlabel('Timestamp')
ax3.set_ylabel('Field 6 Value')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
