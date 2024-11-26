import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker

# Sample Data Creation
# Time in 10-second intervals for 20 minutes (1200 seconds)
time = pd.to_timedelta(np.arange(0, 1200, 10), unit='s')
num_samples = len(time)




# Fix: Create Target array of the correct length
target_values = [90, 130, 170, 210, 250, 290]
target = np.repeat(target_values, num_samples // len(target_values))


df = pd.DataFrame({
    'Time': time,
    'Target': target, # Use corrected target array
    'Power': np.linspace(80, 300, num_samples) + np.random.normal(0, 10, num_samples),
    'VO2': np.linspace(6, 70, num_samples) + np.random.normal(0, 3, num_samples),
    'HR': np.linspace(100, 180, num_samples) + np.random.normal(0, 5, num_samples),
    'VE': np.linspace(30, 150, num_samples) + np.random.normal(0, 10, num_samples),
    'RF': np.linspace(20, 60, num_samples) + np.random.normal(0, 3, num_samples),
    'TV': np.linspace(0.8, 2.5, num_samples) + np.random.normal(0, 0.1, num_samples)
}).set_index('Time')





# Threshold values (replace with your actual values)
vt1 = 138  # Example VT1 value
vt2 = 163  # Example VT2 value

# Zones - in seconds
warmup_end = 300
test_end = 900
cooldown_end = 1200

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Metabolism
ax1 = axes[0]

ax1.plot(df.index, df['Target'], label='Target[W]', color='gray', linewidth=3)
ax1.plot(df.index, df['Power'], label='Power[W]', color='firebrick')
ax1.plot(df.index, df['VO2'], label='VO2[mL/kg/min]', color='tab:blue')
ax1.set_ylabel("Metabolism",rotation=0, ha='right')

ax2 = ax1.twinx()
ax2.plot(df.index, df['HR'], label='HR[bpm]', color='deeppink')



lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# Annotations and Threshold lines
ax2.axhline(vt1, color='firebrick', linestyle='--', linewidth=0.8)
ax2.axhline(vt2, color='firebrick', linestyle='--', linewidth=0.8)
ax2.text(df.index[0], vt1 + 5, f"Ventilatory Threshold 1 (VT1) = {vt1}bpm", va='bottom', ha='left')
ax2.text(df.index[0], vt2 + 5, f"Ventilatory Threshold 2 (VT2) = {vt2}bpm", va='bottom', ha='left')



# Plot 2: Ventilation
ax3= axes[1]
ax3.plot(df.index, df['Target'], label='Target[W]', color='gray', linewidth=3)
ax3.plot(df.index, df['VE'], label='Ve[L/min]', color='seagreen')
ax3.plot(df.index, df['RF']* 5, label='Rf[bpm]', color='c') # multiply by 5 just to visualize
ax3.set_ylabel("Ventilation", rotation=0, ha='right')


ax4 = ax3.twinx()
ax4.plot(df.index, df['TV'] * 100, label='Tv[L]', color='goldenrod') # multiply by 100 just to visualize

lines, labels = ax3.get_legend_handles_labels()
lines2, labels2 = ax4.get_legend_handles_labels()
ax4.legend(lines + lines2, labels + labels2, loc='upper right')

# Zones - Convert Timedelta to seconds
warmup_end_sec = warmup_end  # warmup_end is already in seconds
test_end_sec = test_end    # test_end is already in seconds
cooldown_end_sec = cooldown_end # cooldown_end is already in seconds

for ax in [ax1, ax2, ax3, ax4]:
    # Convert TimedeltaIndex to seconds for axvspan
    times_seconds = df.index.total_seconds()

    ax.axvspan(times_seconds[0], warmup_end_sec, facecolor='lightgrey', alpha=0.5)
    ax.axvspan(warmup_end_sec, test_end_sec, facecolor='lightgrey', alpha=0.5)
    ax.axvspan(test_end_sec, cooldown_end_sec, facecolor='lightgrey', alpha=0.5)


    # Annotations - use seconds for positioning
    ax.text(warmup_end_sec / 2, ax.get_ylim()[0] * 1.1, "Warm-Up", ha='center')
    ax.text((warmup_end_sec + test_end_sec) / 2, ax.get_ylim()[0] * 1.1, "Test", ha='center')
    ax.text((test_end_sec + cooldown_end_sec) / 2, ax.get_ylim()[0] * 1.1, "Cool-Down", ha='center')

# Format x-axis ticks as strings (minutes:seconds)
for ax in [ax1, ax3]:
    ax.xaxis.set_major_locator(mticker.MultipleLocator(60)) 

plt.tight_layout()
plt.show()


plt.show()
