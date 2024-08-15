import argparse
import logging
import sys
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from scipy.signal import savgol_filter


# Air density constants
rhoSTPD = 1.292 # STPD conditions: density at 0°C, MSL, 1013.25 hPa, dry air (in kg/m3)
rhoATPS = 1.225 # ATP conditions: density based on ambient conditions, dry air
rhoBTPS = 1.123 # BTPS conditions: density at ambient  pressure, 35°C, 95% humidity
dry_constant = 287.058 
wet_constant = 461.495
o2_max = 21.20  # % of O2 in air

# Venturi tube areas
area_1 = 0.000531 # 26mm diameter (in m2) 
area_2 = 0.000314 # 20mm diameter (in m2)

# Volume correction factor (measured with calibration pump)
vol_corr = 0.8264
vol_out_corr = 0.995

flow_sensor_threshold = 0.2

default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/1-rest-emil_961hPa_25g.csv'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/vlad_sala_2.csv'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/esala2_p1.csv'

def setup_logging(log_level):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')


def calc_volumetric_flow(diff_pressure, rho):
  # area_1 and area_2 are constants
  mass_flow = 1000 * math.sqrt((abs(diff_pressure) * 2 * rho) / ((1 / (area_2**2)) - (1 / (area_1**2))))  
  return mass_flow / rho # volume/time
def calc_volumetric_flow_bot(delta_p, fluid_density):
    # Calculate the area ratio
    beta = area_2 / area_1
    
    # Calculate the discharge coefficient (Cd)
    # This is an approximation; for more accuracy, you might need to use empirical data
    Cd = 1
    
    # Calculate the mass flow rate
    numerator = 2 * delta_p * fluid_density
    denominator = 1 - beta**4
    
    mass_flow = Cd * area_2 * math.sqrt(numerator / denominator) * 1000
    
    # Calculate the volumetric flow rate
    volumetric_flow = mass_flow / fluid_density
    
    return volumetric_flow
def calc_volumetric_flow_egr(dp, rho):
  Q = 1000 * area_2 * math.sqrt(  (2 * dp) / (rho * ( 1 - (area_2 / area_1)**2 )))
  return Q
def calc_mass_flow(diff_pressure, rho):
  # area_1 and area_2 are constants
  mass_flow = 1000 * math.sqrt((abs(diff_pressure) * 2 * rho) / ((1 / (area_2**2)) - (1 / (area_1**2)))) 
  return mass_flow


def calc_vol_o2(rhoIn, rhoOut, dframe):
  logging.debug(f"calc_vol_o2 millis_start: {dframe.index.min()} millis_end: {dframe.index.max()}")
  vol_total_in = 0
  vol_total_out = 0
  o2_in_stpd = 0
  o2_out_stpd = 0
  in_pressure = 0
  out_pressure = 0
  


  for millis, row in dframe.iterrows():
    # Set to 0 negative pressure values and values lower than sensor threshold
    if row['dpIn'] < flow_sensor_threshold: # includes all negative values
      in_pressure = 0
    else:
      in_pressure = row['dpIn']
    if row['dpOut']  < flow_sensor_threshold:
      out_pressure = 0
    else:
      out_pressure = row['dpOut']


    # TODO - lowpri - don't compute escaped In air during obvious Out breaths 
    if in_pressure > 0:
      vol_in = calc_volumetric_flow_egr(in_pressure, rhoIn) * row['millis_diff'] * vol_corr
      vol_total_in += vol_in
      o2_in_stpd += normalize_to_stpd(vol_in * o2_max / 100,  rhoIn)
    if out_pressure > 0:
      vol_out = calc_volumetric_flow_egr(out_pressure, rhoOut) * row['millis_diff'] * vol_corr
      vol_total_out += vol_out
      o2_out_stpd += normalize_to_stpd(vol_out * row['o2'] / 100,  rhoOut)
  n2_in_stpd = normalize_to_stpd(vol_total_in , rhoIn) - o2_in_stpd
  co2_out_stpd = normalize_to_stpd(vol_total_out, rhoOut) - o2_out_stpd - n2_in_stpd
  return vol_total_in, vol_total_out, o2_in_stpd, o2_out_stpd, co2_out_stpd


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


def normalize_to_stpd(volume, rho_actual):
    return volume * rho_actual / rhoSTPD


def calc_rho(temp_c, humid, pressure):
    # Use simple Tetens equation
    p1 = 6.1078 * (10 ** (7.5 * temp_c / (temp_c + 237.3)))

    pv = humid * p1
    pd = pressure - pv
    temp_k = temp_c + 273.15

    # Assuming dry_constant and wet_constant are defined elsewhere
    rho = (pd / (dry_constant * temp_k)) + (pv / (wet_constant * temp_k))
    
    return rho


def rolling_window(df, window_size_sec, func, *args):
    results = []
    
    # Ensure the index is in milliseconds
    if df.index.name != 'millis':
        df = df.reset_index().set_index('millis')
    
    start_time = df.index.min()
    end_time = df.index.max()
    
    while start_time + window_size_sec * 1000 <= end_time:
        window_end = start_time + window_size_sec * 1000  # Convert seconds to milliseconds
        window_df = df[(df.index >= start_time) & (df.index < window_end)]
        
        if not window_df.empty:
            result = func(*(args + (window_df,)))
            results.append((start_time, result))
        
        start_time += window_size_shift_ms  # Move the window by it (befault 1000)
    
    return results


# Split CSV in 2 sections - ambient constants and flow entries
def split_csv(csv_file):
    with open(csv_file, 'r') as file:
        lines = file.readlines()
    
    part1 = []
    part2 = []
    in_part1 = True
    
    for line_number, line in enumerate(lines, 1):
        fields = line.strip().split(',')
        if in_part1:
            if len(fields) == 8:
                part1.append(line)
            elif len(fields) == 6:
                in_part1 = False
                part2.append(line)
            else:
                raise ValueError(f"Line {line_number} has {len(fields)} fields, expected 8 for first section")
        else:
            if len(fields) == 6:
                part2.append(line)
            else:
                raise ValueError(f"Line {line_number} has {len(fields)} fields, expected 6 for second section")
    
    if not part2:
        raise ValueError("Second section (with 6 fields) not found in CSV file")
    
    #logging.debug(f"Last line of part1: {part1[-1].strip()}")
    #logging.debug(f"First line of part2: {part2[0].strip()}")
    
    return part1, part2


def plot_time_series_multi(start_times, *y_data, plot_fraction=1, smoothing_window=10, title='', xlabel='Time (mm:ss)'):
    # Calculate the start index for plotting
    plot_start_index = int(len(start_times) * (1 - plot_fraction))

    # Convert start_times to seconds relative to the first timestamp
    start_times_sec = [(t - start_times[0]) / 1000 for t in start_times]

    # Plot data against start_time
    plt.figure(figsize=(12, 6))
    
    for label, y_values in y_data:
        # Apply moving average smoothing
        y_values_smooth = pd.Series(y_values).rolling(window=smoothing_window, center=True).mean()

        plt.plot(start_times_sec[plot_start_index:],
                 y_values[plot_start_index:], label=f'{label} (Raw)', alpha=0.5)
        plt.plot(start_times_sec[plot_start_index:],
                 y_values_smooth[plot_start_index:], label=f'{label} (Smoothed)')
    # Set x-axis ticks and labels
    plot_start_time = start_times_sec[plot_start_index]
    plot_end_time = start_times_sec[-1]
    tick_interval = 60  # 60 seconds between ticks
    
    xticks = np.arange(
        math.ceil(plot_start_time / tick_interval) * tick_interval,
        plot_end_time,
        tick_interval
    )
    plt.xticks(xticks, [f'{int(x/60)}:{int(x%60):02d}' for x in xticks])
    plt.xlabel(xlabel)
    plt.ylabel('Value')
    plt.title(f'{title} (last {plot_fraction*100}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_time_series(y_values, start_times, y_label, title, plot_fraction=1, smoothing_window=10):
    # Calculate the start index for plotting
    plot_start_index = int(len(start_times) * (1 - plot_fraction))

    # Apply moving average smoothing
    y_values_smooth = pd.Series(y_values).rolling(window=smoothing_window, center=True).mean()

    # Convert start_times to seconds relative to the first timestamp
    start_times_sec = [(t - start_times[0]) / 1000 for t in start_times]

    # Plot data against start_time
    plt.figure(figsize=(12, 6))
    plt.plot(start_times_sec[plot_start_index:],
             y_values[plot_start_index:], label='Raw data', alpha=0.5)
    plt.plot(start_times_sec[plot_start_index:],
             y_values_smooth[plot_start_index:], label='Smoothed', color='red')

    # Set x-axis ticks and labels
    plot_start_time = start_times_sec[plot_start_index]
    plot_end_time = start_times_sec[-1]
    tick_interval = 60  # 60 seconds between ticks
    
    xticks = np.arange(
        math.ceil(plot_start_time / tick_interval) * tick_interval,
        plot_end_time,
        tick_interval
    )
    plt.xticks(xticks, [f'{int(x/60)}:{int(x%60):02d}' for x in xticks])
    plt.xlabel('Time (mm:ss)')
    plt.ylabel(y_label)
    plt.title(f'{title} (last {plot_fraction*100}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Values and graph for the full file and period
def egr_original_plot(rho_in, rho_out, df):
  result = calc_vol_o2(
    rho_in, 
    rho_out,
    df
    )
  print(result)
  vo2 = (result[2] - result[3]) # egr /1000

  mask = (df['dpIn'] > 0) | (df['dpOut'] > 0)
  minutes = (df[mask].last_valid_index() - df[mask].first_valid_index()) /1000 /60
  print('Minutes:', minutes)
  # if vo2 is NaN, set it to 0
  if math.isnan(vo2):
    vo2 = 0
  print(f'VO2: {round(vo2/minutes)} ml/min')

  print(f'VolIn/Volout:  {result[0]/result[1]}')
  print(f'VolStpdIn/VolStpdOut:  {(result[0] *rho_in/rhoSTPD) /(result[1]*rho_out/rhoSTPD)}')

  # print(df['dpIn'].max())
  # print(df['dpOut'].max())
  # print(df['millis_diff'].max())
  # print(df['o2'].max())
  # print(df.index.max())
  # max_index = df['millis_diff'].idxmax()

  plt.figure(figsize=(12,6))
  plt.plot(df.index, df['o2ini'], 'r',  label='O2Initial')
#   plt.plot(df.index, df['dpIn'], 'r',  label='dpIn')
#   plt.plot(df.index, df['dpOut'], 'b',  label='dpOut')
  # plt.plot(df.index, df['millis_diff'], 'g',  label='millis')

  # plt.plot(df.index, df['o2'], 'g',  label='O2')

  print(df.dpIn.sum())
  print(df.dpOut.sum())

  plt.show()

  plt.figure(figsize=(12,6))
  plt.plot(df.index, df['o2'], 'r',  label='O2')
  plt.show()

import pandas as pd
import numpy as np

def detect_and_flatten_spikes(series, window_size=5, spike_threshold=3):
    # Calculate rolling median and standard deviation
    rolling_median = series.rolling(window=window_size, center=True).median()
    rolling_std = series.rolling(window=window_size, center=True).std()
    
    # Identify spikes
    spikes = np.abs(series - rolling_median) > (spike_threshold * rolling_std)
    
    # Create a new series for flattened data
    flattened = series.copy()
    
    # Get the index as a list for easier manipulation
    index_list = series.index.tolist()
    
    # Flatten spikes
    i = 0
    while i < len(series):
        if spikes.iloc[i]:
            # Find the end of the spike
            j = i
            while j < len(series) and spikes.iloc[j]:
                j += 1
            
            # Replace spike values with interpolated values
            if i > 0 and j < len(series):
                start_val = flattened.iloc[i-1]
                end_val = series.iloc[j]
                start_idx = index_list[i-1]
                end_idx = index_list[j]
                
                # Create a temporary Series for interpolation
                temp_series = pd.Series([start_val, end_val], index=[start_idx, end_idx])
                
                # Interpolate
                interpolated = temp_series.reindex(index_list[i-1:j+1]).interpolate()
                
                # Assign interpolated values
                flattened.iloc[i:j] = interpolated.iloc[1:-1]
            
            i = j
        else:
            i += 1
    
    return flattened


def sanitize_data():
   # Set to 0 negative pressure values and values lower than sensor threshold
  df.loc[df['dpIn'] < flow_sensor_threshold, 'dpIn'] = 0
  df.loc[df['dpOut'] < flow_sensor_threshold, 'dpOut'] = 0


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('csv_file', nargs='?', default=default_csv_file)
  parser.add_argument('--log', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Set the logging level (default: info)')
  args = parser.parse_args()
  setup_logging(args.log)

  # Read the CSV file
  csv_file = args.csv_file
  # TODO: parse first section for ambient parameters
  _, part2 = split_csv(csv_file)
  csv_data = StringIO(''.join(part2))
  df = pd.read_csv(csv_data, 
                  names=['ts', 'type', 'millis', 'dpIn', 'dpOut', 'o2'],
                  dtype={'millis': np.float64, 'dpIn': np.float64, 'dpOut': np.float64, 'o2': np.float64})
  df.set_index('millis', inplace=True)

  # Sanitixe data
  sanitize_data()




  # Calculate the time difference between each row
  df['millis_diff'] = df.index.to_series().diff()
  # Calculate the signed difference in pressure
  df['oneDp'] = df['dpIn'] - df['dpOut'] # signed diff pressure
  df['o2ini'] = df['o2']
  o2_max = df['o2'].max()




  # Define the rolling window size in seconds
  window_size_sec = 30
  #window_size_shift_ms = 30000 #ms
  window_size_shift_ms = 1000 #ms

  # Calculate rho values
  rho_in = calc_rho(24, 50, 100490)
  rho_out = calc_rho(35, 95, 100490)
  rho_btps = calc_rho(37, 100, 100490)

  df['BreathMarker'] = False
  computed_columns = ['volIn', 'volOut', 'o2InStpd', 'o2OutStpd', 'co2Stpd']
  df[computed_columns] = 0
  breath_indexes, breath_indexes_ms = find_breath_limits(df)
  df.loc[breath_indexes_ms, 'BreathMarker'] = True
  breath_mask = df['BreathMarker'] == True
  step = 2


  df['o2'] = detect_and_flatten_spikes(df['o2'], window_size=258, spike_threshold=0.05).rolling(window=230, center=True).mean()

  breath_indexes.append(df.index[-1]) # to process the last section ???
  for i in range(step, len(breath_indexes_ms) - 1, step):
    df.loc[breath_indexes_ms[i-step], computed_columns] = calc_vol_o2(rho_in, rho_out, df.loc[breath_indexes_ms[i-step]:breath_indexes_ms[i]])


  rez_df = df[(df['BreathMarker'] == True) & (df['o2InStpd'] > 0)]


  
  df['o2_flat'] = detect_and_flatten_spikes(df['o2'], window_size=258, spike_threshold=0.05)
  df['o2_roll'] = df['o2'].rolling(window=230, center=True).mean()

  plt.figure(figsize=(12,6))
  # plt.plot(df.index, df['o2'], 'red',  label='Oxy')
  plt.axhline(y=0, color='red', linestyle='--')
#   plt.plot(df.index, savgol_filter(df['o2'], window_length=31, polyorder=3), color='pink',  label='Savitzky-Golay')
#   plt.plot(df.index, df['o2_flat'], 'grey',  label='Oxyflat')
#   plt.plot(df.index, df['o2_roll'], 'blue',  label='Oxyroll')
  plt.plot(df.index, df['oneDp'], 'black',  label='DP')
  plt.show()
#   plt.plot(df.index, df['o2_flat'], 'r',  label='Oxy')
#   plt.plot(df.index, savgol_filter(df['o2'], window_length=31, polyorder=2), color='pink',  label='Savitzky-Golay')
#   plt.plot(rez_df.index, rez_df['o2'].ewm(span=21, adjust=False).mean(), 'cyan',  label='O2InStpd')
  plt.plot(rez_df.index, (rez_df['co2Stpd'] / (rez_df['o2InStpd'] - rez_df['o2OutStpd'])), label='RER', color='red')
  plt.axhline(y=0.5, color='gray', linestyle='--')
  plt.axhline(y=1, color='gray', linestyle='--')
  plt.axhline(y=1.5, color='gray', linestyle='--')
#   for i in range(0, 100, 25):
#     plt.axhline(y=i, color='gray', linestyle='--')
  plt.show()



  rez_df =  pd.DataFrame(columns=['millis', 'volIn', 'volOut', 'o2InStpd', 'o2OutStpd'])
  step = 10
  for i in range(0, len(breath_indexes) - 1, step):
    vi, vo, o2i, o2o, co2 = calc_vol_o2(rho_in, rho_out, df.loc[breath_indexes_ms[i-step]:breath_indexes_ms[i]])
    # TODO: only add volumes which are greater than a minimal breath volume
    new_row = pd.DataFrame({'millis': [breath_indexes_ms[i]], 'volIn': [vi], 'volOut': [vo], 'o2InStpd': [o2i], 'o2OutStpd': [o2o]})
    rez_df = pd.concat([rez_df, new_row], ignore_index=True)

  rez_df.set_index('millis', inplace=True)
  rez_df['millis_diff'] = rez_df.index.to_series().diff()
  rez_df['vO2'] = (rez_df['o2InStpd'] - rez_df['o2OutStpd'])
  rez_df['vO2/min/kg'] = ((rez_df['o2InStpd'] - rez_df['o2OutStpd']) * 60000 / rez_df['millis_diff'])/80
  rez_df['volDifStpd'] = normalize_to_stpd(rez_df['volIn'], rho_in) - normalize_to_stpd(rez_df['volOut'], rho_out)
  rez_df['volDifProc'] = rez_df['volDifStpd'] / (rez_df['volOut'] + 0.0001)
  rez_df['Ve_Vo2'] = rez_df['volOut'] / (rez_df['vO2'] + 0.0001)
  print(rez_df)
    
  plt.figure(figsize=(12,6))
  plt.scatter(rez_df.index, rez_df['vO2/min/kg'], label='vO2/min/kg')
#   plt.scatter(rez_df.index, rez_df['Ve_Vo2'], label='Ve_Vo2')
  plt.scatter(rez_df.index, rez_df['volIn'], label='volIn', color='blue')
#   plt.scatter(rez_df.index, rez_df['millis_diff'], label='millis_diff', color='pink')
#   plt.plot(rez_df.index, rez_df['volIn'], 'b', label='volIn')


  # Add horizontal lines every 10 ticks up to 70
  for i in range(0, 6000, 250):
    plt.axhline(y=i, color='gray', linestyle='--')
  plt.show()

  print('ha!')





  # Use rolling_window function with calc_vol_o2 and correct arguments
  rolling_results = rolling_window(df, window_size_sec, calc_vol_o2, rho_in, rho_out)

  # compute  vol_in / vo2
  logging.info(f"Number of entries in rolling_results: {len(rolling_results)}")
 
  # Process rolling results
  max_vo2 = 0
  max_vo2_start_time = None
  start_times = []
  vo2_values = []
  ve_o2_values = []
  ve_o2_out_values = []
  vol_in_values = []
  vol_out_values = []
  co2_values = []
  vco2_vo2_values = []
  vi_vo2_values = []
  for start_time, result in rolling_results:
      # TODO(): check if this is correct
      vol_o2_in_stpd = result[2]
      vol_o2_out_stpd = result[3]
      vol_co2 = result[4]
      vol_o2 = vol_o2_in_stpd - vol_o2_out_stpd  # O2 in - O2 out (normalized to STPD)
      # normalize vol_in to BTPS; result[0] * rho_in / rho_btps 
      ve_o2 = result[0] * rho_in / rho_btps / vol_o2
      ve_o2_out = result[1] * rho_out / rho_btps / vol_o2
      vol_in = normalize_to_stpd(result[0] / 1000.0,  rho_in)
      vol_out = normalize_to_stpd(result[1] / 1000.0,  rho_out)



      window_minutes = window_size_sec / 60  # Convert window size to minutes
      vo2_per_minute = vol_o2 / window_minutes

      start_times.append(start_time)
      vi_vo2_values.append(vol_in / vol_out)
      vo2_values.append(vo2_per_minute / 80)  # Divide by 80 as requested
      ve_o2_values.append(ve_o2)
      ve_o2_out_values.append(ve_o2_out)
      vol_in_values.append(vol_in)
      vol_out_values.append(vol_out)
      co2_values.append(vol_co2)
      vco2_vo2_values.append(vol_co2 / vol_o2)

      if vo2_per_minute > max_vo2:
        max_vo2 = vo2_per_minute
        max_vo2_start_time = start_time

  plt.ion()

  plot_time_series(vi_vo2_values, start_times, 'VIn/VOut', 'VIn/VOut' )
  plot_time_series(vco2_vo2_values, start_times, 'RER', 'RER' )

  plot_time_series(co2_values, start_times, 'CO2', 'CO2' )  
  plot_time_series_multi(start_times, 
                       ('vol_in', vol_in_values), 
                       ('vol_out', vol_out_values),
                       plot_fraction=1,
                       smoothing_window=10,
                       title='Volume In/Out Comparison')
  plot_time_series(vol_in_values, start_times, 'Vol_in', 'Vol_in' )
  plot_time_series(ve_o2_out_values, start_times, 'Ve_o2_out', 'Ve_o2_out')
  plot_time_series(ve_o2_values, start_times, 'Ve_o2', 'Ve_o2')
  plot_time_series(vo2_values, start_times, 'VO2', 'VO2')
  print(f'Max VO2 (rolling, STPD): {round(max_vo2)} ml/min')
  print(f'Max VO2 segment start time: {max_vo2_start_time} ms')
  
  plt.ioff()
  plt.show(block=True)
  
  egr_original_plot(rho_in, rho_out, df)