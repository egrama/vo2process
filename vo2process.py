import argparse
import logging
import sys
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

# Air density constants
rhoSTPD = 1.292 # STPD conditions: density at 0°C, MSL, 1013.25 hPa, dry air (in kg/m3)
rhoATPS = 1.225 # ATP conditions: density based on ambient conditions, dry air
rhoBTPS = 1.123 # BTPS conditions: density at ambient  pressure, 35°C, 95% humidity
dry_constant = 287.058 
wet_constant = 461.495
o2_max = 20.84  # % of O2 in air

# Venturi tube areas
area_1 = 0.000531 # 26mm diameter (in m2) 
area_2 = 0.000314 # 20mm diameter (in m2)

# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/1-rest-emil_961hPa_25g.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/2-rest-vlad-24C_66hum_964hPa.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/3-rest-emil-25.5C-88humquest-964hPa.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/4-130bpm-25c-963hpa-77hum.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/pompa-eu.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/emil-rest-27g- 56hum-963atm.csv'
#default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/xaa'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/salavlad.csv'



def setup_logging(log_level):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')


def calc_volumetric_flow(diff_pressure, rho):
  # area_1 and area_2 are constants
  mass_flow = 1000 * math.sqrt((abs(diff_pressure) * 2 * rho) / ((1 / (area_2**2)) - (1 / (area_1**2))))  
  return mass_flow / rho # volume/time


def calc_mass_flow(diff_pressure, rho):
  # area_1 and area_2 are constants
  mass_flow = 1000 * math.sqrt((abs(diff_pressure) * 2 * rho) / ((1 / (area_2**2)) - (1 / (area_1**2)))) 
  return mass_flow


def add_row(dataframe, millis, vol_in, vol_out, o2_in, o2_out, o2_diff):
  new_row = pd.DataFrame(({'vol_in': vol_in, 'vol_out': vol_out, 'o2_in': o2_in, 'o2_out': o2_out, 'o2_diff': o2_diff }), index=[millis])
  return pd.concat([dataframe, new_row])


def calc_vol_o2(rhoIn, rhoOut, dframe):
  logging.debug(f"calc_vol_o2 millis_start: {dframe.index.min()} millis_end: {dframe.index.max()}")
  vol_total_in = 0
  vol_total_out = 0
  o2_in_stpd = 0
  o2_out_stpd = 0
  mass_total_in = 0
  mass_total_out = 0
  mass_o2_in = 0
  mass_o2_out = 0

  debug_count = 0
  
  # rez = pd.DataFrame(columns=['millis', 'vol_in', 'vol_out', 'o2_in', 'o2_out', 'o2_diff' ])
  # rez.set_index('millis', inplace=True)
  for millis, row in dframe.iterrows():
    if not (row['dpIn'] > 0 and row['dpOut'] > 0):
      if row['dpIn'] > 0:
        vol_in = calc_volumetric_flow(row['dpIn'], rhoIn) * row['millis_diff']
        vol_total_in += vol_in
        o2_in_stpd += normalize_to_stpd(vol_in * o2_max / 100,  rhoIn)
        # do for mass
        mass_in = calc_mass_flow(row['dpIn'], rhoIn) * row['millis_diff'] / 1000
        mass_total_in += mass_in
        mass_o2_in += mass_in * o2_max / 100
      if row['dpOut'] > 0:
        vol_out = calc_volumetric_flow(row['dpOut'], rhoOut) * row['millis_diff']
        vol_total_out += vol_out
        o2_out_stpd += normalize_to_stpd(vol_out * row['o2'] / 100,  rhoOut)
        # do for mass
        mass_out = calc_mass_flow(row['dpOut'], rhoOut) * row['millis_diff'] / 1000
        mass_total_out += mass_out
        mass_o2_out += mass_out * row['o2'] / 100
      else:
        debug_count += 1
    logging.debug(f'Ignored samples with both pressures positive: {debug_count}')
      # rez = add_row(rez, millis, vol_in, vol_out, o2_in_stpd, o2_out_stpd, o2_in_stpd - o2_out_stpd)
  
  # plt.figure(figsize=(12,6))
  # plt.plot(rez.index, rez['o2_diff'], label='O2-diff')
  # plt.show()
  
  return vol_total_in, vol_total_out, o2_in_stpd, o2_out_stpd, mass_total_in, mass_total_out, mass_o2_in, mass_o2_out
  

# STPD normalization
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
        
        start_time += 1000  # Move the window by 1 second (1000 milliseconds)
    
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('csv_file', nargs='?', default=default_csv_file)
  parser.add_argument('--log', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Set the logging level (default: info)')
  args = parser.parse_args()

  setup_logging(args.log)
  csv_file = args.csv_file

  # Split the CSV file
  # TODO: parse first section for ambient parameters
  _, part2 = split_csv(csv_file)
  csv_data = StringIO(''.join(part2))
  df = pd.read_csv(csv_data, 
                  names=['ts', 'type', 'millis', 'dpIn', 'dpOut', 'o2'],
                  dtype={'millis': np.float64, 'dpIn': np.float64, 'dpOut': np.float64, 'o2': np.float64})
  df.set_index('millis', inplace=True)
  # Calculate the time difference between each row
  df['millis_diff'] = df.index.to_series().diff()
  # o2_max = df['o2'].max()

  # Define the rolling window size in seconds
  window_size_sec = 30

  # Calculate rho values
  rho_in = calc_rho(27, 54, 96662)
  rho_out = calc_rho(35, 95, 96662)

  # Use rolling_window function with calc_vol_o2 and correct arguments
  rolling_results = rolling_window(df, window_size_sec, calc_vol_o2, rho_in, rho_out)
  logging.info(f"Number of entries in rolling_results: {len(rolling_results)}")
 
  # Parameters
  plot_fraction = 0.99  # Plot the last quarter of the data
  smoothing_window = 10  # Number of points for moving average

  # Process rolling results
  max_vo2 = 0
  max_vo2_start_time = None
  start_times = []
  vo2_values = []
  for start_time, result in rolling_results:
      # TODO(): check if this is correct
      vol_o2_in_stpd = result[2]
      vol_o2_out_stpd = result[3]
      vo2 = vol_o2_in_stpd - vol_o2_out_stpd  # O2 in - O2 out (normalized to STPD)

      window_minutes = window_size_sec / 60  # Convert window size to minutes
      vo2_per_minute = vo2 / window_minutes

      start_times.append(start_time)
      vo2_values.append(vo2_per_minute / 80)  # Divide by 80 as requested

      if vo2_per_minute > max_vo2:
        max_vo2 = vo2_per_minute
        max_vo2_start_time = start_time

   # Calculate the start index for plotting
  plot_start_index = int(len(start_times) * (1 - plot_fraction))

  # Apply moving average smoothing
  vo2_values_smooth = pd.Series(vo2_values).rolling(window=smoothing_window, center=True).mean()

  # Convert start_times to seconds relative to the first timestamp
  start_times_sec = [(t - start_times[0]) / 1000 for t in start_times]

  # Plot vo2_per_minute / 80 against start_time
  plt.figure(figsize=(12, 6))
  plt.plot(start_times[plot_start_index:], vo2_values[plot_start_index:], label='Raw data', alpha=0.5)
  plt.plot(start_times[plot_start_index:], vo2_values_smooth[plot_start_index:], label='Smoothed', color='red')

 # Set x-axis ticks and labels
  plot_start_time = start_times_sec[plot_start_index]
  plot_end_time = start_times_sec[-1]
  tick_interval = 60  # 60 seconds between ticks
  
  xticks = np.arange(
      math.ceil(plot_start_time / tick_interval) * tick_interval,
      plot_end_time,
      tick_interval
  )
  #plt.xticks(xticks, [f'{int(x/60)}:{int(x%60):02d}' for x in xticks])

  plt.xlabel('Time (mm:ss)')
  plt.ylabel('VO2 per minute / 80 (ml/min)')
  plt.title(f'VO2 per minute / 80 over time (last {plot_fraction*100}%)')
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.show()

  print(f'Max VO2 (rolling, STPD): {round(max_vo2)} ml/min')
  print(f'Max VO2 segment start time: {max_vo2_start_time} ms')
  
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
  print(f'VO2: {round(vo2/minutes)} ml/min')


# Calculate VO2 from mass flow
  mass_o2 = (result[6] - result[7]) 
  print(f'VO2_from_mass: {mass_o2/rhoSTPD/minutes}')

  print(f'VolIn/Volout:  {result[0]/result[1]}')
  print(f'MassIn/MassOut:  {result[4]/result[5]}')

  print(df['dpIn'].max())
  print(df['dpOut'].max())
  print(df['millis_diff'].max())
  print(df['o2'].max())
  print(df.index.max())
  max_index = df['millis_diff'].idxmax()

  plt.figure(figsize=(12,6))
  plt.plot(df.index - 204206, df['dpIn'], 'r',  label='dpIn')
  plt.plot(df.index - 204206, df['dpOut'], 'b',  label='dpOut')
  # plt.plot(df.index - 204206, df['o2'], 'g',  label='O2')
  plt.show()