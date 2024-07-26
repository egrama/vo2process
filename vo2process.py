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
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/fewoutbreathsrest.csv'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/vlad_sala_1.csv'


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


def add_row(dataframe, millis, vol_in, vol_out, o2_in, o2_out, o2_diff):
  new_row = pd.DataFrame(({'vol_in': vol_in, 'vol_out': vol_out, 'o2_in': o2_in, 'o2_out': o2_out, 'o2_diff': o2_diff }), index=[millis])
  return pd.concat([dataframe, new_row])


def calc_vol_o2(rhoIn, rhoOut, dframe):
  logging.debug(f"calc_vol_o2 millis_start: {dframe.index.min()} millis_end: {dframe.index.max()}")
  vol_total_in = 0
  vol_total_out = 0
  o2_in_stpd = 0
  o2_out_stpd = 0

  debug_count = 0
  for millis, row in dframe.iterrows():
    if not (row['dpIn'] > 0 and row['dpOut'] > 0):
      if row['dpIn'] > 0:
        vol_in = calc_volumetric_flow_egr(row['dpIn'], rhoIn) * row['millis_diff']
        vol_total_in += vol_in
        o2_in_stpd += normalize_to_stpd(vol_in * o2_max / 100,  rhoIn)
      if row['dpOut'] > 0:
        vol_out = calc_volumetric_flow_egr(row['dpOut'], rhoOut) * row['millis_diff']
        vol_total_out += vol_out
        o2_out_stpd += normalize_to_stpd(vol_out * row['o2'] / 100,  rhoOut)
      else:
        debug_count += 1
    logging.debug(f'Ignored samples with both pressures positive: {debug_count}')
  vol_total_in_stpd =  normalize_to_stpd(vol_total_in,  rhoIn)  
  vol_total_out_stpd = normalize_to_stpd(vol_total_out, rhoOut)

  return vol_total_in, vol_total_out, o2_in_stpd, o2_out_stpd


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
def plot_full_file_results(rho_in, rho_out, df):
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

  print(f'VolIn/Volout:  {result[0]/result[1]}')

  # print(df['dpIn'].max())
  # print(df['dpOut'].max())
  # print(df['millis_diff'].max())
  # print(df['o2'].max())
  # print(df.index.max())
  # max_index = df['millis_diff'].idxmax()

  plt.figure(figsize=(12,6))
  #plt.plot(df.index - 204206, df['dpIn'], 'r',  label='dpIn')
  #plt.plot(df.index - 204206, df['dpOut'], 'b',  label='dpOut')
  plt.plot(df.index - 204206, df['millis_diff'], 'g',  label='millis')

  plt.plot(df.index - 204206, df['o2'], 'g',  label='O2')
  plt.show()


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
  # Calculate the time difference between each row
  df['millis_diff'] = df.index.to_series().diff()
  # o2_max = df['o2'].max()

  # TODO: clean this up; Select only first 14 mins
  # Select the first 14 minutes of data
  start_time = df.index.min()
  end_time = start_time + (14 * 60 * 1000)  # 14 minutes in milliseconds
  df_14min = df.loc[start_time:end_time]
  df = df_14min


  # Define the rolling window size in seconds
  window_size_sec = 60
  #window_size_shift_ms = 30000 #ms
  window_size_shift_ms = 2000 #ms

  # Calculate rho values
  rho_in = calc_rho(27, 65, 96162)
  rho_out = calc_rho(35, 95, 96662)
  rho_test = calc_rho(37, 99,101325)
  rho_btps = calc_rho(37, 100, 96162)

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
  for start_time, result in rolling_results:
      # TODO(): check if this is correct
      vol_o2_in_stpd = result[2]
      vol_o2_out_stpd = result[3]
      vo2 = vol_o2_in_stpd - vol_o2_out_stpd  # O2 in - O2 out (normalized to STPD)
      # normalize vol_in to BTPS; result[0] * rho_in / rho_btps 
      ve_o2 = result[0] * rho_in / rho_btps / vo2
      ve_o2_out = result[1] * rho_out / rho_btps / vo2
      vol_in = result[0] / 1000.0
      vol_out = result[1] / 1000.0

      window_minutes = window_size_sec / 60  # Convert window size to minutes
      vo2_per_minute = vo2 / window_minutes

      start_times.append(start_time)
      vo2_values.append(vo2_per_minute / 80)  # Divide by 80 as requested
      ve_o2_values.append(ve_o2)
      ve_o2_out_values.append(ve_o2_out)
      vol_in_values.append(vol_in)
      vol_out_values.append(vol_out)

      if vo2_per_minute > max_vo2:
        max_vo2 = vo2_per_minute
        max_vo2_start_time = start_time
 
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
  
  plot_full_file_results(rho_in, rho_out, df)