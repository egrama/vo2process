import argparse
import logging
import sys
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from scipy.signal import savgol_filter
import json


# Weight of the subject in kg
weight = 78
# How many breaths to process in one group (inhale + exhale = 2 breaths)
step = 14
  #########
# Number of times to run a rolling average on the O2 signal
o2_smoothing_factor = 0;





# Air density constants
rhoSTPD = 1.292 # STPD conditions: density at 0°C, MSL, 1013.25 hPa, dry air (in kg/m3)
rhoBTPS = 1.123 # BTPS conditions: density at ambient  pressure, 35°C, 95% humidity
dry_constant = 287.058 
wet_constant = 461.495
# o2_max = 20.95  # % of O2 in air - we are using the maximum sensor measured value


# measured_temp_c = 25.3
# measured_humid_percent = 50
# measured_pressure_hPa = 100490

# Venturi tube areas
area_1 = 0.000531 # 26mm diameter (in m2) 
area_2 = 0.000314 # 20mm diameter (in m2)

# Volume correction factor (measured with calibration pump)
vol_corr = 0.8264

flow_sensor_threshold = 0.32

default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/1-rest-emil_961hPa_25g.csv'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/vlad_sala_2.csv'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/esala2_p1.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/outsideair.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/esala3_obo_temp.csv'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/newco2/esala4_nesomn_nov_15_24.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/newco2/catevaresp.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/newco2/10pompeout.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/newco2/inceputexhale.csv'
bike_file = default_csv_file.split('.')[0] + '.json'

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
  m_co2_out_stpd = 0
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
      m_co2_out_stpd += normalize_to_stpd(vol_out * row['mCo2'] / 100,  rhoOut)
  n2_in_stpd = normalize_to_stpd(vol_total_in , rhoIn)*0.9996 - o2_in_stpd 
  co2_out_stpd = normalize_to_stpd(vol_total_out, rhoOut) - o2_out_stpd - n2_in_stpd
  return vol_total_in, vol_total_out, o2_in_stpd, o2_out_stpd, co2_out_stpd, m_co2_out_stpd


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


def import_technogym(file):
  # File structure is described in technogym_export_structure.txt
  with open (file) as f:
    data = json.load(f)
  hr_data = data['data']['analitics']['hr']
  hr_df = pd.DataFrame(hr_data)

  equipment_data = data['data']['analitics']['samples']
  flattened_samples = []
  for sample in equipment_data:
    flattened_sample = sample['vs'] + [sample['t']]
    flattened_samples.append(flattened_sample)
  quipment_df = pd.DataFrame(flattened_samples, columns=['power', 'rpm', 'distance', 'level', 't'])
  merged_df = pd.merge(quipment_df, hr_df, on='t')
  merged_df.set_index('t', inplace=True)
  return merged_df

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
            if len(fields) == 8 and fields[1].strip() == 'TTPPH':
                part1.append(line)
            elif len(fields) == 10 and fields[1].strip() == 'mppoctht':
                in_part1 = False
                part2.append(line)
            else:
                raise ValueError(f"Line {line_number} has {len(fields)} fields, \
                                 expected 8 for first section and 10 for the second")
        else:
            if len(fields) == 10:
                part2.append(line)
            else:
                raise ValueError(f"Line {line_number} has {len(fields)} fields, expected 7 for second section")
    
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
    if title =='mRER':
      plt.axhline(y=0.85, color='gray', linestyle='--')
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

  plt.figure(figsize=(8,4))
  plt.plot(df.index, df['o2ini'], 'r',  label='O2Initial')
#   plt.plot(df.index, df['dpIn'], 'r',  label='dpIn')
#   plt.plot(df.index, df['dpOut'], 'b',  label='dpOut')
  # plt.plot(df.index, df['millis_diff'], 'g',  label='millis')

  # plt.plot(df.index, df['o2'], 'g',  label='O2')

  print(df.dpIn.sum())
  print(df.dpOut.sum())

  plt.show()

  plt.figure(figsize=(8,4))
  plt.plot(df.index, df['o2'], 'r',  label='O2')
  plt.show()


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

  bike_data = import_technogym('/Users/egrama/vo2max/vo2process/in_files/esala3.json')



  # Read the CSV file
  csv_file = args.csv_file
  part1, part2 = split_csv(csv_file)
  ambient_data = StringIO(''.join(part1))
  ambient_df = pd.read_csv(ambient_data,
                          names=['ts', 'intTemp', 'outTemp', 'intPressure', 'outPressure', 'humidity'],
                          dtype={'intTemp': np.float64, 'outTemp': np.float64, 'intPressure': np.float64, 'outPressure': np.float64, 'humidity': np.float64})
  ambient_df.set_index('ts', inplace=True)

  # Get tenmperature from the last entry in the ambient data
  measured_temp_c = ambient_df['outTemp'].iloc[-1]
  measured_humid_percent = ambient_df['humidity'].iloc[-1]
  measured_pressure_hPa = (ambient_df['outPressure'].iloc[-1] + ambient_df['intPressure'].iloc[-1])/2
                           
  csv_data = StringIO(''.join(part2))


  df = pd.read_csv(csv_data, 
                  names = ['ts', 'type', 'millis', 'dpIn', 'dpOut', 'o2', 'mCo2', 'co2Temp', 'co2Hum', 'intTemp'],
                  dtype={'millis': np.float64, 'dpIn': np.float64, 'dpOut': np.float64, 'o2': np.float64, 'mCo2': np.float64, 'co2Temp': np.float64, 'co2Hum': np.float64, 'intTemp': np.float64})
  df.set_index('millis', inplace=True)
  df['dpSum'] = 0
  # Sanitixe data
  # sanitize_data()

  # Calculate the time difference between each row
  df['millis_diff'] = df.index.to_series().diff()
  # Calculate the signed difference in pressure
  df['oneDp'] = df['dpIn'] - df['dpOut'] # signed diff pressure
  df['o2ini'] = df['o2']
  o2_max = df['o2'].max()


  # Define the rolling window size in seconds
  window_size_sec = 60
  #window_size_shift_ms = 30000 #ms
  window_size_shift_ms = 1000 #ms

  # Calculate rho values
  rho_in = calc_rho(measured_temp_c, measured_humid_percent, measured_pressure_hPa)
  rho_out = calc_rho(35, 95, measured_pressure_hPa)
  rho_btps = calc_rho(37, 100, measured_pressure_hPa)
  # rho_out = rho_btps



  df['BreathMarker'] = False
  computed_columns = ['volIn', 'volOut', 'o2InStpd', 'o2OutStpd', 'co2Stpd', 'mco2Stpd']
  df[computed_columns] = 0
  breath_indexes, breath_indexes_ms = find_breath_limits(df)
  df.loc[breath_indexes_ms, 'BreathMarker'] = True
  breath_mask = df['BreathMarker'] == True
  # Up until here we have computed the breath limits 
  # and marked them in the originaldataframe

  # Plot O2 before smoothing
  plt.ion()
  plt.figure(figsize=(8,4))
  plt.title('O2% initial')
  plt.plot(df.index, df['o2'], 'red',  label='oxy')


  # Smooth in place the O2 signal
  # df['o2'] = detect_and_flatten_spikes(df['o2'], window_size=258, spike_threshold=0.05).rolling(window=230, center=True).mean()
  while o2_smoothing_factor > 0:
    df['o2'] = df['o2'].rolling(window=430, center=True).mean()
    o2_smoothing_factor -= 1

  # Compute vol, o2, co2Stpd for each group of breaths
  breath_indexes.append(df.index[-1]) # to process the last section ???
  for i in range(step, len(breath_indexes_ms) - 1, step):
    df.loc[breath_indexes_ms[i-step], computed_columns] = calc_vol_o2(rho_in, rho_out, df.loc[breath_indexes_ms[i-step]:breath_indexes_ms[i]])

  # Take out results in separate dataframe
  rez_df = df[(df['BreathMarker'] == True) & (df['o2InStpd'] > 0)]
  breaths_df = df[df['BreathMarker'] == True]
  breaths_df['rf'] = 60000 / (breaths_df.index.astype(int).diff(periods=2)) # respiratory frequency (breaths per minute)

  # Compute vo2max and other physiological parameters
  rez_df['millis_diff'] = rez_df.index.to_series().diff()
  rez_df['vO2'] = (rez_df['o2InStpd'] - rez_df['o2OutStpd'])
  rez_df['vO2/min/kg'] = ((rez_df['o2InStpd'] - rez_df['o2OutStpd']) * 60000 / rez_df['millis_diff'])/weight
  rez_df['volDifStpd'] = normalize_to_stpd(rez_df['volIn'], rho_in) - normalize_to_stpd(rez_df['volOut'], rho_out)
  rez_df['volDifProc'] = rez_df['volDifStpd'] / (rez_df['volOut'] + 0.0001)
  rez_df['Ve_Vo2'] = rez_df['volOut'] / (rez_df['vO2'] + 0.000001)
  rez_df['Ve_Co2'] = rez_df['volOut'] / (rez_df['co2Stpd'] + 0.000001)
  rez_df['RER'] = rez_df['co2Stpd'] * (1 - rez_df['vO2/min/kg']/100 ) / (rez_df['o2InStpd'] - rez_df['o2OutStpd']) # added rough correction for bicarbonate buffering


  # Plot the results
  plt.figure(figsize=(8,4))
  plt.title('Differenial Pressure')
  plt.plot(df.index, df['oneDp'], 'black',  label='DP')
  plt.axhline(y=0, color='red', linestyle='--')
  plt.plot(df.index, df['o2ini'] -15, 'r',  label='O2Initial')
  plt.plot(df.index, df['mCo2'], 'blue',  label='co2')
  plt.plot(df.index, df['co2Temp']-20, 'magenta',  label='TempCo2')
  plt.plot(df.index, df['co2Hum']-80, 'cyan',  label='HumCo2')
  # plot a vertical line for each breath marker
  for i in range(0, len(breath_indexes_ms), 1):
    plt.axvline(x=breath_indexes_ms[i], color='gray', linestyle='--')

  plt.figure(figsize=(8,4))
  plt.title('CO2Stpd')
  plt.scatter(rez_df.index, rez_df['co2Stpd'], color='blue',  label='calculated_co2')
  plt.plot(rez_df.index, rez_df['mco2Stpd'], 'magenta',  label='measured_co2')
  plt.legend()


  # plt.figure(figsize=(8,4))
  # plt.title('O2%')
  # plt.plot(df.index, df['o2'], 'red',  label='oxy')


  # plt.figure(figsize=(8,4))
  # plt.title('RER')
  # plt.scatter(rez_df.index, rez_df['RER'], label='RER', color='red')
  # plt.axhline(y=0.5, color='gray', linestyle='--')
  # plt.axhline(y=1, color='gray', linestyle='--')
  # plt.axhline(y=1.5, color='gray', linestyle='--')
  # # plt.subplots_adjust(left=0.045, right=0.99, top=0.99, bottom=0.056)

  # plt.figure(figsize=(8,4))
  # plt.title('Ve_Co2')
  # plt.scatter(rez_df.index, rez_df['Ve_Co2'], label='Ve_Co2', color='black')
  # # plt.subplots_adjust(left=0.045, right=0.99, top=0.99, bottom=0.056)

  # plt.figure(figsize=(8,4))
  # plt.title('Ve_Vo2')
  # plt.scatter(rez_df.index, rez_df['Ve_Vo2'], label='Ve_Vo2', color='green')
  # # plt.subplots_adjust(left=0.045, right=0.99, top=0.99, bottom=0.056)

  # plt.figure(figsize=(8,4))
  # plt.title('Breath volume(ml)')
  # plt.scatter(rez_df.index, rez_df['volIn']/(step/2), label='Breath volume', color='blue')
  # # plt.subplots_adjust(left=0.045, right=0.99, top=0.99, bottom=0.056)
  # for i in range(1000, 5000, 250):
  #   plt.axhline(y=i, color='gray', linestyle='--')

  # plt.figure(figsize=(8,4))
  # plt.title('vO2/min/kg')
  # plt.scatter(rez_df.index, rez_df['vO2/min/kg'], label='vO2/min/kg')
  # # plt.subplots_adjust(left=0.045, right=0.99, top=0.99, bottom=0.056)
  # plt.axhline(y=45, color='gray', linestyle='--')
  # plt.axhline(y=50, color='gray', linestyle='--')  

  # filtered_temp = df[df['intTemp'] <= 60]
  # plt.figure(figsize=(8,4))
  # plt.title('IntTem/O2/C02conc')
  # plt.plot(filtered_temp.index, filtered_temp['intTemp'], label='intTemp', color='coral')
  # plt.plot(df.index, df['o2'], 'red',  label='oxy')
  # plt.plot(df.index, df['mCo2'], 'blue',  label='co2')


  # plt.subplots_adjust(left=0.045, right=0.99, top=0.99, bottom=0.056)


  plt.figure(figsize=(8,4))
  plt.title('HR and Power')
  plt.plot(bike_data.index, bike_data['power'], label='Power', color='grey')
  plt.plot(bike_data.index, bike_data['hr'], label='HR', color='red')
  plt.legend()
  plt.axhline(y=75, color='black', linestyle='--')
  plt.axhline(y=100, color='black', linestyle='--')
  plt.axhline(y=125, color='black', linestyle='--')


  plt.figure(figsize=(8,4))
  plt.plot(breaths_df.index, breaths_df['rf'].rolling(window=5).mean(), color='green',  label='RF')
  plt.title('Respiratory Frequency')


  # plt.ioff()
  # plt.show(block=True)

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
  ve_mco2_values = []
  vmco2_vo2_values = []

  for start_time, result in rolling_results:
      # TODO(): check if this is correct
      vol_o2_in_stpd = result[2]
      vol_o2_out_stpd = result[3]
      vol_co2 = result[4]
      vol_mco2 = result[5]
      vol_o2 = vol_o2_in_stpd - vol_o2_out_stpd  # O2 in - O2 out (normalized to STPD)
      # normalize vol_in to BTPS; result[0] * rho_in / rho_btps 
      ve_o2 = normalize_to_stpd(result[0], rho_in) / vol_o2
      ve_o2_out = result[1] * rho_out / rho_btps / vol_o2
      vol_in = normalize_to_stpd(result[0] / 1000.0,  rho_in)
      vol_out = normalize_to_stpd(result[1] / 1000.0,  rho_out)
      ve_m_co2 =  normalize_to_stpd(result[0], rho_in)  / vol_mco2


      window_minutes = window_size_sec / 60  # Convert window size to minutes
      vo2_per_minute = vol_o2 / window_minutes

      start_times.append(start_time)
      vi_vo2_values.append(vol_in / vol_out)
      vo2_values.append(vo2_per_minute / 80)  
      ve_o2_values.append(ve_o2)
      ve_o2_out_values.append(ve_o2_out)
      vol_in_values.append(vol_in)
      vol_out_values.append(vol_out)
      co2_values.append(vol_co2)
      vco2_vo2_values.append(vol_co2 / vol_o2)
      vmco2_vo2_values.append(vol_mco2 / vol_o2)
      ve_mco2_values.append(ve_m_co2)

      if vo2_per_minute > max_vo2:
        max_vo2 = vo2_per_minute
        max_vo2_start_time = start_time

  plt.ion()

  # plot_time_series(vi_vo2_values, start_times, 'VIn/VOut', 'VIn/VOut' )
  plot_time_series(vco2_vo2_values, start_times, 'RER', 'RER' )
  plot_time_series(vmco2_vo2_values, start_times, 'mRER', 'mRER' )
  
  plot_time_series(ve_mco2_values, start_times, 'VE_mCO2', 'VE_mCO2' )

  # plot_time_series(co2_values, start_times, 'CO2', 'CO2' )  
  plot_time_series_multi(start_times, 
                       ('vol_in', vol_in_values), 
                       ('vol_out', vol_out_values),
                       plot_fraction=1,
                       smoothing_window=10,
                       title='Volume In/Out Comparison')
  # plot_time_series(vol_in_values, start_times, 'Vol_in', 'Vol_in' )
  # plot_time_series(ve_o2_out_values, start_times, 'Ve_o2_out', 'Ve_o2_out')
  plot_time_series(ve_o2_values, start_times, 'Ve_o2', 'Ve_o2')
  # plot_time_series(vo2_values, start_times, 'VO2', 'VO2')
  print(f'Max VO2 (rolling, STPD): {round(max_vo2)} ml/min')
  print(f'Max VO2 segment start time: {max_vo2_start_time} ms')
  
  plt.ioff()
  plt.show(block=True)
  
  egr_original_plot(rho_in, rho_out, df)