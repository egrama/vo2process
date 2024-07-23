import sys
import pandas as pd
import math
import matplotlib.pyplot as plt

# Air density constants
rhoSTPD = 1.292 # STPD conditions: density at 0°C, MSL, 1013.25 hPa, dry air
rhoATPS = 1.225 # ATP conditions: density based on ambient conditions, dry air
rhoBTPS = 1.123 # BTPS conditions: density at ambient  pressure, 35°C, 95% humidity
dry_constant = 287.058 
wet_constant = 461.495

# Venturi tube areas
area_1 = 0.000531 # 26mm diameter
area_2 = 0.000314 # 20mm diameter

default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/1-rest-emil_961hPa_25g.csv'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/2-rest-vlad-24C_66hum_964hPa.csv'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/3-rest-emil-25.5C-88humquest-964hPa.csv'
default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/4-130bpm-25c-963hpa-77hum.csv'

def calc_volumetric_flow(diff_pressure, rho):
  # area_1 and area_2 are constants
  mass_flow = 1000 * math.sqrt((abs(diff_pressure) * 2 * rho) / ((1 / (area_2**2)) - (1 / (area_1**2))))  # mass/time     g/s
  return mass_flow / rho # volume/time


def calc_mass_flow(diff_pressure, rho):
  # area_1 and area_2 are constants
  mass_flow = 1000 * math.sqrt((abs(diff_pressure) * 2 * rho) / ((1 / (area_2**2)) - (1 / (area_1**2))))  # mass/time     g/s
  return mass_flow


def add_row(dataframe, millis, vol_in, vol_out, o2_in, o2_out, o2_diff):
  new_row = pd.DataFrame(({'vol_in': vol_in, 'vol_out': vol_out, 'o2_in': o2_in, 'o2_out': o2_out, 'o2_diff': o2_diff }), index=[millis])
  return pd.concat([dataframe, new_row])


def calc_vol_o2(rhoIn, rhoOut, dframe):
  vol_total_in = 0
  vol_total_out = 0
  o2_in_stpd = 0
  o2_out_stpd = 0
  mass_total_in = 0
  mass_total_out = 0
  mass_o2_in = 0
  mass_o2_out = 0


  # rez = pd.DataFrame(columns=['millis', 'vol_in', 'vol_out', 'o2_in', 'o2_out', 'o2_diff' ])
  # rez.set_index('millis', inplace=True)
  for millis, row in dframe.iterrows():
    if not (row['dpIn'] > 0 and row['dpOut'] > 0):
      if row['dpIn'] > 0:
        vol_in = calc_volumetric_flow(row['dpIn'], rhoIn) * row['millis_diff']
        vol_total_in += vol_in
        o2_in_stpd += vol_in * o2_max / 100  #* rhoIn / rhoSTPD
        # do for mass
        mass_in = calc_mass_flow(row['dpIn'], rhoIn) * row['millis_diff']
        mass_total_in += mass_in
        mass_o2_in += mass_in * o2_max
      if row['dpOut'] > 0:
        vol_out = calc_volumetric_flow(row['dpOut'], rhoOut) * row['millis_diff']
        vol_total_out += vol_out
        o2_out_stpd += vol_out * row['o2'] / 100 #* rhoOut / rhoSTPD
        # do for mass
        mass_out = calc_mass_flow(row['dpOut'], rhoOut) * row['millis_diff']
        mass_total_out += mass_out
        mass_o2_out += mass_out * row['o2']
      # rez = add_row(rez, millis, vol_in, vol_out, o2_in_stpd, o2_out_stpd, o2_in_stpd - o2_out_stpd)
  
  # plt.figure(figsize=(12,6))
  # plt.plot(rez.index, rez['o2_diff'], label='O2-diff')
  # plt.show()
  
  return vol_total_in, vol_total_out, o2_in_stpd, o2_out_stpd, mass_total_in, mass_total_out, mass_o2_in, mass_o2_out
  

def calc_rho(temp_c, humid, pressure):
    # Use simple Tetens equation
    p1 = 6.1078 * (10 ** (7.5 * temp_c / (temp_c + 237.3)))

    pv = humid * p1
    pd = pressure - pv
    temp_k = temp_c + 273.15

    # Assuming dry_constant and wet_constant are defined elsewhere
    rho = (pd / (dry_constant * temp_k)) + (pv / (wet_constant * temp_k))
    
    return rho


if __name__ == '__main__':
  #  csv file is the first argument, if empty, use default value
  csv_file = sys.argv[1] if len(sys.argv) > 1 else default_csv_file
  df = pd.read_csv(csv_file, names=['ts', 'type', 'millis', 'dpIn','dpOut', 'o2'])
  df.set_index('millis', inplace=True)
  # Calculate the time difference between each row
  df['millis_diff'] = df.index.to_series().diff()
  o2_max = df['o2'].max()
  o2_max = 20.7

  result = calc_vol_o2(
    calc_rho(25, 80, 96000), 
    calc_rho(35, 95, 96000),
    df
    )
  print(result)
  vo2 = (result[2] - result[3]) # egr /1000

  mask = (df['dpIn'] > 0) | (df['dpOut'] > 0)
  minutes = (df[mask].last_valid_index() - df[mask].first_valid_index()) /1000 /60
  print('Minutes:', minutes)
  print(f'VO2: {round(vo2/minutes)} ml/min')


# Calculate VO2 from mass flow
  mass_o2 = (result[6] - result[7]) # in g
  print(f'VO2_from_mass: {round(mass_o2/rhoSTPD/1000)}')


  # plt.figure(figsize=(12,6))
  # plt.plot(df.index, df['millis_diff'], label='M-diff')
  # plt.show()