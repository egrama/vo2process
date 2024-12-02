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
from functools import partial
import matplotlib.dates as mdates


# Weight of the subject in kg
subject_weight = 78
# How many breaths to process in one group (inhale + exhale = 2 breaths)
step = 14
#########
# Number of times to run a rolling average on the O2 signal
o2_smoothing_factor = 0


# Air density constants
rhoSTPD = 1.292  # STPD conditions: density at 0°C, MSL, 1013.25 hPa, dry air (in kg/m3)
rhoBTPS = 1.123  # BTPS conditions: density at ambient  pressure, 35°C, 95% humidity
dry_constant = 287.058
wet_constant = 461.495
# o2_max = 20.95  # % of O2 in air - we are using the maximum sensor measured value


# measured_temp_c = 25.3
# measured_humid_percent = 50
# measured_pressure_hPa = 100490

# Venturi tube areas
area_1 = 0.000531  # 26mm diameter (in m2)
area_2 = 0.000314  # 20mm diameter (in m2)

# Volume correction factor (measured with calibration pump)
vol_corr = 0.8264

flow_sensor_threshold = 0.32

default_csv_file = "/Users/egrama/vo2max/vo2process/in_files/1-rest-emil_961hPa_25g.csv"
default_csv_file = "/Users/egrama/vo2max/vo2process/in_files/vlad_sala_2.csv"
default_csv_file = "/Users/egrama/vo2max/vo2process/in_files/esala2_p1.csv"
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/outsideair.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/old/esala3_obo_temp.csv'
default_csv_file = "/Users/egrama/vo2max/vo2process/in_files/newco2/esala4_nesomn_nov_15_24.csv"
default_csv_file = "/Users/egrama/vo2max/vo2process/in_files/newco2/esala5_run_dryair.csv"
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/newco2/catevaresp.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/newco2/10pompeout.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/newco2/inceputexhale.csv'
# default_csv_file = '/Users/egrama/vo2max/vo2process/in_files/newco2/asala1.csv'
equipment_file = default_csv_file.split(".")[0] + ".json"

plot_old_graphs = False


def setup_logging(log_level):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def calc_volumetric_flow(diff_pressure, rho):
    # area_1 and area_2 are constants
    mass_flow = 1000 * math.sqrt(
        (abs(diff_pressure) * 2 * rho) / ((1 / (area_2**2)) - (1 / (area_1**2)))
    )
    return mass_flow / rho  # volume/time


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
    Q = 1000 * area_2 * math.sqrt((2 * dp) / (rho * (1 - (area_2 / area_1) ** 2)))
    return Q


def calc_mass_flow(diff_pressure, rho):
    # area_1 and area_2 are constants
    mass_flow = 1000 * math.sqrt(
        (abs(diff_pressure) * 2 * rho) / ((1 / (area_2**2)) - (1 / (area_1**2)))
    )
    return mass_flow


def calc_vol_o2(rhoIn, rhoOut, dframe):
    logging.debug(
        f"calc_vol_o2 millis_start: {dframe.index.min()} millis_end: {dframe.index.max()}"
    )
    vol_total_in = 0
    vol_total_out = 0
    o2_in_stpd = 0
    o2_out_stpd = 0
    in_pressure = 0
    m_co2_out_stpd = 0
    out_pressure = 0

    for millis, row in dframe.iterrows():
        # Set to 0 negative pressure values and values lower than sensor threshold
        if row["dpIn"] < flow_sensor_threshold:  # includes all negative values
            in_pressure = 0
        else:
            in_pressure = row["dpIn"]
        if row["dpOut"] < flow_sensor_threshold:
            out_pressure = 0
        else:
            out_pressure = row["dpOut"]

        # TODO - lowpri - don't compute escaped In air during obvious Out breaths
        if in_pressure > 0:
            vol_in = (
                calc_volumetric_flow_egr(in_pressure, rhoIn)
                * row["millis_diff"]
                * vol_corr
            )
            vol_total_in += vol_in
            o2_in_stpd += normalize_to_stpd(vol_in * o2_max / 100, rhoIn)
        if out_pressure > 0:
            vol_out = (
                calc_volumetric_flow_egr(out_pressure, rhoOut)
                * row["millis_diff"]
                * vol_corr
            )
            vol_total_out += vol_out
            o2_out_stpd += normalize_to_stpd(vol_out * row["o2"] / 100, rhoOut)
            m_co2_out_stpd += normalize_to_stpd(vol_out * row["mCo2"] / 100, rhoOut)
    n2_in_stpd = normalize_to_stpd(vol_total_in, rhoIn) * 0.9996 - o2_in_stpd
    co2_out_stpd = normalize_to_stpd(vol_total_out, rhoOut) - o2_out_stpd - n2_in_stpd
    return (
        vol_total_in,
        vol_total_out,
        o2_in_stpd,
        o2_out_stpd,
        co2_out_stpd,
        m_co2_out_stpd,
    )


# similar to the old calc_vol_o2 but using pandas
def calc_volumes(row, rhoIn):
    computed_columns = [
        "volAirIn",
        "volAirInStpd",
        "volAirOut",
        "volAirOutStpd",
        "volO2InStpd",
        "volO2OutStpd",
        "volCo2OutStpd",
    ]
    if row["dpIn"] < flow_sensor_threshold:
        volAirIn = 0
        volAirInStpd = 0
        volO2InStpd = 0
    else:
        volAirIn = (
            calc_volumetric_flow_egr(row["dpIn"], rhoIn) * row["millis_diff"] * vol_corr
        )
        volAirInStpd = normalize_to_stpd(volAirIn, rhoIn)
        volO2InStpd = normalize_to_stpd(volAirIn * o2_max / 100, rhoIn)
    if row["dpOut"] < flow_sensor_threshold:
        volAirOut = 0
        volAirOutStpd = 0
        volO2OutStpd = 0
        volCo2OutStpd = 0
    else:
        # rhoOut = calc_rho(row["co2Temp"], row["co2Hum"], measured_pressure_hPa)
        volAirOut = (
            calc_volumetric_flow_egr(row["dpOut"], row["rhoOut"])
            * row["millis_diff"]
            * vol_corr
        )
        volAirOutStpd = normalize_to_stpd(volAirOut, row["rhoOut"])
        volO2OutStpd = volAirOutStpd * row["o2"] / 100
        volCo2OutStpd = volAirOutStpd * row["mCo2"] / 100
    return volAirIn, volAirInStpd, volAirOut, volAirOutStpd, volO2InStpd, volO2OutStpd, volCo2OutStpd


def find_breath_limits(df, sg_win_lenght=11, sg_polyorder=2):
    savgol_filtered = savgol_filter(
        df["oneDp"], window_length=sg_win_lenght, polyorder=sg_polyorder
    )
    # Find indexes where the line changes sign and crosses the x-axis
    indexes = [0]
    indexes_ms = [df.index[0]]
    for i in range(len(savgol_filtered) - 1):
        # only consider sign changes that have enough pressure (>40)
        # and are at least 20 samples apart (close to max there are valid breaths with less samples!)
        # and the sum of the pressure has a different sign from the previous one
        dpSum = df.loc[indexes_ms[-1] :].head(i - indexes[-1])["oneDp"].sum()
        if (
            (savgol_filtered[i] * savgol_filtered[i + 1] < 0)
            and (abs(dpSum) > 20)
            and ((i - indexes[-1] > 10) or abs(dpSum) > 200)
            and (dpSum * df.loc[indexes_ms[-1], "dpSum"] <= 0)
        ):
            df.loc[df.index[i], "dpSum"] = dpSum
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
    if df.index.name != "millis":
        df = df.reset_index().set_index("millis")

    start_time = df.index.min()
    end_time = df.index.max()

    while start_time + window_size_sec * 1000 <= end_time:
        window_end = (
            start_time + window_size_sec * 1000
        )  # Convert seconds to milliseconds
        window_df = df[(df.index >= start_time) & (df.index < window_end)]

        if not window_df.empty:
            result = func(*(args + (window_df,)))
            results.append((start_time, result))

        start_time += window_size_shift_ms  # Move the window by it (befault 1000)

    return results


def import_technogym(file):
    # File structure is described in technogym_export_structure.txt
    with open(file) as f:
        data = json.load(f)
    hr_data = data["data"]["analitics"]["hr"]
    hr_df = pd.DataFrame(hr_data)
    hr_df.set_index("t", inplace=True)
    if data['data']['equipmentType'] == 'UprightBike':
        headers = ["power", "rpm", "distance", "level", "t"]
    elif data['data']['equipmentType'] == 'Treadmill':
        headers = ["speed", "grade", "distance", "t"]
    else:
        sys.exit(f"Unsupported equipment type: data['data']['equipmentType']")
    equipment_data = data["data"]["analitics"]["samples"]

    equipment_data_df = pd.DataFrame([{**dict(zip(headers, d['vs'])), 't': d['t']} for d in equipment_data])
    equipment_data_df.set_index("t", inplace=True)
    merged_df = pd.merge(hr_df, equipment_data_df, left_index=True, right_index=True, how='outer')
    merged_df = merged_df.sort_index()
    to_fill = headers[:-1] + ['hr']
    merged_df[to_fill] = merged_df[to_fill].ffill()
    full_index=pd.RangeIndex(start=merged_df.index.min(), stop=merged_df.index.max() + 1)
    merged_df = merged_df.reindex(full_index) # some seconds are missing
    merged_df = merged_df.ffill()

    return merged_df


# Split CSV in 2 sections - ambient constants and flow entries
def split_csv(csv_file):
    with open(csv_file, "r") as file:
        lines = file.readlines()

    part1 = []
    part2 = []
    in_part1 = True

    for line_number, line in enumerate(lines, 1):
        fields = line.strip().split(",")
        if in_part1:
            if len(fields) == 8 and fields[1].strip() == "TTPPH":
                part1.append(line)
            elif len(fields) == 10 and fields[1].strip() == "mppoctht":
                in_part1 = False
                part2.append(line)
            else:
                raise ValueError(
                    f"Line {line_number} has {len(fields)} fields, \
                                 expected 8 for first section and 10 for the second"
                )
        else:
            if len(fields) == 10:
                part2.append(line)
            else:
                raise ValueError(
                    f"Line {line_number} has {len(fields)} fields, expected 7 for second section"
                )

    if not part2:
        raise ValueError("Second section (with 6 fields) not found in CSV file")

    # logging.debug(f"Last line of part1: {part1[-1].strip()}")
    # logging.debug(f"First line of part2: {part2[0].strip()}")

    return part1, part2


def plot_time_series_multi(
    start_times,
    *y_data,
    plot_fraction=1,
    smoothing_window=10,
    title="",
    xlabel="Time (mm:ss)",
):
    # Calculate the start index for plotting
    plot_start_index = int(len(start_times) * (1 - plot_fraction))

    # Convert start_times to seconds relative to the first timestamp
    start_times_sec = [(t - start_times[0]) / 1000 for t in start_times]

    # Plot data against start_time
    plt.figure(figsize=(12, 6))

    for label, y_values in y_data:
        # Apply moving average smoothing
        y_values_smooth = (
            pd.Series(y_values).rolling(window=smoothing_window, center=True).mean()
        )

        plt.plot(
            start_times_sec[plot_start_index:],
            y_values[plot_start_index:],
            label=f"{label} (Raw)",
            alpha=0.5,
        )
        plt.plot(
            start_times_sec[plot_start_index:],
            y_values_smooth[plot_start_index:],
            label=f"{label} (Smoothed)",
        )
    # Set x-axis ticks and labels
    plot_start_time = start_times_sec[plot_start_index]
    plot_end_time = start_times_sec[-1]
    tick_interval = 60  # 60 seconds between ticks

    xticks = np.arange(
        math.ceil(plot_start_time / tick_interval) * tick_interval,
        plot_end_time,
        tick_interval,
    )
    plt.xticks(xticks, [f"{int(x/60)}:{int(x%60):02d}" for x in xticks])
    plt.xlabel(xlabel)
    plt.ylabel("Value")
    plt.title(f"{title} (last {plot_fraction*100}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_time_series(
    y_values, start_times, y_label, title, plot_fraction=1, smoothing_window=10
):
    # Calculate the start index for plotting
    plot_start_index = int(len(start_times) * (1 - plot_fraction))

    # Apply moving average smoothing
    y_values_smooth = (
        pd.Series(y_values).rolling(window=smoothing_window, center=True).mean()
    )

    # Convert start_times to seconds relative to the first timestamp
    start_times_sec = [(t - start_times[0]) / 1000 for t in start_times]

    # Plot data against start_time
    plt.figure(figsize=(12, 6))
    plt.plot(
        start_times_sec[plot_start_index:],
        y_values[plot_start_index:],
        label="Raw data",
        alpha=0.5,
    )
    plt.plot(
        start_times_sec[plot_start_index:],
        y_values_smooth[plot_start_index:],
        label="Smoothed",
        color="red",
    )

    # Set x-axis ticks and labels
    plot_start_time = start_times_sec[plot_start_index]
    plot_end_time = start_times_sec[-1]
    tick_interval = 60  # 60 seconds between ticks

    xticks = np.arange(
        math.ceil(plot_start_time / tick_interval) * tick_interval,
        plot_end_time,
        tick_interval,
    )
    plt.xticks(xticks, [f"{int(x/60)}:{int(x%60):02d}" for x in xticks])
    plt.xlabel("Time (mm:ss)")
    plt.ylabel(y_label)
    plt.title(f"{title} (last {plot_fraction*100}%)")
    if title == "mRER":
        plt.axhline(y=0.85, color="gray", linestyle="--")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Values and graph for the full file and period
def egr_original_plot(rho_in, rho_out, df):
    result = calc_vol_o2(rho_in, rho_out, df)
    print(result)
    vo2 = result[2] - result[3]  # egr /1000

    mask = (df["dpIn"] > 0) | (df["dpOut"] > 0)
    minutes = (df[mask].last_valid_index() - df[mask].first_valid_index()) / 1000 / 60
    print("Minutes:", minutes)
    # if vo2 is NaN, set it to 0
    if math.isnan(vo2):
        vo2 = 0
    print(f"VO2: {round(vo2/minutes)} ml/min")

    print(f"volAirIn/volAirOut:  {result[0]/result[1]}")
    print(
        f"VolStpdIn/VolStpdOut:  {(result[0] *rho_in/rhoSTPD) /(result[1]*rho_out/rhoSTPD)}"
    )

    # print(df['dpIn'].max())
    # print(df['dpOut'].max())
    # print(df['millis_diff'].max())
    # print(df['o2'].max())
    # print(df.index.max())
    # max_index = df['millis_diff'].idxmax()

    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df["o2ini"], "r", label="O2Initial")
    #   plt.plot(df.index, df['dpIn'], 'r',  label='dpIn')
    #   plt.plot(df.index, df['dpOut'], 'b',  label='dpOut')
    # plt.plot(df.index, df['millis_diff'], 'g',  label='millis')

    # plt.plot(df.index, df['o2'], 'g',  label='O2')

    print(df.dpIn.sum())
    print(df.dpOut.sum())

    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df["o2"], "r", label="O2")
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
                start_val = flattened.iloc[i - 1]
                end_val = series.iloc[j]
                start_idx = index_list[i - 1]
                end_idx = index_list[j]

                # Create a temporary Series for interpolation
                temp_series = pd.Series(
                    [start_val, end_val], index=[start_idx, end_idx]
                )

                # Interpolate
                interpolated = temp_series.reindex(
                    index_list[i - 1 : j + 1]
                ).interpolate()

                # Assign interpolated values
                flattened.iloc[i:j] = interpolated.iloc[1:-1]

            i = j
        else:
            i += 1

    return flattened


def sanitize_data():
    # Set to 0 negative pressure values and values lower than sensor threshold
    df.loc[df["dpIn"] < flow_sensor_threshold, "dpIn"] = 0
    df.loc[df["dpOut"] < flow_sensor_threshold, "dpOut"] = 0


def our_plot(
    x,
    ydata,
    figsize=(12, 6),
    title="Graph",
    xlabel="Time (MM)",
    grid=True,
    gridalpha=0.5,
    gridaxis="both",
    millis=False
):
    y_defaults = {
        "window": 0,
        "label": 'some_data',
        "color": 'blue',
        "tick_interval": 0,
        "tick_color": 'gray',
        "method": 'plot'
    }
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend(loc="lower right")
    if grid:
        plt.grid(True, alpha=gridalpha, axis=gridaxis)
    if millis:
        minute = 60000
    else:
        minute = 60
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(minute)) # ticks every minute
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{int(x//minute):02}"))
    ax1 = plt.gca()
    axes = [ax1]
    for i, y in enumerate(ydata):
        y = {**y_defaults, **y}
        y_values = y['y']
        if y['window'] > 0:
            y_values = y_values.rolling(window=y['window'], center=True).mean()     
        if len(y_values) < x.size: # index of main df has more values than data to be
            new_index = pd.RangeIndex(start=0, stop=x.size)
            y_values = y_values.reindex(new_index, fill_value=np.nan) # fill with nan up to main df size
        args =[x, y_values]
        kwargs = {'label':y['label'], 'color':y['color']}
        if y['method'] == 'scatter':
            kwargs['s'] = 10
        if y['tick_interval'] ==  0:
            plot_function = getattr(plt, y['method'])
            plot_function(*args, **kwargs)
        else: # separate y axis with ticks
            ax = ax1.twinx()
            # ax.spines['right'].set_position(('outward', 60 * (i + 1)))
            ax.tick_params(axis='y', labelcolor=y["tick_color"])
            ax.yaxis.set_major_locator(plt.MultipleLocator(y['tick_interval']))
            plot_function = getattr(ax, y['method'])
            plot_function(*args, **kwargs)
            ax.grid(True, alpha=0.2, axis='y', color=y["color"]) 
            axes.append(ax)
    lines, labels = [], []
    for ax in axes:
        l, lab = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lab)
    ax1.legend(lines, labels, loc='upper left')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", nargs="?", default=default_csv_file)
    parser.add_argument(
        "--log",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level (default: info)",
    )
    args = parser.parse_args()
    setup_logging(args.log)

    equipment_data = import_technogym(equipment_file)

    # Read the CSV file
    csv_file = args.csv_file
    part1, part2 = split_csv(csv_file)
    ambient_data = StringIO("".join(part1))
    ambient_df = pd.read_csv(
        ambient_data,
        names=["ts", "intTemp", "outTemp", "intPressure", "outPressure", "humidity"],
        dtype={
            "intTemp": np.float64,
            "outTemp": np.float64,
            "intPressure": np.float64,
            "outPressure": np.float64,
            "humidity": np.float64,
        },
    )
    ambient_df.set_index("ts", inplace=True)

    # Get tenmperature from the last entry in the ambient data
    measured_temp_c = ambient_df["outTemp"].iloc[-1]
    measured_humid_percent = ambient_df["humidity"].iloc[-1]
    measured_pressure_hPa = (
        ambient_df["outPressure"].iloc[-1] + ambient_df["intPressure"].iloc[-1]
    ) / 2

    # Calculate rho values
    rho_in = calc_rho(measured_temp_c, measured_humid_percent, measured_pressure_hPa)
    rho_out = calc_rho(31, 94, measured_pressure_hPa)
    rho_btps = calc_rho(37, 100, measured_pressure_hPa)
    # rho_out = rho_btps

    csv_data = StringIO("".join(part2))

    df = pd.read_csv(
        csv_data,
        names=[
            "ts",
            "type",
            "millis",
            "dpIn",
            "dpOut",
            "o2",
            "mCo2",
            "co2Temp",
            "co2Hum",
            "intTemp",
        ],
        dtype={
            "millis": np.float64,
            "dpIn": np.float64,
            "dpOut": np.float64,
            "o2": np.float64,
            "mCo2": np.float64,
            "co2Temp": np.float64,
            "co2Hum": np.float64,
            "intTemp": np.float64,
        },
    )

    computed_columns = [
        "volAirIn",
        "volAirInStpd",
        "volAirOut",
        "volAirOutStpd",
        "volO2InStpd",
        "volO2OutStpd",
        "volCo2OutStpd",
    ]

    ### Start data processing
    df.drop(['ts', 'type'], axis = 1, inplace=True)
    df.set_index("millis", inplace=True)
    min_index = df.index.min()
    df.index = df.index - min_index # start with 0
    df["dpSum"] = 0
    # Sanitixe data
    # sanitize_data()
    df["millis_diff"] = df.index.to_series().diff() # time between samples
    df["oneDp"] = df["dpIn"] - df["dpOut"]  # signed diff pressure
    df["o2ini"] = df["o2"] # save the initial O2 values so 
    o2_max = df["o2"].max()
    df["rhoOut"] = calc_rho(df["co2Temp"], df["co2Hum"], measured_pressure_hPa)

    # Compute gas volumes for each row
    calc_volumes_wrapper = partial(calc_volumes, rhoIn=rho_in)
    df[computed_columns] = df.apply(calc_volumes_wrapper, axis=1, result_type="expand")

    # Compute breath limits
    df["BreathMarker"] = False
    breath_indexes, breath_indexes_ms = find_breath_limits(df)
    df.loc[breath_indexes_ms, "BreathMarker"] = True
    breath_mask = df["BreathMarker"] == True
    breath_indices = df.index[breath_mask] # indexes of breath markers in df (in millis)
    breath_durations = breath_indices.to_series().diff(periods=2)
    df['breathDuration'] = np.nan
    df.loc[breath_mask, 'breathDuration'] = breath_durations

    # Compute breath volumes
    df['breath_group'] = df['BreathMarker'].cumsum()
    breath_volumes = df.groupby('breath_group')['volAirOut'].sum()
    # make nan all values that are less than 400 (breath markers mark the shift from inhale to exhale)
    # 400 ml should be less than minimum exhale volue, except perhaps at rest/sleep
    breath_volumes = breath_volumes.where(breath_volumes > 400) 
    df['volAirOutSum'] = df['breath_group'].map(breath_volumes) # map back to dataframe
    df.loc[~df['BreathMarker'], 'volAirOutSum'] = np.nan # keep the value only for breath markers
    df.drop('breath_group', axis=1, inplace=True)
    exhale_mask = (df["BreathMarker"] == True) & (df['volAirOutSum'].notna())
    # O2 consumed
    df['volO2UsedStpd'] = df['volO2InStpd'] - df['volO2OutStpd']

    ### resample
    tddf = df.copy()
    tddf.index = pd.to_timedelta(tddf.index, unit="milliseconds")
    ares = tddf.resample("1s").sum()
    ares.set_index((ares.index - ares.index[0]).total_seconds(), inplace=True) # set index to seconds
    # Breath markers
    ares_breath_mask = ares["BreathMarker"] == True
    ares_exhale_mask = ((ares["BreathMarker"] == True) & (ares['volAirOutSum'] > 400 ))
    ### End data processing


    # Tidal Volume
    our_plot(
        x=ares[ares_exhale_mask].index,
        ydata=[
            {
                "y": ares[ares_exhale_mask]["volAirOutSum"],
                "window": 15,
                "label": "Exhaled Air Smoothed",
                "color": "blue"
            },
            {
                "y": ares[ares_exhale_mask]["volAirOutSum"],
                "label": "Exhaled Air",
                "color": "cyan",
                "method": "scatter"
            }
        ],
        figsize=(8, 4),
        title="Tidal Volume"
    )
    
    # RF - Respiratory Frequency
    our_plot(
        x=ares[ares_breath_mask].index,
        ydata=[
            {
                "y": (60000 / ares[ares_breath_mask]["breathDuration"]),
                "window": 5,
                "label": "Breaths per minute",
                "color": "green",
            },
            
        ],
        figsize=(8, 4),
        title="RF - Respiratory Frequency",
    )

    # RER
    our_plot(
        x=ares.index,
        ydata=[
            {
                "y": ares["volCo2OutStpd"].rolling(window=60, min_periods=59, center=True).sum() / 
                     ares["volO2UsedStpd"].rolling(window=60, min_periods=59, center=True).sum(),  
                "window": 15,
                "label": "RER",
                "color": "magenta"
            }
        ],
        figsize=(8, 4),
        title="Respiratory Exchange Ratio (RER)"
    )

    # VolIn/VolOut
    our_plot(
        x=ares.index,
        ydata=[
            {
                "y": ares["volAirInStpd"].rolling(window=60, min_periods=59, center=True).sum(),
                "window": 10,
                "label": "In",
                "color": "red",
                "units": "ml"
                
            },
            {
                "y": ares["volAirOutStpd"].rolling(window=60, min_periods=59, center=True).sum(),
                "window": 10,
                "label": "Out",
                "color": "blue"
            },
            {
                "y": equipment_data.loc[:ares.index.max()+1]['hr'],   # don't plot any HR data after the last mask datapoint
                "window": 2,
                "label": "HR",
                "color": "darkmagenta",
                "tick_interval": 10,
                "tick_color": "darkmagenta"
            }
        ],
        title="VolIn vs VolOut (STPD)",
    )

    # Minute Ventilation (VE)
    our_plot(
        x=ares.index,
        ydata=[
            {
                "y": (ares["volAirOut"]*(ares['rhoOut']/rhoBTPS)).rolling(window=60, min_periods=59, center=True).sum() / 
                (ares["volO2UsedStpd"]*(ares['rhoOut']/rhoBTPS) +0.00000001).rolling(window=60, min_periods=59, center=True).sum()
                .rolling(window=10, center=True).mean(),
                "window": 10,
                "label": "VEqO2",
                "color": "red"
            },
            {
                "y": (ares["volAirOut"]*(ares['rhoOut']/rhoBTPS)).rolling(window=40, center=True, min_periods=39).sum() /
                     (ares["volCo2OutStpd"]*(ares['rhoOut']/rhoBTPS)+0.00000001).rolling(window=40, center=True, min_periods=39).sum(),
                "window": 10,
                "label": "VEqCO2",
                "color": "blue"
            },
            {
                "y": equipment_data.loc[:ares.index.max()+1]['hr'],   # don't plot any HR data after the last mask datapoint
                "window": 2,
                "label": "HR",
                "color": "darkmagenta",
                "tick_interval": 10,
                "tick_color": "darkmagenta"
            }
        ],
        title="VE(minute ventilation)"
    )

    # Vo2Max
    our_plot(
        x=ares.index,
        ydata=[
            {
                "y": ((ares["volO2UsedStpd"] / subject_weight).rolling(window=60, min_periods=14, center=True).sum() ),
                "window": 10,
                "label": "VO2MaxSmoothed (60s)",
                "color": "coral"
            },
            {
                "y": (ares["volO2UsedStpd"] * 4 / subject_weight).rolling(window=15, min_periods=14, center=True).sum(),
                "window": 10,
                "label": "VO2Max (15s)",
                "color": "lightgray",
                "method": "scatter"
            },
            {
                "y": equipment_data.loc[:min(ares.index.max(), equipment_data.index.max())+1]['hr'],   # don't plot any HR data after the last mask datapoint
                "window": 2,
                "label": "HR",
                "color": "darkmagenta",
                "tick_interval": 10,
                "tick_color": "darkmagenta"
            }
        ],
        title="VO2Max"
    )


    #@O2
    our_plot(
        x=df.index,
        ydata=[
            {
                "y": df['o2'],
                "window": 5,
                "label": "Oxy",
                "color": "red"
            },
                        {
                "y": df['mCo2'],
                "window": 5,
                "label": "CO2",
                "color": "blue"
            }
            
        ],
        figsize=(8, 4),
        title="Gases",       
        millis =  True
    )

    plt.figure(figsize=(18, 9))
    plt.title("Pressure debug")
    plt.plot(df.index, df["oneDp"], "black", label="DP")
    plt.axhline(y=0, color="red", linestyle="--")
    plt.plot(df.index, df["o2ini"] - 15, "r", label="O2Ini - 15")
    plt.plot(df.index, df["mCo2"], "blue", label="co2")
    plt.plot(df.index, df["co2Temp"] - 20, "magenta", label="TempCo - 20")
    plt.plot(df.index, df["co2Hum"] - 80, "cyan", label="HumCo2 -80")
    # plot a vertical line for each breath marker
    for i in range(0, len(breath_indexes_ms), 1):
        plt.axvline(x=breath_indexes_ms[i], color="gray", linestyle="--")
    plt.legend(loc="upper left")
    plt.tight_layout()

  

    if plot_old_graphs:
      # window_size_shift_ms = 30000 #ms
      window_size_shift_ms = 1000  # ms

      df["BreathMarker"] = False

      computed_columns = [
          "volAirIn",
          "volAirOut",
          "volO2InStpd",
          "volO2OutStpd",
        "volCo2OutStpd",
        "mCo2OutStpd"
      ]

      df[computed_columns] = 0
      breath_indexes, breath_indexes_ms = find_breath_limits(df)
      df.loc[breath_indexes_ms, "BreathMarker"] = True
      breath_mask = df["BreathMarker"] == True
      # Up until here we have computed the breath limits
      # and marked them in the originaldataframe

      # Plot O2 before smoothing
      plt.ion()
      plt.figure(figsize=(8, 4))
      plt.title("O2% initial")
      plt.plot(df.index, df["o2"], "red", label="oxy")

      # Smooth in place the O2 signal
      # df['o2'] = detect_and_flatten_spikes(df['o2'], window_size=258, spike_threshold=0.05).rolling(window=230, center=True).mean()
      # while o2_smoothing_factor > 0:
      #     df["o2"] = df["o2"].rolling(window=430, center=True).mean()
      #     o2_smoothing_factor -= 1

      # Compute vol, o2, co2Stpd for each group of breaths
      breath_indexes.append(df.index[-1])  # to process the last section ???
      for i in range(step, len(breath_indexes_ms) - 1, step):
          df.loc[breath_indexes_ms[i - step], computed_columns] = calc_vol_o2(
              rho_in, rho_out, df.loc[breath_indexes_ms[i - step] : breath_indexes_ms[i]]
          )

      # Take out results in separate dataframe
      rez_df = df[(df["BreathMarker"] == True) & (df["volO2InStpd"] > 0)]
      breaths_df = df[df["BreathMarker"] == True]
      breaths_df["rf"] = 60000 / (
          breaths_df.index.astype(int).diff(periods=2)
      )  # respiratory frequency (breaths per minute)

      # Compute vo2max and other physiological parameters
      rez_df["millis_diff"] = rez_df.index.to_series().diff()
      rez_df["vO2"] = rez_df["volO2InStpd"] - rez_df["volO2OutStpd"]
      rez_df["vO2/min/kg"] = (
          (rez_df["volO2InStpd"] - rez_df["volO2OutStpd"]) * 60000 / rez_df["millis_diff"]
      ) / subject_weight
      rez_df["volDifStpd"] = normalize_to_stpd(
          rez_df["volAirIn"], rho_in
      ) - normalize_to_stpd(rez_df["volAirOut"], rho_out)
      rez_df["volDifProc"] = rez_df["volDifStpd"] / (rez_df["volAirOut"] + 0.0001)
      rez_df["Ve_Vo2"] = rez_df["volAirOut"] / (rez_df["vO2"] + 0.000001)
      rez_df["Ve_Co2"] = rez_df["volAirOut"] / (rez_df["mCo2OutStpd"] + 0.000001)
      rez_df["RER"] = (
          rez_df["mCo2OutStpd"]
          * (1 - rez_df["vO2/min/kg"] / 100)
          / (rez_df["volO2InStpd"] - rez_df["volO2OutStpd"])
      )  # added rough correction for bicarbonate buffering

      # Plot the results
      plt.figure(figsize=(18, 9))
      plt.title("Differential Pressure")
      plt.plot(df.index, df["oneDp"], "black", label="DP")
      plt.axhline(y=0, color="red", linestyle="--")
      plt.plot(df.index, df["o2ini"] - 15, "r", label="O2Initial")
      plt.plot(df.index, df["mCo2"], "blue", label="co2")
      plt.plot(df.index, df["co2Temp"] - 20, "magenta", label="TempCo2")
      plt.plot(df.index, df["co2Hum"] - 80, "cyan", label="HumCo2")
      # plot a vertical line for each breath marker
      for i in range(0, len(breath_indexes_ms), 1):
          plt.axvline(x=breath_indexes_ms[i], color="gray", linestyle="--")

      plt.figure(figsize=(8, 4))
      plt.title("CO2Stpd")
      plt.scatter(rez_df.index, rez_df["volCo2OutStpd"], color="blue", label="calculated_co2")
      plt.plot(rez_df.index, rez_df["mCo2OutStpd"], "magenta", label="measured_co2")
      plt.legend()

      filtered_temp = df[df['intTemp'] <= 60]
      plt.figure(figsize=(8,4))
      plt.title('IntTem/O2/C02conc')
      plt.plot(filtered_temp.index, filtered_temp['intTemp'], label='intTemp', color='coral')
      plt.plot(df.index, df['o2'], 'red',  label='oxy')
      plt.plot(df.index, df['mCo2'], 'blue',  label='co2')

      # plt.subplots_adjust(left=0.045, right=0.99, top=0.99, bottom=0.056)

      plt.figure(figsize=(8, 4))
      plt.title("HR and Power")
      if "power" in equipment_data.columns:
          plt.plot(equipment_data.index, equipment_data["power"], label="Power", color="grey")
      elif "speed" in equipment_data.columns:
          plt.plot(equipment_data.index, equipment_data["speed"]*equipment_data["grade"], label="Speed", color="grey")

      plt.plot(equipment_data.index, equipment_data["hr"], label="HR", color="red")
      plt.legend()
      plt.axhline(y=75, color="black", linestyle="--")
      plt.axhline(y=100, color="black", linestyle="--")
      plt.axhline(y=125, color="black", linestyle="--")

      plt.figure(figsize=(8, 4))
      plt.plot(
          breaths_df.index,
          breaths_df["rf"].rolling(window=5).mean(),
          color="green",
          label="RF",
      )
      plt.title("Respiratory Frequency")


      print("ha!")

      window_size_sec = 60  # 60 seconds
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
          vol_in = normalize_to_stpd(result[0] / 1000.0, rho_in)
          vol_out = normalize_to_stpd(result[1] / 1000.0, rho_out)
          ve_m_co2 = normalize_to_stpd(result[0], rho_in) / vol_mco2

          window_minutes = window_size_sec / 60  # Convert window size to minutes
          vo2_per_minute = vol_o2 / window_minutes

          start_times.append(start_time)
          vi_vo2_values.append(vol_in / vol_out)
          vo2_values.append(vo2_per_minute / subject_weight)
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
      plot_time_series(vco2_vo2_values, start_times, "RER", "RER")
      plot_time_series(vmco2_vo2_values, start_times, "mRER", "mRER")

      plot_time_series(ve_mco2_values, start_times, "VE_mCO2", "VE_mCO2")

      # plot_time_series(co2_values, start_times, 'CO2', 'CO2' )
      plot_time_series_multi(
          start_times,
          ("vol_in", vol_in_values),
          ("vol_out", vol_out_values),
          plot_fraction=1,
          smoothing_window=10,
          title="Volume In/Out Comparison",
      )
      # plot_time_series(vol_in_values, start_times, 'Vol_in', 'Vol_in' )
      # plot_time_series(ve_o2_out_values, start_times, 'Ve_o2_out', 'Ve_o2_out')
      plot_time_series(ve_o2_values, start_times, "Ve_o2", "Ve_o2")
      plot_time_series(vo2_values, start_times, 'VO2', 'VO2')
      print(f"Max VO2 (rolling, STPD): {round(max_vo2)} ml/min")
      print(f"Max VO2 segment start time: {max_vo2_start_time} ms")



      egr_original_plot(rho_in, rho_out, df)

      
    plt.ioff()
    plt.show(block=True)