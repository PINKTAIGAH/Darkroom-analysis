
import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from scipy.optimize import curve_fit

# Scripting constanta
DEBUGGING = False
FILEPATH = "/Users/giorgio/Gsi_data/Eris_run033.txt"
OUTPATH = "/Users/giorgio/GSI_data/output/test.txt"
LINE_STYLE = "-"
LINE_WIDTH = 1.5
MARKER_STYLE = "x"
MARKER_COLOR = "red"
MARKER_SIZE = 0.5
FEATURE_OF_INTEREST_KEYS = ["coord", "hypothesis", "current", "time", "voltage", "opt_param", "opt_param_err", ]
# COORDS_OF_INTEREST_KEYS = ["plateau_1_start", "peak_start", "peak_end",  "plateau_2_end"]
RUN_NUMBER = 21
PLOT_TIME_AXIS = False
AVAILABLE_FEATURE_HYPOTHESIS = ["uniform", "linear", "exponential", "0", "1", "2"]

def uniform(x, a):
    return 0*x + a

def linear(x, a, b):
    return a*x + b

def exponential(x, a, b, c):
    return a*np.exp(b*x) + c




def load_data():
    """
    Read, load and process data and return arrays containing the time, voltage and mean current readings of the run
    """
    # Import the raw data file as an array of strings
    data = np.loadtxt(FILEPATH, dtype=str, usecols=range(6),)

    # Locate elements of array associated to the first elemont of a header
    header_mask = data == "time"
    header_idx = np.argwhere(header_mask)[:, 0]

    # Remove all rows containing headers and convert array elements to floats
    processed_data = np.delete(data, header_idx, axis=0).astype(np.float32)

    # Excract from the data the timiing and meant current datapoints
    time = processed_data[:, 0] 
    applied_voltage = processed_data[:, 1]
    mean_current = processed_data[:, 2] 
    
    return time, applied_voltage, mean_current

def feature_io():
    """
    I/O loop for the name and hypothesis for the feature of interest
    """

    features = []
    while True:
        num_of_features = input("Input number of features. (Press q to quit):  ") 
        if num_of_features == "q" or num_of_features == "Q":
            break
        if not num_of_features.isdigit(): 
            print("Please enter an integer or q to quit.")
            continue
        features.append(int(num_of_features))

    return np.array(features)


def plot_mean_current(mean_current, time, voltage):
    """
    Prodice an interactive plot of current-time series, extract and return coordinates of interest
    """
    # Plot an interactive version of the current respose curve
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, height_ratios=[3, 1], sharex=True)
    if PLOT_TIME_AXIS:
        ax1.plot(time, mean_current, color=MARKER_COLOR)
        ax2.plot(time, voltage)
        ax1.set_xlabel("time (s)")
        ax1.set_xlabel("time (s)")
    else:
        ax1.plot(mean_current, lw=LINE_WIDTH)
        ax2.plot(voltage, lw=LINE_WIDTH)
        ax1.set_xlabel("a.u")
        ax2.set_xlabel("a.u")

    ax1.set_title(f"Current readings for run {RUN_NUMBER}")
    ax1.set_ylabel("Current (pA)")
    ax2.set_ylabel("Voltage (V)")
    plt.show()

def plot_mean_current_interactive(mean_current, time, voltage):
    """
    Prodice an interactive plot of current-time series, extract and return coordinates of interest
    """
    # Plot an interactive version of the current respose curve
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[3, 1], constrained_layout=True)
    if PLOT_TIME_AXIS:
        ax1.plot(time, mean_current, color=MARKER_COLOR)
        ax1.set_xlabel("time (s)")
        ax2.plot(time, voltage)
        ax2.set_xlabel("time (s)")
    else:
        ax1.plot(mean_current, lw=LINE_WIDTH)
        ax1.set_xlabel("a.u")
        ax2.plot(voltage, lw=LINE_WIDTH)
        ax2.set_xlabel("a.u")


    # Initiate the clicker module
    clicker_module = clicker(ax1, ["event"], markers=["x"])

    ax1.set_title(f"Current readings for run {RUN_NUMBER}")
    ax1.set_ylabel("Current (pA)")
    ax2.set_ylabel("Voltage (V)")
    plt.show()

    # Save envent coordiantes
    event_coords = clicker_module.get_positions()["event"]
    print(f"\nNumber of events recorded: {event_coords.shape[0]}\n")

    return event_coords[:, 0].astype(int)

def make_output_string(features, event_coords):

    outstring = f"( "
    init_idx = 0
    for feature_num in features:
        end_idx = init_idx + (feature_num*2)
        outstring += f"{tuple(event_coords[init_idx:end_idx])}, "
        init_idx = end_idx     
    outstring += ")"
    
    return outstring
    

def write_out(outstring):

    with open(OUTPATH, "a") as file:
        file.write("\n")
        file.write(outstring)
        file.write("\n")


def plot_mean_current_special(events, mean_current, time, voltage, plot_time_axis=True):


        # Plot an interactive version of the current respose curve
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, height_ratios=[3, 1], sharex=True)
    if plot_time_axis:
        ax1.plot(time, mean_current, alpha=0.2)
        for idx1, idx2 in zip(events[::2], events[1::2]):
            ax1.plot(time[idx1:idx2], mean_current[idx1:idx2])

        ax2.plot(time, voltage)
        ax1.set_xlabel("time (s)")
        ax1.set_xlabel("time (s)")

    else:
        x_axis = range(len(mean_current))
        ax1.plot(x_axis, mean_current, lw=LINE_WIDTH, alpha=0.2)
        ax1.set_xlabel("a.u")
        ax2.plot(voltage)
        ax1.set_xlabel("time (s)")
        ax1.set_xlabel("time (s)")

        for idx1, idx2 in zip(events[::2], events[1::2]):
                ax1.plot(x_axis[idx1:idx2], mean_current[idx1:idx2])

    ax1.set_title(f"Current readings for run {RUN_NUMBER}")
    ax1.set_ylabel("Current (pA)")
    ax2.set_ylabel("Voltage (V)")
    plt.show()


def main():

    # Obtain measurments of run
    time, voltage, mean_current = load_data()

    plot_mean_current(mean_current, time, voltage)
    
    features = feature_io()

    # Plot data and extract coordinatws of interest
    event_coords = plot_mean_current_interactive(mean_current, time, voltage)

    # Verify number of events match what is expected and obtain combined arrays for the plateus and peaks
    if event_coords.size/2 == np.sum(features):
        # Create arrays containing the current peak and plateu
        outstring = make_output_string(features, event_coords)
        write_out(outstring)
        plot_mean_current_special(event_coords, mean_current, time, voltage, plot_time_axis=False)
    else:
        raise Exception(f"\nThe number of events selected is not in line with the number of expected events of 6\n")


def test():

    pass

if __name__ == "__main__":
    main()
    