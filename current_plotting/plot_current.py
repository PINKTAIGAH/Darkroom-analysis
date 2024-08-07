
import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from scipy.optimize import curve_fit

# Scripting constanta
DEBUGGING = False
# FILEPATH = "/Users/giorgio/Gsi_data/Eris_run009.txt"
FILEPATH = "/Users/giorgio/Gsi_data/Eris_run028.txt"
OUTPATH = "/Users/giorgio/GSI_data/output/test.txt"
LINE_STYLE = "-"
LINE_WIDTH = 0.5
MARKER_STYLE = "x"
MARKER_COLOR = "red"
MARKER_SIZE = 0.5
FEATURE_OF_INTEREST_KEYS = ["coord", "hypothesis", "current", "time", "voltage", "opt_param", "opt_param_err", ]
# COORDS_OF_INTEREST_KEYS = ["plateau_1_start", "peak_start", "peak_end",  "plateau_2_end"]
RUN_NUMBER = 8
PLOT_TIME_AXIS = True
AVAILABLE_FEATURE_HYPOTHESIS = ["uniform", "linear", "exponential", "0", "1", "2"]
VOLTAGE_IDXS = ( (762, 1635), (1962, 2844), (3329, 4123), (6875, 7252), )


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


    negative_mask = processed_data[:,0] < 0.0
    negative_idx = np.argwhere(negative_mask)
    processed_data = np.delete(processed_data, negative_idx, axis=0).astype(np.float32)

    # Excract from the data the timiing and meant current datapoints
    time = processed_data[:, 0] 
    applied_voltage = processed_data[:, 1]
    mean_current = processed_data[:, 2] 

    time_diff= np.diff(time)
    
    return time, applied_voltage, mean_current

def feature_io():
    """
    I/O loop for the name and hypothesis for the feature of interest
    """

    feature_of_interest_names = []
    feature_of_interest_hypothesis = []

    print(f"For hypothesis: 0 => uniform, 2 => linear and 3 => exponential")

    for _ in range(3):
        feature_of_interest_names.append(feature_name_input_loop(feature_of_interest_names))
        feature_of_interest_hypothesis.append(feature_hypothesis_input_loop())

    features_of_interest_dict = {
        name: {
            key: None for key in FEATURE_OF_INTEREST_KEYS
        } for name in feature_of_interest_names
    }

    for name, hypothesis in zip(feature_of_interest_names, feature_of_interest_hypothesis):
        if hypothesis == "uniform" or hypothesis == "0":
            features_of_interest_dict[name]["hypothesis"] = uniform
        if hypothesis == "linear" or hypothesis == "1":
            features_of_interest_dict[name]["hypothesis"] = linear
        if hypothesis == "exponential" or hypothesis == "2":
            features_of_interest_dict[name]["hypothesis"] = exponential

    return features_of_interest_dict
        

def feature_name_input_loop(feature_of_interest_names, feature_name=None,):
    
        while feature_name is None:
            feature_name = input("\nInput name of feature: ")
        
            if feature_name in feature_of_interest_names:
                feature_name = None
                print("\nName of feature is already in use. Please name it something else")
            
        return feature_name

def feature_hypothesis_input_loop(feature_hypothesis=None):

        while feature_hypothesis is None:
            feature_hypothesis = input("\nInput feature hypothesis: ")

            if feature_hypothesis not in AVAILABLE_FEATURE_HYPOTHESIS:
                feature_hypothesis = None
                print(f"\nThe hypothesis selected is not recognised. Please choose from {AVAILABLE_FEATURE_HYPOTHESIS}")
            
        return feature_hypothesis    


def plot_mean_current(mean_current, time, voltage):
    """
    Prodice an interactive plot of current-time series, extract and return coordinates of interest
    """
    # Plot an interactive version of the current respose curve
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, height_ratios=[3, 1], sharex=True)
    if PLOT_TIME_AXIS:
        ax1.scatter(time, mean_current, s=MARKER_SIZE, color=MARKER_COLOR)
        ax2.plot(time, voltage)
        ax1.set_xlabel("time (s)")
        ax1.set_xlabel("time (s)")
    else:
        ax1.plot(mean_current, lw=LINE_WIDTH)
        ax1.set_xlabel("a.u")
        ax2.set_xlabel("a.u")

    ax1.set_title(f"Current readings for run {RUN_NUMBER}")
    ax1.set_ylabel("Current (pA)")
    ax2.set_ylabel("Voltage (V)")
    plt.show()

def plot_mean_current_special(mean_current, time, voltage):


        # Plot an interactive version of the current respose curve
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, height_ratios=[3, 1], sharex=True)
    if PLOT_TIME_AXIS:
        ax1.plot(time, mean_current, alpha=0.2)
        for i, idxs in enumerate(VOLTAGE_IDXS):
            if len(idxs)==2:
                ax1.plot(time[idxs[0]:idxs[1]], mean_current[idxs[0]:idxs[1]])
            if len(idxs) ==4:
                ax1.plot(time[idxs[0]:idxs[1]], mean_current[idxs[0]:idxs[1]])
                ax1.plot(time[idxs[2]:idxs[3]], mean_current[idxs[2]:idxs[3]])
            elif len(idxs) ==6:
                ax1.plot(time[idxs[0]:idxs[1]], mean_current[idxs[0]:idxs[1]])
                ax1.plot(time[idxs[2]:idxs[3]], mean_current[idxs[2]:idxs[3]])
                ax1.plot(time[idxs[4]:idxs[5]], mean_current[idxs[4]:idxs[5]])

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

        for i, idxs in enumerate(VOLTAGE_IDXS):
            if len(idxs)==2:
                ax1.plot(x_axis[idxs[0]:idxs[1]], mean_current[idxs[0]:idxs[1]])
            if len(idxs) ==4:
                ax1.plot(x_axis[idxs[0]:idxs[1]], mean_current[idxs[0]:idxs[1]])
                ax1.plot(x_axis[idxs[2]:idxs[3]], mean_current[idxs[2]:idxs[3]])
            elif len(idxs) ==6:
                ax1.plot(x_axis[idxs[0]:idxs[1]], mean_current[idxs[0]:idxs[1]])
                ax1.plot(x_axis[idxs[2]:idxs[3]], mean_current[idxs[2]:idxs[3]])
                ax1.plot(x_axis[idxs[4]:idxs[5]], mean_current[idxs[4]:idxs[5]])

    ax1.set_title(f"Current readings for run {RUN_NUMBER}")
    ax1.set_ylabel("Current (pA)")
    ax2.set_ylabel("Voltage (V)")
    plt.show()


def main():

    # Obtain measurments of run
    time, voltage, mean_current = load_data()

    """
    Debugging
    """
    if DEBUGGING:
        toy_sin = np.sin(np.linspace(0,100, 300)*0.15)
        time = np.arange(0, toy_sin.size)
        mean_current=toy_sin


    # plot_mean_current(mean_current, time, voltage)
    plot_mean_current_special(mean_current, time, voltage)

def test():

    pass

if __name__ == "__main__":
    main()
    