import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
import argparse
import yaml
import os, sys

# Scripting constanta
CFG_DIR = "config.yaml"
with open(CFG_DIR, "r") as file:
    CFG = yaml.safe_load(file)


def load_data(infile):
    """
    Read, load and process data and return arrays containing the time, voltage and mean current readings of the run
    """
    # Import the raw data file as an array of strings
    data = np.loadtxt(infile, dtype=str, usecols=range(6),)

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


def plot_mean_current(mean_current, time, voltage, run_number=1):
    """
    Prodice an interactive plot of current-time series, extract and return coordinates of interest
    """
    # Plot an interactive version of the current respose curve
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, height_ratios=[3, 1], sharex=True)
    if CFG["plot_time_axis"]:
        ax1.plot(time, mean_current, color=CFG["marker_color"])
        ax2.plot(time, voltage)
        ax1.set_xlabel("time (s)")
        ax1.set_xlabel("time (s)")
    else:
        ax1.plot(mean_current, lw=CFG["line_width"])
        ax2.plot(voltage, lw=CFG["line_width"])
        ax1.set_xlabel("a.u")
        ax2.set_xlabel("a.u")

    ax1.set_title(f"Current readings for run {run_number}")
    ax1.set_ylabel("Current (pA)")
    ax2.set_ylabel("Voltage (V)")
    plt.show()

def plot_mean_current_interactive(mean_current, time, voltage, run_number=1):
    """
    Prodice an interactive plot of current-time series, extract and return coordinates of interest
    """
    # Plot an interactive version of the current respose curve
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[3, 1], constrained_layout=True)
    if CFG["plot_time_axis"]:
        ax1.plot(time, mean_current, color=CFG["marker_color"])
        ax1.set_xlabel("time (s)")
        ax2.plot(time, voltage)
        ax2.set_xlabel("time (s)")
    else:
        ax1.plot(mean_current, lw=CFG["line_width"])
        ax1.set_xlabel("a.u")
        ax2.plot(voltage, lw=CFG["line_width"])
        ax2.set_xlabel("a.u")


    # Initiate the clicker module
    clicker_module = clicker(ax1, ["event"], markers=[CFG["marker_style"]])

    ax1.set_title(f"Current readings for run {run_number}")
    ax1.set_ylabel("Current (pA)")
    ax2.set_ylabel("Voltage (V)")
    plt.show()

    # Save envent coordiantes
    event_coords = clicker_module.get_positions()["event"]
    print(f"\nNumber of events recorded: {event_coords.shape[0]}\n")

    return event_coords[:, 0].astype(int)

def make_output_string(features, event_coords):

    outstring = f"[ "
    init_idx = 0
    for feature_num in features:
        end_idx = init_idx + (feature_num*2)
        outstring += f"{list(event_coords[init_idx:end_idx])}, "
        init_idx = end_idx     
    outstring += "]"
    
    return outstring
    

def write_out(outpath, outstring, print_to_cli=False, ):

    if print_to_cli:
        print(outstring)

    # if append_to_config:
    #     print(r"sed -i 's/\\[ \\[.*\\], \\]/%s/g' config.yaml" % outstring)
    #     os.system(r"sed -i 's/\\[ \\[.*\\], \\]/{outstring}/g' config.yaml")

    with open(outpath, "a") as file:
        file.write("\n")
        file.write(outstring)
        file.write("\n")


def plot_mean_current_special(events, mean_current, time, voltage, run_number=1):

        # Plot an interactive version of the current respose curve
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, height_ratios=[3, 1], sharex=True)
    if CFG["plot_time_axis"]:
        ax1.plot(time, mean_current, alpha=0.2)
        for idx1, idx2 in zip(events[::2], events[1::2]):
            ax1.plot(time[idx1:idx2], mean_current[idx1:idx2])

        ax2.plot(time, voltage)
        ax1.set_xlabel("time (s)")
        ax1.set_xlabel("time (s)")

    else:
        x_axis = range(len(mean_current))
        ax1.plot(x_axis, mean_current, lw=CFG["line_width"], alpha=0.2)
        ax1.set_xlabel("a.u")
        ax2.plot(voltage)
        ax1.set_xlabel("time (s)")
        ax1.set_xlabel("time (s)")

        for idx1, idx2 in zip(events[::2], events[1::2]):
                ax1.plot(x_axis[idx1:idx2], mean_current[idx1:idx2])

    ax1.set_title(f"Current readings for run {run_number}")
    ax1.set_ylabel("Current (pA)")
    ax2.set_ylabel("Voltage (V)")
    plt.show()

def time_coord_conversion(event_times, time_array):
    event_coords= []
    for time in event_times:
        coord = np.argmin(np.abs(time_array - time))
        # coord = np.where(time_array==time)
        event_coords.append(coord)
    return np.array(event_coords)

def initialise_argument_parser():
    # Define argiment parcer to process cli flags
    parser = argparse.ArgumentParser()
    # -i InputFile -o OutFile -r RunNumber 
    parser.add_argument("-i", "--infile", dest="infile", default="in.txt",help= "Input file")
    parser.add_argument("-o", "--outfile", dest="outfile", default="/Users/giorgio/Data/Darkroom/output/coord_extraction.txt", help="Output file")
    parser.add_argument("-r", "--run", dest="run", default="1", help="Run number")
    # parser.add_argument("-a", "--append", dest="append", default=False, help="Append coords to config file")

    return parser


"""
DEBUG
"""

def debug(time, mean_current, voltage, run_number=1):
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, height_ratios=[3, 1], sharex=True)
    plt.plot((np.diff(voltage))[-20::-1])
    ax1.plot(time, mean_current, color=CFG["marker_color"])
    ax2.plot(time, voltage)
    ax1.set_xlabel("time (s)")
    ax1.set_xlabel("time (s)")

    ax1.set_title(f"Current readings for run {run_number}")
    ax1.set_ylabel("Current (pA)")
    ax2.set_ylabel("Voltage (V)")
    plt.show()


def main():
    # Initialise argument parser object
    parser = initialise_argument_parser()
    arguments = parser.parse_args() 

    # Test if infile given exists
    if not os.path.exists(arguments.infile):
        raise Exception(f"The input file provided does not exist.")

    # Obtain measurments of run
    time, voltage, mean_current = load_data(arguments.infile)

    if CFG["debug"]:
        debug(time, mean_current, voltage)
        sys.exit()

    plot_mean_current(mean_current, time, voltage, run_number=arguments.run)
    
    features = feature_io()

    # Plot data and extract coordinatws of interest
    event_coords = plot_mean_current_interactive(mean_current, time, voltage, run_number=arguments.run)
    event_coords = time_coord_conversion(event_coords, time)

    # Verify number of events match what is expected and obtain combined arrays for the plateus and peaks
    if event_coords.size/2 == np.sum(features):
        # Create arrays containing the current peak and plateu
        outstring = make_output_string(features, event_coords)
        write_out(arguments.outfile, outstring, print_to_cli=CFG["print_coords_to_cli"], )
        plot_mean_current_special(event_coords, mean_current, time, voltage, run_number=arguments.run)
    else:
        raise Exception(f"\nThe number of events selected is not in line with the number of expected events of 6\n")


def test():

    pass

if __name__ == "__main__":
    main()
    
