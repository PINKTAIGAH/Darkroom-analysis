import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os 

CFG_DIR = "config.yaml"
with open(CFG_DIR, "r") as file:
    CFG = yaml.safe_load(file)

VOLTAGE_IDXS = [ [4505, 5189], [8033, 9062], ]

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

    time_diff= np.diff(time)

    return time, applied_voltage, mean_current

def plot_mean_current_special(mean_current, time, voltage, run_number=1):

        # Plot an interactive version of the current respose curve
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, height_ratios=[3, 1], sharex=True)
    if CFG["plot_time_axis"]:
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
        ax1.plot(x_axis, mean_current, lw=CFG["line_width"], alpha=0.2)
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

    ax1.set_title(f"Current readings for run {run_number}")
    ax1.set_ylabel("Current (pA)")
    ax2.set_ylabel("Voltage (V)")
    plt.show()

def initialise_argument_parser():
    # Define argiment parcer to process cli flags
    parser = argparse.ArgumentParser()
    # -i InputFile -r RunNumber 
    parser.add_argument("-i", "--infile", dest="infile", default="in.txt",help= "Input file")
    parser.add_argument("-r", "--run", dest="run", default="1", help="Run number")
    return parser


def main():

    # Initialise argument parser object
    parser = initialise_argument_parser()
    arguments = parser.parse_args() 

    # Test if infile given exists
    if not os.path.exists(arguments.infile):
        raise Exception(f"The input file provided does not exist.")

    # Obtain measurments of run
    time, voltage, mean_current = load_data(arguments.infile)

    # plot_mean_current(mean_current, time, voltage)
    plot_mean_current_special(mean_current, time, voltage, run_number=arguments.run)

def test():

    pass

if __name__ == "__main__":
    main()

