import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os 

"""
This script is designed to plot the mean current and voltages of a run in addition to the current and voltage ranges provided
in the config file. To run this script, the following command must be typed in the command line:

    python <path/to/script>/plot_current.py -i <path/to/input/file> -r <run number>

Where the input file is the raw txt file containing the data recorded by the LabView software
"""

# Directory of config file which will be read by this script. This script assumes that the config file is located in the 
# same directory this script
CFG_DIR = "config.yaml"
with open(CFG_DIR, "r") as file:
    # Read in config file and store values as a dictionary
    CFG = yaml.safe_load(file)

def load_data(infile):
    """
    Read, load and process data and return arrays containing the time, voltage and mean current readings of the run
    
    Parameters
    ----------
    filename: string
        String containing directory of file to be opened

    Returns
    -------
    time: ndArray
        Returns numpy array containing processed time values from a run
    
    voltage: ndArray
        Returns numpy array containing processed voltage values from a run

    mean_current: ndArray
        Returns numpy array containing processed mean current values from a run
    """
    # Import the raw data file as an array of strings
    data = np.loadtxt(infile, dtype=str, usecols=range(6),)

    # Locate elements of array associated to the first elemont of a header
    header_mask = data == "time"
    header_idx = np.argwhere(header_mask)[:, 0]

    # Remove all rows containing headers and convert array elements to floats
    processed_data = np.delete(data, header_idx, axis=0).astype(np.float32)

    # Create a mask for all negative time entries in the dataset
    negative_mask = processed_data[:,0] < 0.0
    negative_idx = np.argwhere(negative_mask)
    # Remove all entries with a negative time value
    processed_data = np.delete(processed_data, negative_idx, axis=0).astype(np.float32)

    # Excract from the data the timiing and meant current datapoints
    time = processed_data[:, 0]
    applied_voltage = processed_data[:, 1]
    mean_current = processed_data[:, 2]

    # (FOR DEBUGGING) Compute the differential of the time array 
    time_diff= np.diff(time)

    return time, applied_voltage, mean_current

def plot_mean_current_with_ranges(mean_current, time, voltage, run_number=1):
    """
    Function which will plot the mean current of a run highlighting all ranges specified in the configuration file.
    
    Parameters
    ----------
    time: ndArray
        Numpy array containing processed time values from a run
    
    voltage: ndArray
        Numpy array containing processed voltage values from a run

    mean_current: ndArray
        Numpy array containing processed mean current values from a run

    run_number: int, optional
        Run number for current measurments
    """ 

    # Initialise axes for plot
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, height_ratios=[3, 1], sharex=True)
    # Code block for plotting current using time as the x axis
    if CFG["plot_time_axis"]:
        # Plot overall mean current of the run
        ax1.plot(time, mean_current, alpha=0.2)
        # Iterate over index ranges for each current measurment
        for i, idxs in enumerate(CFG["voltage_boundaries"]):
            # Plot the mean current if there is only one index pair
            if len(idxs)==2:
                ax1.plot(time[idxs[0]:idxs[1]], mean_current[idxs[0]:idxs[1]])
            # Plot the mean current if there are only two index pair
            if len(idxs) ==4:
                ax1.plot(time[idxs[0]:idxs[1]], mean_current[idxs[0]:idxs[1]])
                ax1.plot(time[idxs[2]:idxs[3]], mean_current[idxs[2]:idxs[3]])
            # Plot the mean current if there are only three index pair
            elif len(idxs) ==6:
                ax1.plot(time[idxs[0]:idxs[1]], mean_current[idxs[0]:idxs[1]])
                ax1.plot(time[idxs[2]:idxs[3]], mean_current[idxs[2]:idxs[3]])
                ax1.plot(time[idxs[4]:idxs[5]], mean_current[idxs[4]:idxs[5]])

        # Plot the voltage of the run
        ax2.plot(time, voltage)
        ax1.set_xlabel("time (s)")
        ax2.set_xlabel("time (s)")

    # Code block for plotting current using indexes as the x axis
    else:
        # Create array of indexes for the x axis
        x_axis = range(len(mean_current))
        # Plot the mean currents
        ax1.plot(x_axis, mean_current, lw=CFG["line_width"], alpha=0.2)
        ax1.set_xlabel("a.u")
        ax2.set_xlabel("a.u")
        # Plot the voltages
        ax2.plot(voltage)

        # Iterate over index ranges for each current measurment
        for i, idxs in enumerate(CFG["voltage_boundaries"]):
            # Plot the mean current if there is only one index pair
            if len(idxs)==2:
                ax1.plot(x_axis[idxs[0]:idxs[1]], mean_current[idxs[0]:idxs[1]])
            # Plot the mean current if there are only two index pair
            if len(idxs) ==4:
                ax1.plot(x_axis[idxs[0]:idxs[1]], mean_current[idxs[0]:idxs[1]])
                ax1.plot(x_axis[idxs[2]:idxs[3]], mean_current[idxs[2]:idxs[3]])
            # Plot the mean current if there are only three index pair
            elif len(idxs) ==6:
                ax1.plot(x_axis[idxs[0]:idxs[1]], mean_current[idxs[0]:idxs[1]])
                ax1.plot(x_axis[idxs[2]:idxs[3]], mean_current[idxs[2]:idxs[3]])
                ax1.plot(x_axis[idxs[4]:idxs[5]], mean_current[idxs[4]:idxs[5]])

    # Add labels and titles to the plot
    ax1.set_title(f"Current readings for run {run_number}")
    ax1.set_ylabel("Current (pA)")
    ax2.set_ylabel("Voltage (V)")
    plt.show()

def initialise_argument_parser():
    """
    Initialises the argument parser used when starting script in CLI
    
    Returns
    ----------
    parser: ArgumentParser object
        Argparser object used to parse the arguments when calling this script in the CLI. The argparser looks for the following
        flags:
            -i: Represents directory contiaining input file
            -r: Represents int for the run number being analysed
    """ 
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
    plot_mean_current_with_ranges(mean_current, time, voltage, run_number=arguments.run)

def test():

    pass

if __name__ == "__main__":
    main()

