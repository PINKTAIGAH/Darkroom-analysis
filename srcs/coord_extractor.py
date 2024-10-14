import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
import argparse
import yaml
import os, sys

"""
This script is designed to produce an interactive plot of the mean current for a run to allow for easy extraction of coords.
This script will format a string containing the indexes of each point selected in the interactive plot into groups of indexes
and then the indexes for each group.

An example of an outstring produced by this script is given below with an explanation of how the string is formatted.

Example of feature array:
    [
        [121, 145, 432, 654],
        [698, 715],
    ]

In the above example, there are two groups of index pairs, which could represent current ranges when applying 50V and 200V
to a diamond respectivly.

Within the first group there are two index pairs. The first index pair is 121 and 145 and represents a current ranging
between those two indexes. This principle is identical for every other index pair in this structure. 

The above example contains two index pairs in the first group: 121-145 and 432-654.
The second group contains one index pair: 698-715.

To run this script, the following command must be typed in the command line:

    python <path/to/script>/coord_extractor.py -i <path/to/input/file> -o <path/to/output/file> -r <run number>

Where the input file is the raw txt file containing the data recorded by the LabView software and the output file is where the 
outstring produces by this script will be written

For a detailed explanation on how to use this script, see the LISA Wiki entry: ADD LINK HERE
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
    
    return time, applied_voltage, mean_current

def feature_io():
    """
    I/O loop for the name and hypothesis for the feature of interest

    Returns
    -------
    features: ndArray
        Numpy array containing the information of the number of index groups and group pairs which are going to be selected
        in the interactive plot
    """

    features = []
    count = 1
    while True:
        num_of_features = input(f"Input number of index pairs for index group {count}. (Press q to quit):  ") 
        if num_of_features == "q" or num_of_features == "Q":
            break
        if not num_of_features.isdigit(): 
            print("Please enter an integer or q to quit.")
            continue
        features.append(int(num_of_features))
        count+=1

    return np.array(features)


def plot_mean_current(mean_current, time, voltage, run_number=1):
    """
    Function which will plot the mean current of a run .
    
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
        # Plot the mean current and voltage of the run
        ax1.plot(time, mean_current, color=CFG["marker_color"])
        ax2.plot(time, voltage)
        ax1.set_xlabel("time (s)")
        ax1.set_xlabel("time (s)")
    # Code block for plotting current using indexes as the x axis
    else:
        # Plot the mean current and voltage of the run
        ax1.plot(mean_current, lw=CFG["line_width"])
        ax2.plot(voltage, lw=CFG["line_width"])
        ax1.set_xlabel("a.u")
        ax2.set_xlabel("a.u")

    # Set title and labels
    ax1.set_title(f"Current readings for run {run_number}")
    ax1.set_ylabel("Current (pA)")
    ax2.set_ylabel("Voltage (V)")
    plt.show()

def plot_mean_current_interactive(mean_current, time, voltage, run_number=1):
    """
    Prodice an interactive plot of current-time series, extract and return coordinates of interest
    
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
    
    Returns
    ----------
    Event coords: ndArray of ints
        Array containing the indexes for the x-axis of the mean current values which were selected in the interactive plot.
        The shape of the array is (1,n) where n is the number of points selected in the plot.
    """
    # Plot an interactive version of the current respose curve
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[3, 1], constrained_layout=True)
    # Code block for plotting current using time as the x axis
    if CFG["plot_time_axis"]:
        ax1.plot(time, mean_current, color=CFG["marker_color"])
        ax1.set_xlabel("time (s)")
        ax2.plot(time, voltage)
        ax2.set_xlabel("time (s)")
    # Code block for plotting current using indexes as the x axis
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
    print(f"\nNumber of indexes recorded: {event_coords.shape[0]}\n")

    return event_coords[:, 0].astype(int)

def make_output_string(features, event_coords):
    """
    Creates a string of a formatted array with the structure which can be later used to indicate groups of index pairs and the index
    pairs of current ranges

    Parameters
    ----------
    features: ndArray
        Numpy array containing the information of the number of index groups and group pairs which are going to be selected
        in the interactive plot

    Event coords: ndArray of ints
        Array containing the indexes for the x-axis of the mean current values which were selected in the interactive plot.
        The shape of the array is (1,n) where n is the number of points selected in the plot.

    Returns
    -------
    outstring: string
        Returns string representing an array containing the number of groups of index pairs for a given current range 
        Structure of array is given by it's size of (n,m) where n in the number of groups of index pairs (may represent
        index pairs for a given bias voltage applied to a diamond) and m represents the indexes which represent the begining 
        and end of a current range.
        
        Example of feature array:
        [
            [121, 145, 432, 654],
            [698, 715],
        ]

        In the above example, there are two groups of index pairs, which could represent current ranges when applying 50V and 200V
        to a diamond respectivly.

        Within the first group there are two index pairs. The first index pair is 121 and 145 and represents a current ranging
        between those two indexes. This principle is identical for every other index pair in this structure. 

        The above example contains two index pairs in the first group: 121-145 and 432-654.
        The second group contains one index pair: 698-715.
    """
    # Initialise outstring
    outstring = f"[ "
    # Initialise index pair counter
    init_idx = 0
    # Iterate over each feature (ie: groups of index pairs)
    for feature_num in features:
        # Determin the total nunber of indexes in a specific group of index pairs
        end_idx = init_idx + (feature_num*2)
        # Create the cotstring for this specific group of index pairs
        outstring += f"{list(event_coords[init_idx:end_idx])}, "
        # Change the begining index of the next group
        init_idx = end_idx     
    outstring += "]"
    
    return outstring
    

def write_out(outpath, outstring, print_to_cli=False, ):
    """
    Function which will print out the outstring of indexes to the cli and write them to a designated file
    
    Parameters
    ----------
    outpath: string 
        Path of file where outstring will be written

    outstring: string 
       Outstring of the formatted indexes 
    
    print_to_cli: bool, optional
        If true the outstring will be printed to the cli in addition to be written out to the indicated file
    """

    if print_to_cli:
        # Print outstring to the CLI
        print(outstring)

    # if append_to_config:
    #     print(r"sed -i 's/\\[ \\[.*\\], \\]/%s/g' config.yaml" % outstring)
    #     os.system(r"sed -i 's/\\[ \\[.*\\], \\]/{outstring}/g' config.yaml")

    with open(outpath, "a") as file:
        # Append the outstring to the designated file in a new line
        file.write("\n")
        file.write(outstring)
        file.write("\n")

def plot_mean_current_with_ranges(events, mean_current, time, voltage, run_number=1):
    """
    Function which will plot the mean current of a run highlighting all ranges.
    
    Parameters
    ----------
    events: ndArray of ints
        Array containing the index pairs for all the ranges to be plotted

    time: ndArray
        Numpy array containing processed time values from a run
    
    voltage: ndArray
        Numpy array containing processed voltage values from a run

    mean_current: ndArray
        Numpy array containing processed mean current values from a run

    run_number: int, optional
        Run number for current measurments
    """ 


    # Code block for plotting current using time as the x axis
    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, height_ratios=[3, 1], sharex=True)
    if CFG["plot_time_axis"]:
        # Plot the overall mean current of the run
        ax1.plot(time, mean_current, alpha=0.2)
        # Iterate over each index pair
        for idx1, idx2 in zip(events[::2], events[1::2]):
            # Plot the highlited current range
            ax1.plot(time[idx1:idx2], mean_current[idx1:idx2])

        # Plot the voltages
        ax2.plot(time, voltage)
        ax1.set_xlabel("time (s)")
        ax2.set_xlabel("time (s)")

    # Code block for plotting current using indexes as the x axis
    else:
        # Create array of indexes for x axis
        x_axis = range(len(mean_current))
        # Plot the overall mean current of the run
        ax1.plot(x_axis, mean_current, lw=CFG["line_width"], alpha=0.2)
        ax1.set_xlabel("a.u")
        ax2.set_xlabel("a.u")
        ax2.plot(voltage)

        # Iterate over each index pair
        for idx1, idx2 in zip(events[::2], events[1::2]):
            # Plot the highlited current range
            ax1.plot(x_axis[idx1:idx2], mean_current[idx1:idx2])

    # Set title and axis labels
    ax1.set_title(f"Current readings for run {run_number}")
    ax1.set_ylabel("Current (pA)")
    ax2.set_ylabel("Voltage (V)")
    plt.show()

def time_coord_conversion(event_times, time_array):
    """
    Function which correct the indexes of the selected points in the interactive plot.
    
    Parameters
    ----------
    events_times: ndArray of ints
        Array containing the index pairs for all the ranges to be plotted

    timearray: ndArray
        Numpy array containing processed time values from a run
    
    Returns
    -------
    event_coords: ndArray of indexes
        An array containing the corrected indexes selected in the interactive plot
    """ 
    # Initialise empty list to contain the event coords
    event_coords= []
    # Loop over each coord
    for time in event_times:
        # Find corresponding coord for a specific time
        coord = np.argmin(np.abs(time_array - time))
        # coord = np.where(time_array==time)
       
        # Append coord to list
        event_coords.append(coord)
    # Convert list to array and return
    return np.array(event_coords)

def initialise_argument_parser():
    """
    Initialises the argument parser used when starting script in CLI
    
    Returns
    ----------
    parser: ArgumentParser object
        Argparser object used to parse the arguments when calling this script in the CLI. The argparser looks for the following
        flags:
            -i: Represents directory contiaining input file
            -o: Represents directory contiaining output file where outstring will be written
            -r: Represents int for the run number being analysed
    """ 
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
        plot_mean_current_with_ranges(event_coords, mean_current, time, voltage, run_number=arguments.run)
    else:
        raise Exception(f"\nThe number of events selected is not in line with the number of expected events of 6\n")


def test():

    pass

if __name__ == "__main__":
    main()
    
