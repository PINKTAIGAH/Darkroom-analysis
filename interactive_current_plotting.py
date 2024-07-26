
import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker

# Scripting constanta
DEBUGGING = False
FILEPATH = "/Users/giorgio/Gsi_data/Eris_run010.txt"
LINE_STYLE = "-"
LINE_WIDTH = 0.5
MARKER_STYLE = "x"
MARKER_COLOR = "red"
MARKER_SIZE = 0.5
COORDS_OF_INTEREST_KEYS = ["plateau_1_start", "plateau_1_end", "peak_start", "peak_end", "plateau_2_start", "plateau_2_end"]
# COORDS_OF_INTEREST_KEYS = ["plateau_1_start", "peak_start", "peak_end",  "plateau_2_end"]
EXPECTED_EVENT_COORDS = len(COORDS_OF_INTEREST_KEYS)
RUN_NUMBER = 8
PLOT_TIME_AXIS = True

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


def plot_mean_current(mean_current, time):
    """
    Prodice an interactive plot of current-time series, extract and return coordinates of interest
    """
    # Plot an interactive version of the current respose curve
    fig, ax = plt.subplots(constrained_layout=True)
    if PLOT_TIME_AXIS:
        ax.scatter(time, mean_current, s=MARKER_SIZE, color=MARKER_COLOR)
        ax.set_xlabel("time (s)")
    else:
        ax.plot(mean_current, lw=LINE_WIDTH)
        ax.set_xlabel("a.u")


    # Initiate the clicker module
    clicker_module = clicker(ax, ["event"], markers=["x"])

    ax.set_title(f"Current readings for run {RUN_NUMBER}")
    ax.set_ylabel("Current (pA)")
    plt.show()

    # Save envent coordiantes
    event_coords = clicker_module.get_positions()["event"]
    print(f"\nNumber of events recorded: {event_coords.shape[0]}\n")

    return event_coords

def generate_combined_arrays(event_coords, mean_current, time):
    """
    Use event coords to create combined arrays of features of interest
    """ 
        
    # Create a dictionary containing the x coords for the coordinates of interest    
    COORDS_OF_INTEREST = dict(zip(COORDS_OF_INTEREST_KEYS, event_coords[:, 0].astype(int) ))
    # print(COORDS_OF_INTEREST)
    # print(COORDS_OF_INTEREST["peak_start"], COORDS_OF_INTEREST["peak_end"])

    # Create an array for the mean current peak
    mean_current_peak = mean_current[COORDS_OF_INTEREST["peak_start"] : COORDS_OF_INTEREST["peak_end"]]
    time_peak = time[COORDS_OF_INTEREST["peak_start"] : COORDS_OF_INTEREST["peak_end"]]

    # Create a combined array for the mean current platau
    mean_current_plateu = []
    mean_current_plateu.extend(
        mean_current[COORDS_OF_INTEREST["plateau_1_start"]: COORDS_OF_INTEREST["plateau_1_end"]].tolist()
    )
    mean_current_plateu.extend(
        mean_current[COORDS_OF_INTEREST["plateau_2_start"]: COORDS_OF_INTEREST["plateau_2_end"]].tolist()
    )
    # print(len(mean_current_plateu))
    mean_current_plateu = np.array(mean_current_plateu)
    # mean_current_plateu = np.array([
    #     [mean_current[COORDS_OF_INTEREST["plateau_1_start"], COORDS_OF_INTEREST["plateau_1_end"]]],
    #     [mean_current[COORDS_OF_INTEREST["plateau_2_start"], COORDS_OF_INTEREST["plateau_2_end"]]],
    # ]).ravel()
    # mean_current_plateu = np.delete(
    #     mean_current, np.arange(COORDS_OF_INTEREST["peak_start"], COORDS_OF_INTEREST["peak_end"], 1)
    #     )[ COORDS_OF_INTEREST["plateau_1_start"] : COORDS_OF_INTEREST["plateau_2_end"]]
    time_plateu = np.arange(COORDS_OF_INTEREST["plateau_1_start"], COORDS_OF_INTEREST["plateau_1_start"] + mean_current_plateu.size)
 
    return mean_current_peak, mean_current_plateu, time_peak, time_plateu

def plot_features_of_interest(mean_current_peak, mean_current_plateau, time_peak, time_plateu):
    """
    Plot the user selected current peak and combined platau
    """

    fig, (ax1, ax2) = plt.subplots(2, 1)

    if PLOT_TIME_AXIS:
        ax1.scatter(time_peak, mean_current_peak, s=MARKER_SIZE)
        ax1.set_xlabel("time (s)")
    else:
        ax1.plot(mean_current_peak, lw=LINE_WIDTH)
        ax1.set_xlabel("a.u")

    ax1.set_title("Mean current peak")
    ax1.set_ylabel("Current (pA)")
    ax1.grid()
    
    if PLOT_TIME_AXIS:
        ax2.scatter(time_plateu, mean_current_plateau, s=MARKER_SIZE)
        ax2.set_xlabel("time (s)")
    else:
        ax2.plot(mean_current_plateau, lw=LINE_WIDTH)
        ax2.set_xlabel("a.u")

    ax2.set_title("Combined mean current plateau")
    ax2.set_ylabel("Current (pA)")
    ax2.grid()
    
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    # Obtain measurments of run
    time, voltage, mean_current = load_data()

    """
    Debugging
    """
    if DEBUGGING:
        toy_sin = np.sin(np.linspace(0,100, 300)*0.15)
        time = np.arange(0, toy_sin.size)
        mean_current=toy_sin

    # Plot data and extract coordinatws of interest
    event_coords = plot_mean_current(mean_current, time)

    # Verify number of events match what is expected and obtain combined arrays for the plateus and peaks
    if event_coords.shape[0] == EXPECTED_EVENT_COORDS:
        # Create arrays containing the current peak and plateu
        mean_current_peak, mean_current_plateu, time_peak, time_plateu= generate_combined_arrays(event_coords, mean_current, time)

    else:
        raise Exception(f"\nThe number of events selected is not in line with the number of expected events of {EXPECTED_EVENT_COORDS}\n")

    # Plot the peak and plateu 
    plot_features_of_interest(mean_current_peak, mean_current_plateu, time_peak, time_plateu)   