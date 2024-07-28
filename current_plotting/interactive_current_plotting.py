
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
FEATURE_OF_INTEREST_KEYS = ["coord", "hypothesis", "current", "time", "voltage"]
# COORDS_OF_INTEREST_KEYS = ["plateau_1_start", "peak_start", "peak_end",  "plateau_2_end"]
EXPECTED_EVENT_COORDS = len(COORDS_OF_INTEREST_KEYS)
RUN_NUMBER = 8
PLOT_TIME_AXIS = True
AVAILABLE_FEATURE_HYPOTHESIS = ["uniform", "linear", "exponential"]

def uniform(x, a):
    return a*x

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

    feature_of_interest_names = []
    feature_of_interest_hypothesis = []

    for _ in range(3):
        feature_of_interest_names.append(feature_name_input_loop(feature_of_interest_names))
        feature_of_interest_hypothesis.append(feature_hypothesis_input_loop())

    features_of_interest_dict = {
        name: {
            key: None for key in FEATURE_OF_INTEREST_KEYS
        } for name in feature_of_interest_names
    }

    for name, hypothesis in zip(feature_of_interest_names, feature_of_interest_hypothesis):
        features_of_interest_dict[name]["hypothesis"] = hypothesis

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
    print(type(event_coords))
    print(f"\nNumber of events recorded: {event_coords.shape[0]}\n")

    return event_coords

def generate_combined_arrays(event_coords, mean_current, time, feature_dict):
    """
    Use event coords to create combined arrays of features of interest
    """ 
        
    # Create a dictionary containing the x coords for the coordinates of interest    

    COORDS_OF_INTEREST = dict(zip(COORDS_OF_INTEREST_KEYS, event_coords[:, 0].astype(int) ))
    for idx, name in enumerate(feature_dict):
        feature_dict[name]["coord"] = (event_coords[idx, 0].astype(int), event_coords[idx+1, 0].astype(int))

    dict_names = list(feature_dict.keys())
    # Create an array for the mean current peak
    feature_dict[dict_names[1]]["current"] = mean_current[feature_dict[dict_names[1]]["coord"][0] : feature_dict[dict_names[1]]["coord"][1]]
    feature_dict[dict_names[1]]["time"] = time[feature_dict[dict_names[1]]["coord"][0] : feature_dict[dict_names[1]]["coord"][1]]

    # Create a combined array for the mean current platau
    mean_current_plateu = []
    mean_current_plateu.extend(
        mean_current[feature_dict[dict_names[0]]["coord"][0] : feature_dict[dict_names[0]]["coord"][1]]
    )
    mean_current_plateu.extend(
        mean_current[feature_dict[dict_names[2]]["coord"][0] : feature_dict[dict_names[2]]["coord"][1]]
    )
    # print(len(mean_current_plateu))
    feature_dict[dict_names[0]]["current"] = np.array(mean_current_plateu)
    feature_dict[dict_names[2]]["current"] = np.array(mean_current_plateu)

    feature_dict[dict_names[0]]["time"] = np.arange(feature_dict[dict_names[0]]["coord"][0], feature_dict[dict_names[0]]["coord"][0] + len(mean_current_plateu))
 
    return feature_dict

def plot_features_of_interest(feature_dict):
    """
    Plot the user selected current peak and combined platau
    """

    dict_names = list(feature_dict.keys())
    fig, (ax1, ax2) = plt.subplots(2, 1)

    if PLOT_TIME_AXIS:
        ax1.scatter(feature_dict[dict_names[1]]["time"], feature_dict[dict_names[1]]["current"], s=MARKER_SIZE)
        ax1.set_xlabel("time (s)")
    else:
        ax1.plot(feature_dict[dict_names[1]]["current"], lw=LINE_WIDTH)
        ax1.set_xlabel("a.u")

    ax1.set_title("Mean current peak")
    ax1.set_ylabel("Current (pA)")
    ax1.grid()
    
    if PLOT_TIME_AXIS:
        ax2.scatter(feature_dict[dict_names[0]]["time"], feature_dict[dict_names[0]]["current"], s=MARKER_SIZE)
        ax2.set_xlabel("time (s)")
    else:
        ax2.plot(feature_dict[dict_names[0]]["current"], lw=LINE_WIDTH)
        ax2.set_xlabel("a.u")

    ax2.set_title("Combined mean current plateau")
    ax2.set_ylabel("Current (pA)")
    ax2.grid()
    
    fig.tight_layout()
    plt.show()

def main():

    # Obtain measurments of run
    time, voltage, mean_current = load_data()

    features_of_interest_dict = feature_io()


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
    if event_coords.shape[0]/2 == len(features_of_interest_dict):
        # Create arrays containing the current peak and plateu
        features_of_interest_dict = generate_combined_arrays(event_coords, mean_current, time, features_of_interest_dict)

    else:
        raise Exception(f"\nThe number of events selected is not in line with the number of expected events of {EXPECTED_EVENT_COORDS}\n")

    # Plot the peak and plateu 
    plot_features_of_interest(features_of_interest_dict)   


def test():

    main()


if __name__ == "__main__":
    test()
    