import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def uniform(x, a):
    """
    Applies a uniform function to the input array

    Parameters
    ----------
    x: ndArray
        Input array
    
    a: float
        Uniform constant

    Returns
    -------
    x: ndArray
        Returns input array after application of uniform constant
    """
    return 0*x + a

def linear(x, a, b):
    """
    Applies a linear function to the input array

    Parameters
    ----------
    x: ndArray
        Input array
    
    a: float
        Linear gradient
    
    b: float
        Offset constant

    Returns
    -------
    x: ndArray
        Returns linear function using the input array
    """
    return a*x + b

def exponential(x, a, b, c):
    """
    Applies a exponential function to the input array

    Parameters
    ----------
    x: ndArray
        Input array
    
    a: float
        Amplitude
    
    b: float
        Exponential constant
    
    c: float 
        Offset constant

    Returns
    -------
    x: ndArray
        Returns exponential function using the input array
    """
    return a*np.exp(b*x) + c

def fowler_relationship(x, intensity, delta_param):
    """
    Applies the Fowler relationship to an input array. The equation of the Fowler relationship implemented assumes 0 dark current.

    Parameters
    ----------
    x: ndArray
        Input array
    
    intensity: float
        Intensity of irradiation
    
    delta_param: float
        Exponential linearity term

    Returns
    -------
    x: ndArray
        Returns induced current according to the Fowler relationship
    """ 
    return intensity*x**delta_param

def load_data(filename):
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

    mean_current_err: ndArray
        Returns numpy array containing the standard deviation of the mean current values from a run
    """
    # Import the raw data file as an array of strings
    data = np.loadtxt(filename, dtype=str, usecols=range(6),)

    # Locate elements of array associated to the first elemont of a header
    header_mask = data == "time"
    header_idx = np.argwhere(header_mask)[:, 0]

    # Remove all rows containing headers and convert array elements to floats
    processed_data = np.delete(data, header_idx, axis=0).astype(np.float32)

    # Create a mask for all negative time entries in the dataset
    negative_mask = processed_data[:,0] < 0.0
    negative_idx = np.argwhere(negative_mask)
    
    # Remove all entries from the dataset which contain negative time values     
    processed_data = np.delete(processed_data, negative_idx, axis=0).astype(np.float32)

    # Excract from the data the timiing and meant current datapoints
    time = processed_data[:, 0] 
    voltage = processed_data[:, 1]
    mean_current = processed_data[:, 2] 
    mean_current_err = processed_data[:, 3] 
    
    return time, voltage, mean_current, mean_current_err

def hypothesis_test_uniform(indexes, time, current, current_err):
    """
    Perform a uniform hypothesis test on the mean current of a run between two index ranges and return the fitted mean current
    and its associated error

    Parameters
    ----------
    indexes: tuple of int
        Tuple containing the array indexes for the range of mean current to be fitted   

    time: ndArray
        Numpy array containing processed time values from a run

    mean_current: ndArray
        Numpy array containing processed mean current values from a run

    mean_current_err: ndArray
        Numpy array containing the standard deviation of the mean current values from a run


    Returns
    -------
    fitted_curr: float
        The value for the fitted mean current within the indicated range
    
    fitted_curr_err: float
        The value for the fitted mean current error within the indicated range.
    """
    # Unpack the index values representing the range of the fit
    idx1, idx2 = indexes

    # Obtain the time, mean current and mean current error values within the indicated range
    time, current, current_err = time[idx1:idx2], current[idx1:idx2], current_err[idx1:idx2] if current_err is not None else None

    # Preform the uniform fit
    popt, pcov = curve_fit(uniform, time, current, sigma=None)

    # Extract fitted mean current and it's associated error
    fitted_curr = popt[-1]
    fitted_curr_err = np.sqrt(np.diag(pcov))[-1]

    return fitted_curr, fitted_curr_err

def hypothesois_test_exponential(indexes, time, current, current_err):
    """
    Perform an exponential hypothesis test on the mean current of a run between two index ranges and return the fitted exponential constant
    and its associated error

    Parameters
    ----------
    indexes: tuple of int
        Tuple containing the array indexes for the range of mean current to be fitted   

    time: ndArray
        Numpy array containing processed time values from a run

    mean_current: ndArray
        Numpy array containing processed mean current values from a run

    mean_current_err: ndArray
        Numpy array containing the standard deviation of the mean current values from a run


    Returns
    -------
    time_const: float
        The value for the fitted time const within the indicated range
    
    time_const_err: float
        The value for the fitted time const error within the indicated range.
    """
    
    # Unpack the index values representing the range of the fit
    idx1, idx2 = indexes

    # Obtain the time, mean current and mean current error values within the indicated range
    time, current, current_err = time[idx1:idx2], current[idx1:idx2], current_err[idx1:idx2] if current_err is not None else None

    # Preform the uniform fit
    popt, pcov = curve_fit(exponential, time, current, sigma=None)

    time_const = popt[1]
    time_const_err = np.sqrt(np.diag(pcov))[1]

    # Extract fitted time const and it's associated error
    return time_const, time_const_err

def hypothesis_test_xray_intensity(voltage, current, current_err):
    """
    Perform an induced current hypothesis test according to the fowler relationship.

    Parameters
    ----------
    voltage: ndArray
        Numpy array containing the bias voltages applied to the SC-diamonds`  

    current: ndArray
        Numpy array containing the induced currents measured from the SC-diamonds

    current_err: ndArray
        Numpy array containing the standard deviation of the induced currents measured from the SC-diamonds


    Returns
    -------
    intensity: float
        The value for the fitted intensity
    
    intensity_err: float
        The value for the fitted intensity error
    
    delta_param: float
        The value for the delta parameter from the fitted Fowler relationship
    
    delta_param_err: float
        The value for the delta parameter error from the fitted Fowler relationship
    """ 
    # Apply fit to Fowler relationship
    popt, pcov = curve_fit(fowler_relationship, voltage, current, sigma=current_err)
    # Extract the fitted intensity and delta parameter
    intensity, delta_param = popt
    # Extract the errors for the fitting parameters
    intensity_err, delta_param_err = np.sqrt(np.diag(pcov))

    return intensity, intensity_err, delta_param, delta_param_err


def process_current_intensity(
        time, mean_current, mean_current_err, base_idx, xray_idx, xray_curr=None, xray_curr_err=None, scaling_param=1.0
    ):
    """
        Extract the fitted values for the induced currents for runs where the instensity of the irradiation is being modified.
        This function will return arrays containing the induced current for each intensity. The function can automatically append the 
        indiced current values from two seperat runs together.

        Parameters
        ----------
        time: ndArray
            Numpy array containing processed time values from a run

        mean_current: ndArray
            Numpy array containing processed mean current values from a run

        mean_current_err: ndArray
            Numpy array containing the standard deviation of the mean current values from a run 
        
        base_idx: ndArray or tuple of ints
            Array or tuple containing a pair of indexes which represent the ranges for the currents corresponding to the dark current
            (ie: no irradiation) of the diamond.
        
        xray_idx: ndArray or tuple of ints
            Array or tuple containing the indexes which represent the ranges of indexes for the currents corresponding to 
            measured induced current at different intensities. The size of the array or tuple must be (n, 2) where n is the 
            number induced current measurments for each different intensity measured. 
            The 2 elements are the indexes which represent in the following order:
                - Index of the mean current array for the start of the induced current measurment
                - Index of the mean current array for the end of the induced current measurment
        
        xray_curr: ndArray, optional
            Array containing previous induced current values obtained from a previous instance of this function. If an array if 
            provided, the indiced current values from this intance of the function will be appended to the xray_curr
            array and returned to the user. If no array is passed than an empty array will be intanciated and used to store the induced 
            current values computed in this function.

        xray_curr_err: ndArray, optional
            Array containing previous induced current error values obtained from a previous instance of this function. If an array if 
            provided, the indiced current error values from this intance of the function will be appended to the xray_curr_err
            array and returned to the user. If no array is passed than an empty array will be intanciated and used to store the induced 
            current error values computed in this function.
        
        scaling_param: float, optional
            Scaling parameter which will be applied to the final fitted current values. This can be used to convert all 
            current outputs to a particular unit.

        Returns
        -------
        baseline_curr_hv_linearity: ndArray
            Array containing the baseline current values calculated for all index ranges provided. If a previous baseline_curr_hv_linearity
            array was passed as an input to this function, the baseline current values calculated will be appended to the passed array.

        baseline_curr_err_hv_linearity: ndArray
            Array containing the baseline current error values calculated for all index ranges provided. If a previous baseline_curr_err_hv_linearity
            array was passed as an input to this function, the baseline current error values calculated will be appended to the passed array.
        
        delta_curr_hv_linearity: ndArray
            Array containing the delta current values calculated for all index ranges provided. If a previous delta_curr_hv_linearity
            array was passed as an input to this function, the delta current values calculated will be appended to the passed array.

        delta_curr_err_hv_linearity: ndArray
            Array containing the delta current error values calculated for all index ranges provided. If a previous delta_curr_error_hv_linearity
            array was passed as an input to this function, the delta current error values calculated will be appended to the passed array.
        """ 
    # Apply a uniform fit to obtain the baseline current 
    base_curr, base_curr_err = hypothesis_test_uniform(
    base_idx, time, mean_current, mean_current_err
    )
    # Check if a xray current array was passed
    if xray_curr is None: 
        # Initialise empty array
        xray_curr = []
    else: 
        # Convert passed array to a list
        xray_curr = xray_curr.tolist()
    
    # Check if a xray current error array was passed
    if xray_curr_err is None: 
        # Initialise empty array
        xray_curr_err = []
    else: 
        # Convert passed array to a list
        xray_curr_err = xray_curr_err.tolist()


    # Iterate over index values for each xray intensity measured
    for idxs in xray_idx:
        # Unpack index pairs for induced current
        idx1, idx2 = idxs
        # Apply a uniform fit to obtain the induced current 
        _curr, _curr_err = hypothesis_test_uniform(
            (idx1,idx2), time, mean_current, mean_current_err
        )
        # Append delta current to array
        xray_curr.append(_curr-base_curr)
        # Append the propogated delta cuttent error to array
        xray_curr_err.append(_curr_err+base_curr_err)

    # Apply scaling parameter to calculated currents
    xray_curr = scaling_param * np.array(xray_curr)
    xray_curr_err = scaling_param * np.array(xray_curr_err)
    
    return xray_curr, xray_curr_err

def process_current_irradiation(
        time, mean_current, mean_current_err, voltage_idx, use_abs=True, scaling_param=1.0, baseline_curr_hv_linearity=None,
        baseline_curr_err_hv_linearity=None, delta_curr_hv_linearity=None, delta_curr_err_hv_linearity=None,
    ):
    """
    Extract the fitted values for the induced currents and the dark currents for a number of ranges within a specifoc run.
    This function will return arrays containing the induced current and dark currents. The function can automatically append the 
    indiced current and dark current valus from two seperat runs together.

    Parameters
    ----------
    time: ndArray
        Numpy array containing processed time values from a run

    mean_current: ndArray
        Numpy array containing processed mean current values from a run

    mean_current_err: ndArray
        Numpy array containing the standard deviation of the mean current values from a run 
    
    voltage_idx: ndArray or tuple
        Array or tuple containing the indexes which represent the ranges of indexes for the currents corresponding to 
        the measured dark current and measured induced current. The size of the array or tuple must be (n, 4) where n is the 
        number of dark current & induced current measurments for each bias voltage applied. 
        The 4 elements are the indexes which represent in the following order:
            - Index of the mean current array for the start of the dark current measurment
            - Index of the mean current array for the end of the dark current measurment
            - Index of the mean current array for the start of the induced current measurment
            - Index of the mean current array for the end of the induced current measurment

    use_abs: bool, optional
        If true the absolute values of the dark current and induced current will be used to compute their difference
    
    scaling_param: float, optional
        Scaling parameter which will be applied to the final fitted current values. This can be used to convert all 
        current outputs to a particular unit.
    
    baseline_curr_hv_linearity: ndArray, optional
        Array containing previous baseline current values obtained from a previous instance of this function. If an array if 
        provided, the baseline current values from this intance of the function will be appended to the baseline_curr_hv_linearity
        array and returned to the user. If no array is passed than an empty array will be intanciated and used to store the baseline 
        current values computed in this function.

    baseline_curr_err_hv_linearity: ndArray, optional
        Array containing previous baseline current error values obtained from a previous instance of this function. If an array if 
        provided, the baseline current error values from this intance of the function will be appended to the baseline_curr_err_hv_linearity
        array and returned to the user. If no array is passed than an empty array will be intanciated and used to store the baseline 
        current error values computed in this function.
    
    delta_curr_hv_linearity: ndArray, optional
        Array containing previous delta current values obtained from a previous instance of this function. If an array if 
        provided, the delta current values from this intance of the function will be appended to the delta_curr_hv_linearity
        array and returned to the user. If no array is passed than an empty array will be intanciated and used to store the delta 
        current values computed in this function.

    delta_curr_err_hv_linearity: ndArray, optional
        Array containing previous delta current error values obtained from a previous instance of this function. If an array if 
        provided, the delta current error values from this intance of the function will be appended to the delta_curr_err_hv_linearity
        array and returned to the user. If no array is passed than an empty array will be intanciated and used to store the delta
        current error values computed in this function.

    Returns
    -------
    xray_curr: ndArray, 
        Array containing the baseline current values calculated for all index ranges provided. If a previous baseline_curr_hv_linearity
        array was passed as an input to this function, the baseline current values calculated will be appended to the passed array.

    xray_curr_err: ndArray, 
        Array containing the baseline current error values calculated for all index ranges provided. If a previous baseline_curr_err_hv_linearity
        array was passed as an input to this function, the baseline current error values calculated will be appended to the passed array.
    
    delta_curr_hv_linearity: ndArray,
        Array containing the delta current values calculated for all index ranges provided. If a previous delta_curr_hv_linearity
        array was passed as an input to this function, the delta current values calculated will be appended to the passed array.

    delta_curr_err_hv_linearity: ndArray,
        Array containing the delta current error values calculated for all index ranges provided. If a previous delta_curr_error_hv_linearity
        array was passed as an input to this function, the delta current error values calculated will be appended to the passed array.
    """ 

    # Check if a baseline current array was passed
    if baseline_curr_hv_linearity is None:
        # Initialise empty array
        baseline_curr_hv_linearity = []
    else:
        # Convert passed array to a list
        baseline_curr_hv_linearity = baseline_curr_hv_linearity.tolist()

    # Check if a baseline current error array was passed
    if baseline_curr_err_hv_linearity is None:
        # Initialise empty array
        baseline_curr_err_hv_linearity = []
    else:
        # Convert passed array to a list
        baseline_curr_err_hv_linearity = baseline_curr_err_hv_linearity.tolist()
    
    # Check if a delta current array was passed
    if delta_curr_hv_linearity is None:
        # Initialise empty array
        delta_curr_hv_linearity = []
    else:
        # Convert passed array to a list
        delta_curr_hv_linearity = delta_curr_hv_linearity.tolist()

    # Check if a delta current error array was passed
    if delta_curr_err_hv_linearity is None:
        # Initialise empty array
        delta_curr_err_hv_linearity = []
    else:
        # Convert passed array to a list
        delta_curr_err_hv_linearity = delta_curr_err_hv_linearity.tolist()


    # Iterate over index values for each bias voltage measured
    for idxs in voltage_idx:
        # Unpack indexes for specific bias voltage measurment
        idx1, idx2, idx3, idx4 = idxs

        # Apply a uniform fit to obtain the baseline current 
        base_curr, base_curr_err = hypothesis_test_uniform(
            (idx1,idx2), time, mean_current, mean_current_err 
        )
        # Apply a uniform current to obtain the peak current 
        peak_curr, peak_curr_err = hypothesis_test_uniform(
            (idx3,idx4), time, mean_current, mean_current_err
        )
        # Compute delta current by subtracting baseline current from peak current
        # Use absolute value of subtraction if use_abs is True
        if use_abs:
            delta_curr = abs(peak_curr-base_curr)
        else:
            delta_curr = peak_curr - base_curr
        # Append delta current to the array
        delta_curr_hv_linearity.append(delta_curr)
        # Append the propogated delta cuttent error to array
        delta_curr_err_hv_linearity.append(abs(peak_curr_err)+abs(base_curr_err))
        # Append baseline current to the array
        baseline_curr_hv_linearity.append(base_curr)
        # Append baseline current error to the array
        baseline_curr_err_hv_linearity.append(base_curr_err)

    # Apply the scaling parameter to all current measurments calulated 
    baseline_curr_hv_linearity = scaling_param * np.array(baseline_curr_hv_linearity)
    baseline_curr_err_hv_linearity = scaling_param * np.array(baseline_curr_err_hv_linearity)
    delta_curr_hv_linearity = scaling_param * np.array(delta_curr_hv_linearity)
    delta_curr_err_hv_linearity = scaling_param * np.array(delta_curr_err_hv_linearity)

    return baseline_curr_hv_linearity, baseline_curr_err_hv_linearity, delta_curr_hv_linearity, delta_curr_err_hv_linearity

def process_current_irradiation_alpha(
        time, mean_current, mean_current_err, voltage_idx, key, delta_curr_dict, delta_curr_err_dict,
        baseline_curr_dict, baseline_curr_err_dict, time_const_dict=None, time_const_err_dict=None, scaling_param=1.0
):
    """
    Extract fitted values for induced currents and the dark currents for alpha irradiation measurments. The function is designed
    specifically for measurments for runs where a bias voltage of 0V, 50V and 200V were taken. The function is capable of processing 
    measurments where for each bias voltage only 1, 2 or 3 index range pairs are given. Each iondex pair range would represent the
    baseline, overshoot decay and the peak respectivly. Current implementation of the function will extract the baseline and delta current
    for each bias voltage measurment respectivly.

    Parameters
    ----------
    time: ndArray
        Numpy array containing processed time values from a run

    mean_current: ndArray
        Numpy array containing processed mean current values from a run

    mean_current_err: ndArray
        Numpy array containing the standard deviation of the mean current values from a run 
    
    voltage_idx: ndArray or tuple
        Array or tuple containing the indexes which represent the ranges of indexes for the currents corresponding to 
        the measured dark current and measured induced current. The size of the array or tuple must be either (3, 2), (3, 4) or (3,6)
        where 3 represents the number of bias voltage measurments (ie: 0V, 50V and 200V). 
        If the tuple contains has 2 indexes per bias voltage measurment, the elements represent the the indexes of:
            - Index of the mean current array for the start of the dark current measurment
            - Index of the mean current array for the end of the dark current measurment
        If the tuple contains has 4 indexes per bias voltage measurment, the elements represent the the indexes of:
            - Index of the mean current array for the start of the dark current measurment
            - Index of the mean current array for the end of the dark current measurment 
            - Index of the mean current array for the start of the peak current measurment
            - Index of the mean current array for the end of the peak current measurment 
        If the tuple contains has 6 indexes per bias voltage measurment, the elements represent the the indexes of:
            - Index of the mean current array for the start of the dark current measurment
            - Index of the mean current array for the end of the dark current measurment 
            - Index of the mean current array for the start of the overshoot current measurment
            - Index of the mean current array for the end of the overshoot current measurment
            - Index of the mean current array for the start of the peak current measurment
            - Index of the mean current array for the end of the peak current measurment 
    key: string
        Key used for each current dictionary where the fitted currents computed by this function will be stored 
    
    baseline_curr_dict: dictionary of ndArrays
        Dictionary where the baseline current measurments calculated by this function will be stored. The structure of this 
        dictionary be:
            {   "0":  
                    "key",
                    ...
                ,
                "50":
                    "key",
                    ...
               ,
                "200":
                    "key",
                    ...       
            }
        Where key represents the string passed to the key parameter for the input of this function. This structure allows for 
        the same dictionary to be passed to this function in order to store the current measurments from multiple diamonds
        in the same object

    baseline_curr_err_dict: dictionary of ndArrays
         Dictionary where the baseline current error measurments calculated by this function will be stored. See the entry of
         baseline_curr_dict for an explanation of the dictionary's structure
    
    delta_curr_dict: dictionary of ndArrays
         Dictionary where the delta current measurments calculated by this function will be stored. See the entry of
         baseline_curr_dict for an explanation of the dictionary's structure

    delta_curr_err_dict: dictionary of ndArrays
         Dictionary where the delta current error measurments calculated by this function will be stored. See the entry of
         baseline_curr_dict for an explanation of the dictionary's structure

    scaling_param: float, optional
        Scaling parameter which will be applied to the final fitted current values. This can be used to convert all 
        current outputs to a particular units

    Returns
    -------
    baseline_curr_dict: dictionary of ndArrays
        Dictionary where the baseline current measurments calculated by this function will be stored. The structure of this 
        dictionary be:
            {   "0":  
                    "key",
                    ...
                ,
                "50":
                    "key",
                    ...
               ,
                "200":
                    "key",
                    ...       
            }
        Where key represents the string passed to the key parameter for the input of this function. This structure allows for 
        the same dictionary to be passed to this function in order to store the current measurments from multiple diamonds
        in the same object

    baseline_curr_err_dict: dictionary of ndArrays
         Dictionary where the baseline current error measurments calculated by this function will be stored. See the entry of
         baseline_curr_dict for an explanation of the dictionary's structure
    
    delta_curr_dict: dictionary of ndArrays
         Dictionary where the delta current measurments calculated by this function will be stored. See the entry of
         baseline_curr_dict for an explanation of the dictionary's structure

    delta_curr_err_dict: dictionary of ndArrays
         Dictionary where the delta current error measurments calculated by this function will be stored. See the entry of
         baseline_curr_dict for an explanation of the dictionary's structure
    """ 
    # Iterate for all bias voltage mmeasurments
    for i, idxs  in enumerate(voltage_idx):

        # If only one index pair for the bias voltage measurment
        if len(idxs) == 2:
            # Unpack the index pairs
            idx1, idx2, = idxs
        # If only two index pair for the bias voltage measurment
        elif len(idxs) == 4:
            # Unpack the index pairs
            idx1, idx2, idx3, idx4 = idxs
        # If only three index pair for the bias voltage measurment
        elif len(idxs) == 6:
            # Unpack the index pairs
            idx1, idx2, _, _, idx3, idx4 = idxs


        # Compute currents for first bias voltage measurment (Assumed to be 0V)
        if i==0:
            # Apply a uniform fit to obtain the baseline current 
            base_curr, base_curr_err = hypothesis_test_uniform(
                (idx1,idx2), time, mean_current, mean_current_err 
            )

            # Save rhe fitted current value and their corresponding errors to their respective dictionary
            delta_curr_dict["0"][key] = base_curr
            delta_curr_err_dict["0"][key] = base_curr_err
            baseline_curr_dict["0"][key] = base_curr
            baseline_curr_err_dict["0"][key] = base_curr_err

        if i==1:
            # Apply a uniform fit to obtain the baseline current 
            base_curr, base_curr_err = hypothesis_test_uniform(
                (idx1,idx2), time, mean_current, mean_current_err
            )
            # Apply a uniform fit to obtain the peak current 
            peak_curr, peak_curr_err = hypothesis_test_uniform(
                (idx3,idx4), time, mean_current, mean_current_err
            )

            # Save rhe fitted current value and their corresponding errors to their respective dictionary
            delta_curr_dict["50"][key] = abs(peak_curr-base_curr)
            delta_curr_err_dict["50"][key] = abs(peak_curr_err)+abs(base_curr_err)
            baseline_curr_dict["50"][key] = base_curr
            baseline_curr_err_dict["50"][key] = base_curr_err

        if i==2:
            # Apply a uniform fit to obtain the baseline current 
            base_curr, base_curr_err = hypothesis_test_uniform(
                (idx1,idx2), time, mean_current, mean_current_err
            )
            # Apply a uniform fit to obtain the peak current 
            peak_curr, peak_curr_err = hypothesis_test_uniform(
                (idx3,idx4), time, mean_current, mean_current_err
            )

            # Save rhe fitted current value and their corresponding errors to their respective dictionary
            delta_curr_dict["200"][key] = abs(peak_curr-base_curr)
            delta_curr_err_dict["200"][key] = abs(peak_curr_err)+abs(base_curr_err)
            baseline_curr_dict["200"][key] = base_curr
            baseline_curr_err_dict["200"][key] = base_curr_err

    return delta_curr_dict, delta_curr_err_dict, baseline_curr_dict, baseline_curr_err_dict

def plot_intensity_ranges(time, mean_current, base_idx, xray_idx):
    """
        Plot the mean current of a run with the current ranges corresponding to the passed indexes highlighted with different 
        colours. This function is desigend for linearity measurments where the intensity of xray tube is being modified throughout the run.

        Parameters
        ----------
        time: ndArray
            Numpy array containing processed time values from a run

        mean_current: ndArray
            Numpy array containing processed mean current values from a run

        base_idx: ndArray or tuple of ints
            Array or tuple containing a pair of indexes which represent the ranges for the currents corresponding to the dark current
            (ie: no irradiation) of the diamond.
        
        xray_idx: ndArray or tuple of ints
            Array or tuple containing the indexes which represent the ranges of indexes for the currents corresponding to 
            measured induced current at different intensities. The size of the array or tuple must be (n, 2) where n is the 
            number induced current measurments for each different intensity measured. 
            The 2 elements are the indexes which represent in the following order:
                - Index of the mean current array for the start of the induced current measurment
                - Index of the mean current array for the end of the induced current measurment
        """ 
    # Plot the current range corresponding to the baseline
    plt.plot(time[base_idx[0]:base_idx[1]], mean_current[base_idx[0]:base_idx[1]])

    # Iterate over induced current ranges for each intensity 
    for idx in xray_idx:
        # Plot the current range corresponding to the induced current
        plt.plot(time[idx[0]:idx[1]], mean_current[idx[0]:idx[1]])

    # Plot the full mean current for the run
    plt.plot(time, mean_current, alpha=0.2)

def plot_irradiation_ranges(time, mean_current, voltage_idx):
    """
        Plot the mean current of a run with the current ranges corresponding to the passed indexes highlighted with different 
        colours. This function is desigend for linearity measurments where the bias voltage applied ot the diamond is being 
        modified throughout the run 

        Parameters
        ----------
        time: ndArray
            Numpy array containing processed time values from a run

        mean_current: ndArray
            Numpy array containing processed mean current values from a run
        
        voltage_idx: ndArray or tuple
            Array or tuple containing the indexes which represent the ranges of indexes for the currents corresponding to 
            the measured dark current and measured induced current. The size of the array or tuple must be either (3, 2), (3, 4) or (3,6)
            where 3 represents the number of bias voltage measurments (ie: 0V, 50V and 200V). 
            If the tuple contains has 2 indexes per bias voltage measurment, the elements represent the the indexes of:
                - Index of the mean current array for the start of the dark current measurment
                - Index of the mean current array for the end of the dark current measurment
            If the tuple contains has 4 indexes per bias voltage measurment, the elements represent the the indexes of:
                - Index of the mean current array for the start of the dark current measurment
                - Index of the mean current array for the end of the dark current measurment 
                - Index of the mean current array for the start of the peak current measurment
                - Index of the mean current array for the end of the peak current measurment 
            If the tuple contains has 6 indexes per bias voltage measurment, the elements represent the the indexes of:
                - Index of the mean current array for the start of the dark current measurment
                - Index of the mean current array for the end of the dark current measurment 
                - Index of the mean current array for the start of the overshoot current measurment
                - Index of the mean current array for the end of the overshoot current measurment
                - Index of the mean current array for the start of the peak current measurment
                - Index of the mean current array for the end of the peak current measurment 
        """ 
    # Iterate for all bias voltage mmeasurments
    for i, idx in enumerate(voltage_idx):
        # Plot the range for each current measurments depending if the tuple contains 1, 2 or 3 index pairs
        if len(idx) == 2:
            plt.plot(time[idx[0]:idx[1]], mean_current[idx[0]:idx[1]])
        elif len(idx)==4:
            plt.plot(time[idx[0]:idx[1]], mean_current[idx[0]:idx[1]])
            plt.plot(time[idx[2]:idx[3]], mean_current[idx[2]:idx[3]])
        else:
            plt.plot(time[idx[0]:idx[1]], mean_current[idx[0]:idx[1]])
            plt.plot(time[idx[2]:idx[3]], mean_current[idx[2]:idx[3]])
            plt.plot(time[idx[4]:idx[5]], mean_current[idx[4]:idx[5]])

    # Plot the full mean current of the run
    plt.plot(time, mean_current, alpha =0.2)
