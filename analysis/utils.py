import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def uniform(x, a):
    return 0*x + a

def linear(x, a, b):
    return a*x + b

def exponential(x, a, b, c):
    return a*np.exp(b*x) + c

def xray_current_hypothesis(x, intensity, fitting_param):
    return intensity*x**fitting_param

def load_data(filename):
    """
    Read, load and process data and return arrays containing the time, voltage and mean current readings of the run
    """
    # Import the raw data file as an array of strings
    data = np.loadtxt(filename, dtype=str, usecols=range(6),)

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
    voltage = processed_data[:, 1]
    mean_current = processed_data[:, 2] 
    mean_current_err = processed_data[:, 3] 
    
    return time, voltage, mean_current, mean_current_err


def hypothesis_test_uniform(indexes, time, current, current_err):

    idx1, idx2 = indexes

    time, current, current_err = time[idx1:idx2], current[idx1:idx2], current_err[idx1:idx2] if current_err is not None else None

    popt, pcov = curve_fit(uniform, time, current, sigma=None)

    fitted_curr = popt[-1]
    fitted_curr_err = np.sqrt(np.diag(pcov))[-1]

    return fitted_curr, fitted_curr_err

def hypothesois_test_exponential(indexes, time, current, current_err):

    idx1, idx2 = indexes

    time, current, current_err = time[idx1:idx2], current[idx1:idx2], current_err[idx1:idx2] if current_err is not None else None

    popt, pcov = curve_fit(exponential, time, current, sigma=None)

    time_const = popt[1]
    time_const_err = np.sqrt(np.diag(pcov))[1]

    return time_const, time_const_err

def hypothesis_test_xray_intensity(voltage, current, current_err):

    popt, pcov = curve_fit(xray_current_hypothesis, voltage, current, sigma=current_err)
    intensity, fitting_param = popt
    intensity_err, fitting_param_err = np.sqrt(np.diag(pcov))

    return intensity, intensity_err, fitting_param, fitting_param_err


def process_current_intensity(time, mean_current, mean_current_err, base_idx, xray_idx, xray_curr=None, xray_curr_err=None, scaling_param=1.0):
    base_curr, base_curr_err = hypothesis_test_uniform(
    base_idx, time, mean_current, mean_current_err
    )
    if xray_curr is None: 
        xray_curr = []
    else: 
        xray_curr = xray_curr.tolist()
    
    if xray_curr_err is None: 
        xray_curr_err = []
    else: 
        xray_curr_err = xray_curr_err.tolist()


    for idxs in xray_idx:
        idx1, idx2 = idxs
        _curr, _curr_err = hypothesis_test_uniform(
            (idx1,idx2), time, mean_current, mean_current_err
        )
        xray_curr.append(_curr-base_curr)
        xray_curr_err.append(_curr_err+base_curr_err)

    xray_curr = scaling_param * np.array(xray_curr)
    xray_curr_err = scaling_param * np.array(xray_curr_err)
    
    return xray_curr, xray_curr_err

def process_current_irradiation(
        time, mean_current, mean_current_err, voltage_idx, use_abs=True, scaling_param=1.0, baseline_curr_hv_linearity=None,
        baseline_curr_err_hv_linearity=None, delta_curr_hv_linearity=None, delta_curr_err_hv_linearity=None,
    ):

    if baseline_curr_hv_linearity is None:
        baseline_curr_hv_linearity = []
    else:
        baseline_curr_hv_linearity = baseline_curr_hv_linearity.tolist()

    if baseline_curr_err_hv_linearity is None:
        baseline_curr_err_hv_linearity = []
    else:
        baseline_curr_err_hv_linearity = baseline_curr_err_hv_linearity.tolist()
    
    if delta_curr_hv_linearity is None:
        delta_curr_hv_linearity = []
    else:
        delta_curr_hv_linearity = delta_curr_hv_linearity.tolist()

    if delta_curr_err_hv_linearity is None:
        delta_curr_err_hv_linearity = []
    else:
        delta_curr_err_hv_linearity = delta_curr_err_hv_linearity.tolist()


    for idxs in voltage_idx:
        idx1, idx2, idx3, idx4 = idxs

        base_curr, base_curr_err = hypothesis_test_uniform(
            (idx1,idx2), time, mean_current, mean_current_err 
        )

        peak_curr, peak_curr_err = hypothesis_test_uniform(
            (idx3,idx4), time, mean_current, mean_current_err
        )

        if use_abs:
            delta_curr_hv_linearity.append(abs(peak_curr)-abs(base_curr))
        else:
            delta_curr_hv_linearity.append(peak_curr - base_curr)

        delta_curr_err_hv_linearity.append(abs(peak_curr_err)+abs(base_curr_err))

        baseline_curr_hv_linearity.append(base_curr)
        baseline_curr_err_hv_linearity.append(base_curr_err)

    baseline_curr_hv_linearity = scaling_param * np.array(baseline_curr_hv_linearity)
    baseline_curr_err_hv_linearity = scaling_param * np.array(baseline_curr_err_hv_linearity)
    delta_curr_hv_linearity = scaling_param * np.array(delta_curr_hv_linearity)
    delta_curr_err_hv_linearity = scaling_param * np.array(delta_curr_err_hv_linearity)

    return baseline_curr_hv_linearity, baseline_curr_err_hv_linearity, delta_curr_hv_linearity, delta_curr_err_hv_linearity

def process_current_irradiation_alpha(
        time, mean_current, mean_current_err, voltage_idx, key, delta_curr_dict, delta_curr_err_dict,
        baseline_curr_dict, baseline_curr_err_dict, time_const_dict=None, time_const_err_dict=None, scaling_param=1.0
):
    
    for i, idxs  in enumerate(voltage_idx):

        if len(idxs) == 2:
            idx1, idx2, = idxs

        elif len(idxs) == 4:
            idx1, idx2, idx3, idx4 = idxs

        elif len(idxs) == 6:
            idx1, idx2, _, _, idx3, idx4 = idxs


        if i==0:
            base_curr, base_curr_err = hypothesis_test_uniform(
                (idx1,idx2), time, mean_current, mean_current_err 
            )

            delta_curr_dict["0"][key] = base_curr
            delta_curr_err_dict["0"][key] = base_curr_err
            baseline_curr_dict["0"][key] = base_curr
            baseline_curr_err_dict["0"][key] = base_curr_err

        if i==1:
            base_curr, base_curr_err = hypothesis_test_uniform(
                (idx1,idx2), time, mean_current, mean_current_err
            )
            peak_curr, peak_curr_err = hypothesis_test_uniform(
                (idx3,idx4), time, mean_current, mean_current_err
            )

            delta_curr_dict["50"][key] = abs(peak_curr)-abs(base_curr)
            delta_curr_err_dict["50"][key] = abs(peak_curr_err)+abs(base_curr_err)
            baseline_curr_dict["50"][key] = base_curr
            baseline_curr_err_dict["50"][key] = base_curr_err

        if i==2:
            base_curr, base_curr_err = hypothesis_test_uniform(
                (idx1,idx2), time, mean_current, mean_current_err
            )
            peak_curr, peak_curr_err = hypothesis_test_uniform(
                (idx3,idx4), time, mean_current, mean_current_err
            )

            delta_curr_dict["200"][key] = abs(peak_curr)-abs(base_curr)
            delta_curr_err_dict["200"][key] = abs(peak_curr_err)+abs(base_curr_err)
            baseline_curr_dict["200"][key] = base_curr
            baseline_curr_err_dict["200"][key] = base_curr_err

    return delta_curr_dict, delta_curr_err_dict, baseline_curr_dict, baseline_curr_err_dict

def plot_intensity_ranges(time, mean_current, base_idx, xray_idx):
    plt.plot(time[base_idx[0]:base_idx[1]], mean_current[base_idx[0]:base_idx[1]])

    for idx in xray_idx:
        plt.plot(time[idx[0]:idx[1]], mean_current[idx[0]:idx[1]])

    plt.plot(time, mean_current, alpha=0.2)

def plot_irradiation_ranges(time, mean_current, voltage_idx):
    for i, idx in enumerate(voltage_idx):
        if len(idx) == 2:
            plt.plot(time[idx[0]:idx[1]], mean_current[idx[0]:idx[1]])
        elif len(idx)==4:
            plt.plot(time[idx[0]:idx[1]], mean_current[idx[0]:idx[1]])
            plt.plot(time[idx[2]:idx[3]], mean_current[idx[2]:idx[3]])
        else:
            plt.plot(time[idx[0]:idx[1]], mean_current[idx[0]:idx[1]])
            plt.plot(time[idx[2]:idx[3]], mean_current[idx[2]:idx[3]])
            plt.plot(time[idx[4]:idx[5]], mean_current[idx[4]:idx[5]])


    plt.plot(time, mean_current, alpha =0.2)
