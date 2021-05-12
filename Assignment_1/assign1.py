import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import probplot


PLOT_DIR = os.sep.join([os.path.dirname(__file__), 'plots'])


def assign1(data_file: str, headless: bool = False):
    """
    This function will complete all questions, generating plots to the 'plots'
    subdirectory, outputting question information to the screen as well as a
    log file if included.

    Args:
        data_file: string indicating the name of the originating data file.
        headless: Boolean value to inhibit displaying plots. Plots will still be
                  saved to the the global PLOT_DIR directory.

    Returns: None

    """

    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    sns.plotting_context("paper")
    sns.set_style("darkgrid")

    timeseries_data = question_1(data_file, ylim=(-3, 3), xlim=(0, 0.5))
    sample_rate, num_samples, mean_voltage = question_2(timeseries_data, data_file)
    timeseries_data, standard_deviation, variance = question_3(timeseries_data, mean_voltage)
    max_amplitude, stddev_ratio = question_4(timeseries_data, standard_deviation)
    question_5(timeseries_data)
    question_6(timeseries_data)

    if not headless:
        plt.show()


def question_1(data_file: str,
               ylim: tuple = None,
               xlim: tuple = None) -> pd.DataFrame:
    """

    Args:
        data_file:
        headless: Boolean value to inhibit displaying plots. Plots will still be
                  saved to the the global PLOT_DIR directory.
        ylim: If provided, plots will scale the y-axis to the bounds specified in a (bottom, upper) tuple.
        xlim: If provided, plots will scale the x-axis to the bounds specified in a (bottom, upper) tuple.

    Returns: The read in timeseries data in a pandas DataFrame.

    """
    header_list = ['time', 'voltage', 'unknown1', 'unknown2']
    # Read in the data
    timeseries_data = pd.read_csv(data_file, names=header_list)
    # Handle removing extraneous data.
    timeseries_data.drop(columns=['unknown1', 'unknown2'], inplace=True)

    # Plot the timeseries data:
    xlabel = 'time (s)'
    ylabel = 'voltage (V)'
    title = f'Assignment 1, Question 1\nTimeseries of {os.path.basename(data_file)}'

    plt.figure()

    lineplot = sns.lineplot(x='time', y='voltage', data=timeseries_data)
    lineplot.set(xlabel=xlabel, ylabel=ylabel, title=title)
    lineplot.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    lineplot.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    lineplot.grid(b=True, which='minor', linewidth=0.5, linestyle=':')
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    lineplot.figure.savefig(PLOT_DIR + os.sep + f'question_1_{os.path.basename(data_file)}.png')

    return timeseries_data


def question_2(timeseries_data: pd.DataFrame, data_file: str):

    num_samples = len(timeseries_data)
    sample_rate = (num_samples - 1) / (timeseries_data['time'].iloc[-1] - timeseries_data['time'].iloc[0])
    mean_voltage = np.mean(timeseries_data['voltage'])

    return sample_rate, num_samples, mean_voltage


def question_3(timeseries_data: pd.DataFrame, mean_voltage):
    timeseries_data['zero mean voltage'] = timeseries_data['voltage'] - mean_voltage
    standard_deviation = np.std(timeseries_data['zero mean voltage'])
    variance = standard_deviation ** 2

    return timeseries_data, standard_deviation, variance


def question_4(timeseries_data, standard_deviation):
    max_amplitude = np.max(np.abs(timeseries_data['zero mean voltage']))
    stddev_ratio = max_amplitude / standard_deviation

    return max_amplitude, stddev_ratio


def question_5(timeseries_data: pd.DataFrame, headless=True):
    plt.figure()
    sns.histplot(data=timeseries_data['zero mean voltage'], stat='density', binwidth=0.1)


def question_6(timeseries_data: pd.DataFrame):
    fig, ax = plt.subplots()
    res = probplot(timeseries_data['zero mean voltage'], plot=ax)
