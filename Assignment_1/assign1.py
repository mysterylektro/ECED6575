"""
    -*- coding: utf-8 -*-
    Time    : 2021-05-11 7:53 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: assign1.py

    {Description}
    -------------

"""

import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import sounddevice as sd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import probplot
from scipy.signal import resample


PLOT_DIR = os.sep.join([os.path.dirname(__file__), 'plots'])
TEXTBOX_PROPS = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.5}
BIN_MODELS = {'fd': 'Freedman-Diaconis choice',
              'doane': 'Doane\'s formula',
              'scott': 'Scott\'s normal reference rule',
              'stone': 'Stone\'s formula',
              'rice': 'Rice rule',
              'sturges': 'Sturge\'s formula',
              'sqrt': 'Square-root choice'}

sns.plotting_context("paper")
sns.set_style("darkgrid")


def set_minor_gridlines(ax):
    """
    Helper function to set minor gridlines on seaborn plots.
    Args:
        ax: input plot axis

    Returns: None

    """
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='minor', linewidth=0.5, linestyle=':')


def assign1(data_file: str, headless: bool = True):
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

    data_name = os.path.splitext(os.path.basename(data_file))[0]

    """
    Question 1
    a) Read in the data into a Pandas DataFrame
    b) Generate a figure and plot of the timeseries data.
    """
    timeseries_data = question_1a(data_file)
    timeseries_figure, timeseries_plot = question_1b(timeseries_data,
                                                     data_name,
                                                     y_lim=(-3, 3),
                                                     x_lim=(0, 0.5))

    """
    Question 2
    a) Calculate the sample rate, number of samples, and average (mean) voltage value.
    b) Add those values to the timeseries plot generated in question 1
    """
    sample_rate, num_samples, mean_voltage = question_2a(timeseries_data, data_name)
    textbox = question_2b(timeseries_plot, sample_rate, num_samples, mean_voltage, data_name)

    """
    Question 3
    a) Generate zero-mean time series and calculate variance and standard deviation.
    b) Generate plot with standard deviation and variance.
    """
    timeseries_data, standard_deviation, variance = question_3a(timeseries_data, mean_voltage)

    # TODO: Generate plot and add text values.

    """
    # Question 4
    # a) Calculate max amplitude and standard deviation ratio
    # b) Append values to the zero-mean plot textbox.
    """
    max_amplitude, stddev_ratio = question_4a(timeseries_data, standard_deviation)
    # TODO: add text to plot

    """
    Question 5
    Generate histogram of the data for various bin width models.
    """
    # question_5(timeseries_data, data_name)

    """
    Question 6
    Generate normal probability plot
    """
    question_6(timeseries_data, data_name)
    # TODO: Comment on results

    """
    Question 7
    a) Resample (upsample) signal to 44100 Hz
    b) Play sound in speakers.
    """
    normalized_data, sample_rate, duration = question_7a(timeseries_data)
    # question_7b(normalized_data, sample_rate, duration)

    if not headless:
        plt.show()


def question_1a(data_file: str) -> pd.DataFrame:
    """

    Args:
        data_file: path to timeseries .csv file.

    Returns: The read in timeseries data in a pandas DataFrame.

    """
    header_list = ['time', 'voltage']
    # Read in the data
    timeseries_data = pd.read_csv(data_file, usecols=[0, 1], names=header_list)
    return timeseries_data


def question_1b(timeseries_data: pd.DataFrame,
                data_name: str,
                y_lim: tuple = None,
                x_lim: tuple = None):
    """
    Args:
        timeseries_data:
        data_name:
        y_lim: If provided, plots will scale the y-axis to the bounds specified in a (bottom, upper) tuple.
        x_lim: If provided, plots will scale the x-axis to the bounds specified in a (bottom, upper) tuple.

    Returns:

    """
    x_label = 'time (s)'
    y_label = 'voltage (V)'
    title = f'Assignment 1, Question 1\nTimeseries of {data_name}'

    fig = plt.figure()
    line_plot = sns.lineplot(data=timeseries_data, x='time', y='voltage')
    line_plot.set(xlabel=x_label, ylabel=y_label, title=title)
    set_minor_gridlines(line_plot)

    if y_lim is not None:
        plt.ylim(y_lim)

    if x_lim is not None:
        plt.xlim(x_lim)

    output_filename = PLOT_DIR + os.sep + f'question_1_{data_name}.png'
    line_plot.figure.savefig(output_filename)

    return fig, line_plot


def question_2a(timeseries_data: pd.DataFrame, data_name: str):
    # TODO: Introduce logging

    num_samples = len(timeseries_data)
    sample_rate = (num_samples - 1) / (timeseries_data['time'].iloc[-1] - timeseries_data['time'].iloc[0])
    mean_voltage = np.mean(timeseries_data['voltage'])

    return sample_rate, num_samples, mean_voltage


def question_2b(timeseries_axis,
                sample_rate,
                num_samples,
                mean_voltage,
                data_name):

    text_string = '\n'.join([r'$f_{s}=%.2f\ Hz$' % sample_rate,
                         r'$N=%d$' % num_samples,
                         r'$\mu=%.2f\ V$' % mean_voltage])
    text_obj = timeseries_axis.text(0.05, 0.95, text_string,
                                    transform=timeseries_axis.transAxes,
                                    fontsize=14, verticalalignment='top', bbox=TEXTBOX_PROPS)
    title = f'Assignment 1, Question 2\nTimeseries of {data_name}'
    timeseries_axis.set(title=title)
    output_filename = PLOT_DIR + os.sep + f'question_2_{data_name}.png'
    timeseries_axis.figure.savefig(output_filename)
    return text_obj


def question_3a(timeseries_data: pd.DataFrame, mean_voltage):
    timeseries_data['zero mean voltage'] = timeseries_data['voltage'] - mean_voltage
    standard_deviation = np.std(timeseries_data['zero mean voltage'])
    variance = standard_deviation ** 2

    return timeseries_data, standard_deviation, variance


def question_4a(timeseries_data, standard_deviation):
    max_amplitude = np.max(np.abs(timeseries_data['zero mean voltage']))
    stddev_ratio = max_amplitude / standard_deviation

    return max_amplitude, stddev_ratio


def question_5(timeseries_data: pd.DataFrame,
               data_name: str,
               bin_model: str = None):

    hist_plot_dir = f'{PLOT_DIR}{os.sep}histograms'
    if not os.path.exists(hist_plot_dir):
        os.makedirs(hist_plot_dir)

    x_label = 'zero-mean voltage bin (V)'
    y_label = 'probability density function'

    plt.figure()
    if bin_model is None:
        bin_models = BIN_MODELS
    else:
        try:
            bin_models = {bin_model: BIN_MODELS[bin_model]}
        except KeyError:
            raise ValueError("Invalid bin model name.")

    for bin_model, bin_title in bin_models.items():
        plt.clf()
        title = f'Assignment 1, Question 5\nSample distribution of zero-mean {data_name} data\n{bin_title}'
        histogram_plot = sns.histplot(data=timeseries_data['zero mean voltage'], stat='density', bins=bin_model)
        histogram_plot.set(xlabel=x_label, ylabel=y_label, title=title)
        set_minor_gridlines(histogram_plot)
        filename = f'{hist_plot_dir}{os.sep}question_5_{data_name}_{bin_model}.png'
        histogram_plot.figure.savefig(filename)


def question_6(timeseries_data: pd.DataFrame, data_name: str):
    fig, ax = plt.subplots()
    result = probplot(timeseries_data['zero mean voltage'], plot=ax)
    set_minor_gridlines(ax)
    filename = f'{PLOT_DIR}{os.sep}question_6_probplot_{data_name}.png'
    fig.savefig(filename)


def question_7a(timeseries_data: pd.DataFrame):
    sample_rate = 44100
    duration = timeseries_data['time'].iloc[-1] - timeseries_data['time'].iloc[0]
    resampled_data = resample(timeseries_data['zero mean voltage'], int(np.floor(duration * sample_rate)))
    normalized_data = resampled_data / np.max(resampled_data)
    return normalized_data, sample_rate, duration


def question_7b(normalized_data, sample_rate, duration):
    sd.play(normalized_data, sample_rate)
    time.sleep(duration)
    sd.stop()
