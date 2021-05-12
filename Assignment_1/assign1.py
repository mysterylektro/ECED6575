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
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import sounddevice as sd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import probplot, norm
from scipy.signal import resample


PLOT_DIR = os.sep.join([os.path.dirname(__file__), 'plots'])
TEXTBOX_PROPS = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.75}
BIN_MODELS = {'fd': 'Freedman-Diaconis choice',
              'doane': 'Doane\'s formula',
              'scott': 'Scott\'s normal reference rule',
              'stone': 'Stone\'s formula',
              'rice': 'Rice rule',
              'sturges': 'Sturge\'s formula',
              'sqrt': 'Square-root choice'}

sns.plotting_context("paper")
sns.set_style("darkgrid")


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
    """
    header_list = ['time', 'voltage']
    # Read in the data
    timeseries_data = pd.read_csv(data_file, usecols=[0, 1], names=header_list)

    # Generate a figure and plot of the timeseries data.
    timeseries_figure, timeseries_plot = plot_timeseries(timeseries_data['time'],
                                                         timeseries_data['voltage'],
                                                         title=f'Assignment 1, Question 1\nTimeseries of {data_name}',
                                                         x_label='time (s)',
                                                         y_label='voltage (V)',
                                                         y_lim=(-3, 3),
                                                         x_lim=(0, 0.5))

    # Save the plot to the plots directory
    timeseries_plot.figure.savefig(f'{PLOT_DIR}{os.sep}question_1_{data_name}.png')

    """
    Question 2
    """
    # Calculate the sample rate, number of samples, and average (mean) voltage value.
    num_samples = len(timeseries_data)
    sample_rate = (num_samples - 1) / (timeseries_data['time'].iloc[-1] - timeseries_data['time'].iloc[0])
    mean_voltage = np.mean(timeseries_data['voltage'])

    # Add those values to the timeseries plot generated in question 1
    text_string = '\n'.join([r'$f_{s}=%.2f\ Hz$' % sample_rate,
                             r'$N=%d$' % num_samples,
                             r'$\mu=%.2f\ V$' % mean_voltage])
    text_obj = timeseries_plot.text(0.05, 0.95, text_string,
                                    transform=timeseries_plot.transAxes,
                                    fontsize=14, verticalalignment='top', bbox=TEXTBOX_PROPS)
    # Change title of plot to reflect question number
    timeseries_plot.set(title=f'Assignment 1, Question 2\nTimeseries of {data_name}')

    # Save to plots directory.
    timeseries_plot.figure.savefig(f'{PLOT_DIR}{os.sep}question_2_{data_name}.png')

    """
    Question 3
    """
    # Generate zero-mean time series and calculate variance and standard deviation.
    timeseries_data['zero mean voltage'] = timeseries_data['voltage'] - mean_voltage
    standard_deviation = np.std(timeseries_data['zero mean voltage'])
    variance = standard_deviation ** 2

    # Generate timeseries plot.
    zm_timeseries_figure, zm_timeseries_plot = plot_timeseries(timeseries_data['time'],
                                                               timeseries_data['zero mean voltage'],
                                                               title=f'Assignment 1, Question 3\nZero-mean timeseries of {data_name}',
                                                               x_label='time (s)',
                                                               y_label='voltage (V)',
                                                               y_lim=(-3, 3),
                                                               x_lim=(0, 0.5))

    # Add text to plot.
    text_string = '\n'.join([r'$\sigma=%.2f\ V$' % standard_deviation,
                             r'$\sigma^2=%.2f\ V$' % variance])
    text_obj = zm_timeseries_plot.text(0.05, 0.95, text_string,
                                       transform=timeseries_plot.transAxes,
                                       fontsize=14, verticalalignment='top', bbox=TEXTBOX_PROPS)

    # Save the plot to the plots directory
    zm_timeseries_plot.figure.savefig(f'{PLOT_DIR}{os.sep}question_3_{data_name}.png')

    """
    Question 4
    """
    # Calculate max amplitude and standard deviation ratio
    max_amplitude = np.max(np.abs(timeseries_data['zero mean voltage']))
    stddev_ratio = max_amplitude / standard_deviation

    # Append values to the zero-mean plot textbox.
    text_string += ('\n' + '\n'.join([r'$V_{max}=%.2f$' % max_amplitude,
                                      r'$\frac{V_{max}}{\sigma}=%.2f$' %stddev_ratio]))
    text_obj.set_text(text_string)

    # Change title of plot to reflect question number
    zm_timeseries_plot.set(title=f'Assignment 1, Question 4\nZero-mean timeseries of {data_name}')

    # Save to plots directory.
    zm_timeseries_plot.figure.savefig(f'{PLOT_DIR}{os.sep}question_4_{data_name}.png')

    """
    Question 5
    """
    # Create a sub-directory to house all the histogram data
    hist_plot_dir = f'{PLOT_DIR}{os.sep}histograms'
    if not os.path.exists(hist_plot_dir):
        os.makedirs(hist_plot_dir)

    # common axis labels
    x_label = 'zero-mean voltage(V)'
    y_label = 'probability density function'

    # Generate histogram for various bin width models:
    for bin_model in BIN_MODELS:
        bin_title = BIN_MODELS[bin_model]
        # Create a unique title
        title = f'Assignment 1, Question 5\nSample distribution of zero-mean {data_name} data\n{bin_title}'
        # Generate histogram plot
        fig, histogram_plot = plot_histogram(timeseries_data['zero mean voltage'],
                                        x_label=x_label,
                                        y_label=y_label,
                                        title=title,
                                        bin_model=bin_model)

        # Save plot
        histogram_plot.figure.savefig(f'{hist_plot_dir}{os.sep}question_5_{data_name}_{bin_model}.png')

    """
    Question 6
    """
    # Generate normal probability plot
    fig, ax = normal_probability_plot(timeseries_data['zero mean voltage'])

    # Update x-axis label and title.
    plt.xlabel('zero-mean sample values (V)')
    plt.title(f'Assignment 1, Question 6\nNormal probability plot for {data_name}')

    # Save plot.
    fig.savefig(f'{PLOT_DIR}{os.sep}question_6_probplot_{data_name}.png')

    """
    Question 7
    """
    # Resample (upsample) signal to 44100 Hz
    sample_rate = 44100
    duration = timeseries_data['time'].iloc[-1] - timeseries_data['time'].iloc[0]
    resampled_data = resample(timeseries_data['zero mean voltage'], int(np.floor(duration * sample_rate)))

    # Normalize to maximum amplitude of 1
    normalized_data = resampled_data / np.max(resampled_data)

    # Play sound in speakers.
    sd.play(normalized_data, sample_rate)
    time.sleep(duration)
    sd.stop()

    if not headless:
        plt.show()


def plot_timeseries(x: np.array,
                    y: np.array,
                    title: str = '',
                    x_label: str = '',
                    y_label: str = '',
                    y_lim: tuple = None,
                    x_lim: tuple = None):
    """
    Args:
        x:
        y:
        title:
        x_label:
        y_label:
        y_lim: If provided, plots will scale the y-axis to the bounds specified in a (bottom, upper) tuple.
        x_lim: If provided, plots will scale the x-axis to the bounds specified in a (bottom, upper) tuple.

    Returns:

    """
    fig = plt.figure()
    line_plot = sns.lineplot(x, y)
    line_plot.set(xlabel=x_label, ylabel=y_label, title=title)
    set_minor_gridlines(line_plot)

    if y_lim is not None:
        plt.ylim(y_lim)

    if x_lim is not None:
        plt.xlim(x_lim)

    return fig, line_plot


def plot_histogram(data: np.array,
                   x_label: str = '',
                   y_label: str = '',
                   title: str = '',
                   bin_model: str = 'sqrt'):

    fig = plt.figure()
    if bin_model is None:
        bin_model = 'sqrt'

    histogram_plot = sns.histplot(data=data, stat='density', bins=bin_model)
    histogram_plot.set(xlabel=x_label, ylabel=y_label, title=title)
    set_minor_gridlines(histogram_plot)
    return fig, histogram_plot


def normal_probability_plot(x):
    (quantiles, values), (slope, intercept, r) = probplot(x, dist='norm')
    fig, ax = plt.subplots()
    ax.plot(values, quantiles, '+b')
    ax.plot(quantiles * slope + intercept, quantiles, 'r')
    percentiles = np.array([0.01, 0.02, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.98, 0.99])
    ticks_quan = [norm.ppf(p) for p in percentiles]
    plt.yticks(ticks_quan, percentiles)
    plt.ylabel('probability')
    plt.xlabel('data')
    plt.ylim(norm.ppf(0.01), norm.ppf(0.99))
    plt.xlim(-3, 3)
    set_minor_gridlines(ax)
    return fig, ax


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