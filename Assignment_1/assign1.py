"""
    -*- coding: utf-8 -*-
    Time    : 2021-05-11 7:53 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: assign1.py

    {Description}
    -------------

    This file contains all of the helper functions to complete assignment 1 for Dalhousie course
    ECED-6575 Underwater Acoustics Engineering.

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


# Setup output directories.
PLOT_DIR = os.sep.join([os.path.dirname(__file__), 'plots'])
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

HIST_PLOT_DIR = f'{PLOT_DIR}{os.sep}histograms'
if not os.path.exists(HIST_PLOT_DIR):
    os.makedirs(HIST_PLOT_DIR)

# Setup common graphics properties.
TEXTBOX_PROPS = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.75}

# Histogram bin-width models.
BIN_MODELS = {'fd': 'Freedman-Diaconis choice',
              'doane': 'Doane\'s formula',
              'scott': 'Scott\'s normal reference rule',
              'stone': 'Stone\'s formula',
              'rice': 'Rice rule',
              'sturges': 'Sturge\'s formula',
              'sqrt': 'Square-root choice'}

# Setup plotting styles.
sns.plotting_context("paper")
sns.set_style("darkgrid")

# Setup logging.
logger = logging.getLogger('assignment1')
logger.setLevel(logging.DEBUG)

output_log_filename = 'assignment1.log'
log_file_handler = logging.FileHandler(output_log_filename, 'w+')
log_file_handler.setLevel(logging.INFO)
logger.addHandler(log_file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)


def assign1(data_file: str, headless: bool = True, log_level: int = None):
    f"""
    This function will complete all problems, generating plots to the 'plots/'
    subdirectory, outputting diagnostic information to the screen as well as
    computed information to the log file {output_log_filename}.

    Args:
        data_file: string indicating the name of the originating data file.
        headless: Boolean value to inhibit displaying plots. Plots will still be
                  saved to the the global PLOT_DIR directory.
        log_level: integer value indicating the logger output log level.

    Returns: None

    """
    if log_level is not None:
        logger.setLevel(log_level)
    logger.log(logging.INFO, f"-----Starting analysis for {data_file}-----\n")

    # Parse data basename for use later on.
    data_name = os.path.splitext(os.path.basename(data_file))[0]

    """
    Problem 1
    """
    logger.log(logging.DEBUG, f"Starting problem 1...")

    # Read in the data
    logger.log(logging.DEBUG, f"Reading input .CSV file...")
    header_list = ['time', 'voltage']
    timeseries_data = pd.read_csv(data_file, usecols=[0, 1], names=header_list)

    # Generate a figure and plot of the timeseries data.
    logger.log(logging.DEBUG, f"Plotting timeseries data...")
    timeseries_figure, timeseries_plot = plot_timeseries(timeseries_data['time'],
                                                         timeseries_data['voltage'],
                                                         title=f'Assignment 1, Problem 1\nTimeseries of {data_name}',
                                                         x_label='time (s)',
                                                         y_label='voltage (V)',
                                                         y_lim=(-3, 3),
                                                         x_lim=(0, 0.5))

    # Save the plot to the plots directory
    logger.log(logging.DEBUG, f"Saving plot...")
    timeseries_plot.figure.savefig(f'{PLOT_DIR}{os.sep}problem_1_{data_name}.png')

    logger.log(logging.DEBUG, f"Problem 1 complete!")

    """
    Problem 2
    """
    logger.log(logging.DEBUG, f"Starting problem 2...")

    # Calculate the sample rate, number of samples, and average (mean) voltage value.
    logger.log(logging.DEBUG, f"Calculating number of samples, sample rate, and mean voltage...")
    num_samples = len(timeseries_data)
    sample_rate = (num_samples - 1) / (timeseries_data['time'].iloc[-1] - timeseries_data['time'].iloc[0])
    mean_voltage = np.mean(timeseries_data['voltage'])

    # Log computed values to file
    logger.log(logging.INFO, '\n'.join([f"{data_name}",
                                        f"Number of samples: {num_samples}",
                                        f"Sample rate (Hz): {sample_rate}",
                                        f"Mean voltage (V): {mean_voltage}"]) + '\n')

    logger.log(logging.DEBUG, f"Updating plot...")

    # Add newly computed values to a textbox in the timeseries plot generated in problem 1
    text_string = '\n'.join([r'$f_{s}=%.2f\ Hz$' % sample_rate,
                             r'$N=%d$' % num_samples,
                             r'$\mu=%.2f\ V$' % mean_voltage])
    timeseries_plot.text(0.05, 0.95, text_string,
                         transform=timeseries_plot.transAxes,
                         fontsize=14, verticalalignment='top', bbox=TEXTBOX_PROPS)

    # Change title of plot to reflect problem number
    timeseries_plot.set(title=f'Assignment 1, Problem 2\nTimeseries of {data_name}')

    # Save to plots directory.
    logger.log(logging.DEBUG, f"Saving plot...")
    timeseries_plot.figure.savefig(f'{PLOT_DIR}{os.sep}problem_2_{data_name}.png')

    logger.log(logging.DEBUG, f"Problem 2 complete!")

    """
    Problem 3
    """
    logger.log(logging.DEBUG, f"Starting problem 3...")

    # Generate zero-mean time series and calculate variance and standard deviation.
    logger.log(logging.DEBUG, f"Generating zero-mean timeseries...")
    timeseries_data['zero mean voltage'] = timeseries_data['voltage'] - mean_voltage

    logger.log(logging.DEBUG, f"Calculating standard deviation and variance...")
    standard_deviation = np.std(timeseries_data['zero mean voltage'])
    variance = standard_deviation ** 2

    # Log computed values to file
    logger.log(logging.INFO, '\n'.join([f"Zero-mean {data_name}",
                                        f"Standard deviation (V): {standard_deviation}",
                                        f"Variance (V): {variance}"]) + '\n')

    # Generate timeseries plot.
    logger.log(logging.DEBUG, f'Generating timeseries plot...')
    zm_timeseries_figure, zm_timeseries_plot = plot_timeseries(timeseries_data['time'],
                                                               timeseries_data['zero mean voltage'],
                                                               title=f'Assignment 1, Problem 3\nZero-mean '
                                                                     f'timeseries of {data_name}',
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
    logger.log(logging.DEBUG, f'Saving plot...')
    zm_timeseries_plot.figure.savefig(f'{PLOT_DIR}{os.sep}problem_3_{data_name}.png')

    logger.log(logging.DEBUG, f'Problem 3 complete!')

    """
    Problem 4
    """
    logger.log(logging.DEBUG, f'Starting problem 4...')

    # Calculate max amplitude and standard deviation ratio
    logger.log(logging.DEBUG, f'Calculating maximum amplitude and standard deviation ratio...')
    max_amplitude = np.max(np.abs(timeseries_data['zero mean voltage']))
    stddev_ratio = max_amplitude / standard_deviation

    # Log computed values to file
    logger.log(logging.INFO, '\n'.join([f"Zero-mean {data_name}",
                                        f"Maximum amplitude (V): {max_amplitude}",
                                        f"Max amplitude : Standard deviation ratio: {stddev_ratio}"]) + '\n')

    # Append values to the zero-mean plot textbox.
    logger.log(logging.DEBUG, f'Updating plot...')
    text_string += ('\n' + '\n'.join([r'$V_{max}=%.2f\ V$' % max_amplitude,
                                      r'$\frac{V_{max}}{\sigma}=%.2f$' % stddev_ratio]))
    text_obj.set_text(text_string)

    # Change title of plot to reflect problem number
    zm_timeseries_plot.set(title=f'Assignment 1, Problem 4\nZero-mean timeseries of {data_name}')

    # Save to plots directory.
    logger.log(logging.DEBUG, f'Saving plot...')
    zm_timeseries_plot.figure.savefig(f'{PLOT_DIR}{os.sep}problem_4_{data_name}.png')

    logger.log(logging.DEBUG, f'Problem 4 complete!')

    """
    Problem 5
    """
    logger.log(logging.DEBUG, f'Staring problem 5...')

    # Common axis labels
    x_label = 'zero-mean voltage(V)'
    y_label = 'probability density function'

    # Generate histogram for various bin width models:
    logger.log(logging.DEBUG, f'Generating histogram plots...')
    for bin_model in BIN_MODELS:
        bin_title = BIN_MODELS[bin_model]
        logger.log(logging.DEBUG, f'Generating {bin_title} histogram plot...')

        # Create a unique title based on the data file name and the bin model
        title = f'Assignment 1, Problem 5\nSample distribution of zero-mean {data_name} data\n{bin_title}'

        # Generate histogram plot
        fig, histogram_plot = plot_histogram(timeseries_data['zero mean voltage'],
                                             x_label=x_label,
                                             y_label=y_label,
                                             title=title,
                                             bin_model=bin_model)

        # Save plot
        logger.log(logging.DEBUG, f'Saving {bin_title} histogram plot...')
        histogram_plot.figure.savefig(f'{HIST_PLOT_DIR}{os.sep}problem_5_{data_name}_{bin_model}.png')
    logger.log(logging.DEBUG, f'Histogram plots complete!')

    logger.log(logging.DEBUG, f'Problem 5 complete!')

    """
    Problem 6
    """
    logger.log(logging.DEBUG, f'Starting problem 6...')

    # Generate normal probability plot
    logger.log(logging.DEBUG, f'Generating normal probability plot...')
    fig, ax, (slope, intercept, r) = normal_probability_plot(timeseries_data['zero mean voltage'])
    text_string = '\n'.join([r'$m=%.2f\ V$' % slope,
                             r'$b=%.2f\ V$' % intercept,
                             r'$r=%.2f$' % r])
    ax.text(0.05, 0.95, text_string,
            transform=timeseries_plot.transAxes,
            fontsize=14, verticalalignment='top', bbox=TEXTBOX_PROPS)

    # Update x-axis label and title.
    plt.xlabel('zero-mean sample values (V)')
    plt.title(f'Assignment 1, Problem 6\nNormal probability plot for {data_name}')

    # Save plot.
    logger.log(logging.DEBUG, f'Saving normal probability plot...')
    fig.savefig(f'{PLOT_DIR}{os.sep}problem_6_probplot_{data_name}.png')

    logger.log(logging.DEBUG, f'Problem 6 complete!')

    """
    Problem 7
    """
    logger.log(logging.DEBUG, f'Starting problem 7...')

    # Resample (upsample) signal to 44100 Hz
    logger.log(logging.DEBUG, f'Resampling to 44100 Hz...')
    sample_rate = 44100
    duration = timeseries_data['time'].iloc[-1] - timeseries_data['time'].iloc[0]
    resampled_data = resample(timeseries_data['zero mean voltage'], int(np.floor(duration * sample_rate)))

    # Normalize to maximum amplitude of 1
    logger.log(logging.DEBUG, f'Normalizing to 1V max...')
    normalized_data = resampled_data / np.max(resampled_data)

    # Play sound in speakers.
    logger.log(logging.DEBUG, f'Playing sound...')
    sd.play(normalized_data, sample_rate)
    time.sleep(duration)
    sd.stop()

    logger.log(logging.DEBUG, f'Problem 7 complete!')

    if not headless:
        logger.log(logging.DEBUG, f'Showing all plots...')
        plt.show()


def plot_timeseries(x: np.array,
                    y: np.array,
                    title: str = '',
                    x_label: str = '',
                    y_label: str = '',
                    y_lim: tuple = None,
                    x_lim: tuple = None):
    """
    This is a convenience function that creates a new figure and uses the Seaborn lineplot
    to generate a timeseries plot.

    Args:
        x: x values
        y: y values
        title: string indicating the title of the plot. Defaults to an empty string.
        x_label: string indicating the x-axis label. Defaults to an empty string.
        y_label: string indicating the y-axis label. Defaults to an empty string.
        y_lim: If provided, plots will scale the y-axis to the bounds specified in a (bottom, upper) tuple.
        x_lim: If provided, plots will scale the x-axis to the bounds specified in a (bottom, upper) tuple.

    Returns: (fig, ax) A matplotlib figure reference, and the axis reference to the lineplot.

    """
    fig = plt.figure()
    line_plot = sns.lineplot(x=x, y=y)
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
    """
    This is a convenience wrapper to create a new figure and plot a histogram using Seaborn's
    histplot function.

    Args:
        data: 1-dimensional data array to compute histogram for.
        x_label: string label for the x-axis. Defaults to an empty string.
        y_label: string label for the y-axis. Defaults to an empty string.
        title: string label to title the plot. Defaults to an empty string.
        bin_model: string indicating the type of bin-width model to use.

    Returns: (fig, ax) A matplotlib figure reference, and the axis reference to the histplot.

    """
    fig = plt.figure()
    if bin_model is None:
        bin_model = 'sqrt'

    histogram_plot = sns.histplot(data=data, stat='density', bins=bin_model)
    histogram_plot.set(xlabel=x_label, ylabel=y_label, title=title)
    set_minor_gridlines(histogram_plot)
    return fig, histogram_plot


def normal_probability_plot(x):
    """
    This is a convenience function that wraps SciPy's probplot function.
    This will reorder the y-axis tick values to use a normal distribution percentiles
    instead of the probplot default data quantiles.

    Args:
        x: 1-dimensional data to compute the normal probability plot for.

    Returns: fig, ax, (slope, intercept, r)
        A matplotlib figure reference, and the axis reference to the normal probability plot.
        The best-fit line slope, intercept, and coefficient of determination values are returned as
        a tuple.

    """
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
    return fig, ax, (slope, intercept, r)


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
