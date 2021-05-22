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
import logging
import pandas as pd
from signal_analysis_tools.utilities import setup_logging
from signal_analysis_tools.timeseries import *

# Setup output directories.
PLOT_DIR = os.sep.join([os.path.dirname(__file__), 'plots'])
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

HIST_PLOT_DIR = f'{PLOT_DIR}{os.sep}histograms'
if not os.path.exists(HIST_PLOT_DIR):
    os.makedirs(HIST_PLOT_DIR)


def assign1(data_file: str, log_level: int = None):
    """
    This function will complete all problems, generating plots to the 'plots/'
    subdirectory, outputting diagnostic information to the screen as well as
    computed information to the output log file.

    Args:
        data_file: string indicating the name of the originating data file.
        log_level: integer value indicating the logger output log level.

    Returns: None
    """

    logger = setup_logging()

    # Set random seed for reproducibility
    np.random.seed(100)

    if log_level is not None:
        logger.setLevel(log_level)
    logger.log(logging.INFO, f"-----Starting analysis for {data_file}-----\n")

    # Parse data basename for use later on.
    data_name = os.path.splitext(os.path.basename(data_file))[0]

    # Read in the data
    logger.log(logging.DEBUG, f"Reading input .CSV file...")
    header_list = ['time', 'voltage']
    data = pd.read_csv(data_file, usecols=[0, 1], names=header_list)

    sample_rate = 1. / (data['time'].iloc[1] - data['time'].iloc[0])
    timeseries = Timeseries(data['time'], data['voltage'], sample_rate)

    # Create a timeseries analyzer object
    timeseries_analyzer = TimeseriesAnalyzer(timeseries)

    # Generate a figure and plot of the timeseries data.
    logger.log(logging.DEBUG, f"Plotting timeseries data...")
    fig, p = timeseries_analyzer.plot_time_domain(title=f'Assignment 1, Problem 1\nTimeseries of {data_name}',
                                                  x_label='time (s)',
                                                  y_label='voltage (V)',
                                                  y_lim=(-3, 3),
                                                  x_lim=(0, 0.5),
                                                  filename=f'{PLOT_DIR}{os.sep}problem_1_{data_name}.png')
    plt.close(fig)

    logger.log(logging.DEBUG, f"Problem 1 complete!")

    """
    Problem 2
    """
    logger.log(logging.DEBUG, f"Starting problem 2...")

    # Calculate the sample rate, number of samples, and average (mean) voltage value.
    logger.log(logging.DEBUG, f"Calculating number of samples, sample rate, and mean voltage...")

    # Log computed values to file
    logger.log(logging.INFO, '\n'.join([f"{data_name}",
                                        f"Number of samples: {timeseries.num_samples()}",
                                        f"Sample rate (Hz): {timeseries.sample_rate}",
                                        f"Mean voltage (V): {timeseries.mean()}"]) + '\n')

    fig, p = timeseries_analyzer.plot_time_domain(title=f'Assignment 1, Problem 2\nTimeseries of {data_name}',
                                                  x_label='time (s)',
                                                  y_label='voltage (V)',
                                                  y_lim=(-3, 3),
                                                  x_lim=(0, 0.5),
                                                  filename=f'{PLOT_DIR}{os.sep}problem_2_{data_name}.png',
                                                  stats=['num_samples', 'sample_rate', 'mean'])

    plt.close(fig)
    logger.log(logging.DEBUG, f"Problem 2 complete!")

    """
    Problem 3
    """
    logger.log(logging.DEBUG, f"Starting problem 3...")

    # Generate zero-mean time series and calculate variance and standard deviation.
    logger.log(logging.DEBUG, f"Generating zero-mean timeseries...")
    zero_mean_timeseries = Timeseries(timeseries.time(),
                                      timeseries.amplitude() - timeseries.mean(),
                                      timeseries.sample_rate)

    # Log computed values to file
    logger.log(logging.INFO, '\n'.join([f"Zero-mean {data_name}",
                                        f"Standard deviation (V): {zero_mean_timeseries.std()}",
                                        f"Variance (V^2): {zero_mean_timeseries.var()}"]) + '\n')

    timeseries_analyzer.set_timeseries(zero_mean_timeseries)

    fig, p = timeseries_analyzer.plot_time_domain(title=f'Assignment 1, Problem 3\nZero-mean '
                                                        f'timeseries of {data_name}',
                                                  x_label='time (s)',
                                                  y_label='voltage (V)',
                                                  y_lim=(-3, 3),
                                                  x_lim=(0, 0.5),
                                                  filename=f'{PLOT_DIR}{os.sep}problem_3_{data_name}.png',
                                                  stats=['std', 'var'])
    plt.close(fig)
    logger.log(logging.DEBUG, f'Problem 3 complete!')

    """
    Problem 4
    """
    logger.log(logging.DEBUG, f'Starting problem 4...')

    # Calculate max amplitude and standard deviation ratio
    logger.log(logging.DEBUG, f'Calculating maximum amplitude and standard deviation ratio...')

    # Log computed values to file
    logger.log(logging.INFO, '\n'.join([f"Zero-mean {data_name}",
                                        f"Maximum amplitude (V): {zero_mean_timeseries.max()}",
                                        f"Max amplitude : Standard deviation ratio: "
                                        f"{zero_mean_timeseries.stddev_ratio()}"]) + '\n')

    fig, p = timeseries_analyzer.plot_time_domain(title=f'Assignment 1, Problem 4\nZero-mean '
                                                        f'timeseries of {data_name}',
                                                  x_label='time (s)',
                                                  y_label='voltage (V)',
                                                  y_lim=(-3, 3),
                                                  x_lim=(0, 0.5),
                                                  filename=f'{PLOT_DIR}{os.sep}problem_4_{data_name}.png',
                                                  stats=['std', 'var', 'max', 'std_ratio'])
    plt.close(fig)
    logger.log(logging.DEBUG, f'Problem 4 complete!')

    """
    Problem 5
    """
    logger.log(logging.DEBUG, f'Starting problem 5...')

    # Common axis label
    x_label = 'zero-mean voltage(V)'

    # Generate histogram for various bin width models:
    logger.log(logging.DEBUG, f'Generating histogram plots...')
    for bin_model in timeseries_analyzer.BIN_MODELS:
        bin_title = timeseries_analyzer.BIN_MODELS[bin_model]
        logger.log(logging.DEBUG, f'Generating {bin_title} histogram plot...')

        # Create a unique title based on the data file name and the bin model
        title = f'Assignment 1, Problem 5\nSample distribution of zero-mean {data_name} data\n{bin_title}'
        fig, p = timeseries_analyzer.plot_histogram(bin_model=bin_model,
                                                    x_label=x_label,
                                                    y_label='probability density function',
                                                    title=title,
                                                    filename=f'{HIST_PLOT_DIR}{os.sep}problem_5_'
                                                             f'{data_name}_{bin_model}.png')
        plt.close(fig)
    logger.log(logging.DEBUG, f'Histogram plots complete!')
    logger.log(logging.DEBUG, f'Problem 5 complete!')

    """
    Problem 6
    """
    logger.log(logging.DEBUG, f'Starting problem 6...')

    # Generate normal probability plot
    logger.log(logging.DEBUG, f'Generating normal probability plot...')
    timeseries_analyzer.plot_normal_probability(x_label='zero-mean sample values (V)',
                                                title=f'Assignment 1, Problem 6\nNormal probability '
                                                      f'plot for {data_name}',
                                                filename=f'{PLOT_DIR}{os.sep}problem_6_probplot_'
                                                         f'{data_name}.png')

    """
    Problem 7
    """
    logger.log(logging.DEBUG, f'Starting problem 7...')

    timeseries_analyzer.playback_timeseries(sample_rate=44100)
    logger.log(logging.DEBUG, f'Problem 7 complete!')
