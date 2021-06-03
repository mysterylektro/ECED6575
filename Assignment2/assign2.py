"""
    -*- coding: utf-8 -*-
    Time    : 2021-06-01 5:27 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: assign2.py

    {Description}
    -------------

"""

from signal_analysis_tools.wavefiles import WaveFileReader
from signal_analysis_tools.spectrogram import timeseries_to_spectrum, set_minor_gridlines, timeseries_to_spectrogram
from signal_analysis_tools.utilities import setup_logging
import logging
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
import os

# Setup output directories.
PLOT_DIR = os.sep.join([os.path.dirname(__file__), 'plots'])
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

TEXTBOX_PROPS = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.75}

# Setup plotting styles.
sns.plotting_context("paper")
sns.set_style("darkgrid")

logger = setup_logging('2')

# Set random seed for reproducibility.
random.seed(100)

text_strings = {'max_freq': r'$f_{max}\ =\ $%.2f Hz',
                'psd_vrms': r'$f_{maxV_{rms}}\ =\ $%.2f $\frac{V}{\sqrt{Hz}}$',
                'psd_mean_noise': r'$\mu_{noise}\ = \ $%.2f $\frac{V}{\sqrt{Hz}}$'}


def problem_1():
    logger.log(logging.INFO, "Starting problem 1...\n")

    # ----- part a ----- #

    # Read in the full 200 * 1024 records of the wave file into a timeseries object.
    logger.log(logging.DEBUG, "Reading in timeseries file...")
    filename = 'S_plus_N_20.wav'
    reader = WaveFileReader(filename)
    timeseries = reader.next_samples(200 * 1024)

    # ----- part b ----- #

    # Construct a spectrogram of 1024 samples each.
    logger.log(logging.DEBUG, "Constructing 1024 pt spectrogram...")
    spectrogram = timeseries_to_spectrogram(timeseries,
                                            fft_size=1024,
                                            n_time_samples=1024)

    # Retrieve the single sided power spectrum and positive frequency axis for each record
    logger.log(logging.DEBUG, "Computing frequency axis and Gxx...")
    f = spectrogram.positive_frequency_axis()
    gxx = spectrogram.gxx() * spectrogram.f_res

    # Compute the mean Gxx over all 200 records
    logger.log(logging.DEBUG, "Computing mean Gxx...")
    mean_gxx = np.mean(gxx, axis=1)

    # Compute RMS values:
    rms_mean_gxx = np.sqrt(mean_gxx)

    # Compute the maximum value (RMS) and the corresponding frequency bin.
    logger.log(logging.DEBUG, "Computing maximum frequency and corresponding Vrms...")
    max_idx = np.argmax(rms_mean_gxx)
    max_freq = f[max_idx]
    max_rms = rms_mean_gxx[max_idx]

    # Compute the mean noise
    noise_columns = list(range(len(rms_mean_gxx)))
    noise_columns.remove(max_idx)
    mean_noise = np.mean(rms_mean_gxx[noise_columns])

    logger.log(logging.INFO, f"Mean Gxx max Vrms: {max_rms} V/sqrt(Hz)\n"
                             f"Mean Gxx frequency: {max_freq} Hz\n"
                             f"Mean Gxx noise: {mean_noise} V/sqrt(Hz)")

    # Generate a plot of the RMS values.
    logger.log(logging.DEBUG, "Generating plot...")
    fig = plt.figure()
    psd_plot = sns.lineplot(x=f, y=rms_mean_gxx)
    psd_plot.set(xlabel='frequency (Hz)',
                 ylabel=r'$\sqrt{PSD}\ \frac{V}{\sqrt{Hz}}$',
                 title='Assignment 2, Problem 1b\nMean $G_{xx}$, N = 200, FFT = 1024')
    set_minor_gridlines(psd_plot)

    # Overlay the textbox
    text_string = '\n'.join([text_strings.get('max_freq') % max_freq,
                             text_strings.get('psd_vrms') % max_rms,
                             text_strings.get('psd_mean_noise') % mean_noise])

    psd_plot.text(0.95, 0.95, text_string,
                  transform=psd_plot.transAxes,
                  fontsize=14,
                  verticalalignment='top',
                  horizontalalignment='right',
                  bbox=TEXTBOX_PROPS)

    # Save the figure
    logger.log(logging.DEBUG, "Saving plot...")
    fig.savefig(PLOT_DIR + os.sep + 'problem_1b_mean_gxx.png')

    # ----- part c ----- #

    idx = random.choice(range(200))
    logger.log(logging.DEBUG, f"Generating plot for individual Gxx (index: {idx})...")
    fig = plt.figure()
    psd_plot = sns.lineplot(x=f, y=np.sqrt(gxx[:, idx]))
    psd_plot.set(xlabel='frequency (Hz)',
                 ylabel=r'$\sqrt{PSD}\ \frac{V}{\sqrt{Hz}}$',
                 title='Assignment 2, Problem 1c\nSingle $G_{xx}$[%d], FFT = 1024' % idx)
    set_minor_gridlines(psd_plot)

    logger.log(logging.DEBUG, "Computing individual Gxx Vrms...")
    gxx_vrms = np.sqrt(gxx[max_idx, idx])
    logger.log(logging.INFO, f"Gxx[{idx}] {max_freq} Hz Vrms: {gxx_vrms} V/sqrt(Hz)\n")

    logger.log(logging.DEBUG, "Computing individual Gxx mean noise...")
    noise_columns = list(range(gxx.shape[0]))
    noise_columns.remove(max_idx)
    mean_noise = np.mean(np.sqrt(gxx[noise_columns, idx]))

    logger.log(logging.INFO, f"Gxx[{idx}] mean Vrms noise: {mean_noise} V/sqrt(Hz)\n")

    # Overlay the textbox
    text_string = '\n'.join([text_strings.get('max_freq') % max_freq,
                             text_strings.get('psd_vrms') % gxx_vrms,
                             text_strings.get('psd_mean_noise') % mean_noise])

    psd_plot.text(0.95, 0.95, text_string,
                  transform=psd_plot.transAxes,
                  fontsize=14,
                  verticalalignment='top',
                  horizontalalignment='right',
                  bbox=TEXTBOX_PROPS)

    # Save the figure
    logger.log(logging.DEBUG, "Saving plot...")
    fig.savefig(PLOT_DIR + os.sep + 'problem_1c_individual_gxx.png')

    # ----- part d ----- #
    logger.log(logging.DEBUG, "Computing Gxx for full timeseries...")
    spectrum = timeseries_to_spectrum(timeseries)
    f = spectrum.positive_frequencies()
    rms_gxx = np.sqrt(spectrum.single_sided_power_spectral_density() * spectrum.bin_size)

    # Compute the maximum value (RMS) and the corresponding frequency bin.
    logger.log(logging.DEBUG, "Computing maximum frequency and corresponding Vrms...")
    max_idx = np.argmax(rms_gxx)
    max_freq = f[max_idx]
    max_rms = rms_gxx[max_idx]

    # Compute the mean noise
    noise_columns = list(range(len(rms_gxx)))
    noise_columns.remove(max_idx)
    mean_noise = np.mean(rms_gxx[noise_columns])

    logger.log(logging.INFO, f"Full Gxx max Vrms: {max_rms} V/sqrt(Hz)\n"
                             f"Full Gxx frequency: {max_freq} Hz\n"
                             f"Full Gxx noise: {mean_noise} V/sqrt(Hz)")

    logger.log(logging.DEBUG, "Generating plot...")
    fig = plt.figure()
    psd_plot = sns.lineplot(x=f, y=rms_gxx)
    psd_plot.set(xlabel='frequency (Hz)',
                 ylabel=r'$\sqrt{PSD}\ \frac{V}{\sqrt{Hz}}$',
                 title='Assignment 2, Problem 1d\nFull timeseries $G_{xx}$, FFT = 204800')

    # Overlay the textbox
    text_string = '\n'.join([text_strings.get('max_freq') % max_freq,
                             text_strings.get('psd_vrms') % max_rms,
                             text_strings.get('psd_mean_noise') % mean_noise])

    psd_plot.text(0.95, 0.95, text_string,
                  transform=psd_plot.transAxes,
                  fontsize=14,
                  verticalalignment='top',
                  horizontalalignment='right',
                  bbox=TEXTBOX_PROPS)

    # Save the figure
    logger.log(logging.DEBUG, "Saving plot...")
    fig.savefig(PLOT_DIR + os.sep + 'problem_1d_full_timeseries_gxx.png')


if __name__ == '__main__':
    problem_1()

