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
from signal_analysis_tools.spectrogram import *
from signal_analysis_tools.utilities import setup_logging
from signal_analysis_tools.wave_models import PlaneWave, SphericalWave, PressureUnits
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

text_strings = {'peak_freq': r'$\mathcal{f}_{peak}\ =\ $%.2f Hz',
                'peak_vrms': r'$V_{rms}\ =\ $%.2f V',
                'mean_root_noise': r'$\mu_{n}\ = \ $%.2f $\frac{V}{\sqrt{Hz}}$'}

# Standard constants
RHO = 1000.  # kg/m^3 - Fresh water, 20 deg C
C = 1481.  # m/s - Fresh water, 20 deg C


def problem_1():
    def generate_plot(freq, gxx, peak_f, peak_f_vrms, mean_noise, title, output_filename):
        # Generate a plot of the RMS values.
        logger.log(logging.DEBUG, "Generating plot...")
        fig = plt.figure()
        psd_plot = sns.lineplot(x=freq, y=np.sqrt(gxx))
        psd_plot.set(xlabel='frequency (Hz)',
                     ylabel=r'$\sqrt{PSD}\ \frac{V}{\sqrt{Hz}}$',
                     title=title)
        set_minor_gridlines(psd_plot)

        # Overlay the textbox
        text_string = '\n'.join([text_strings.get('peak_freq') % peak_f,
                                 text_strings.get('peak_vrms') % peak_f_vrms,
                                 text_strings.get('mean_root_noise') % mean_noise])

        psd_plot.text(0.95, 0.95, text_string,
                      transform=psd_plot.transAxes,
                      fontsize=14,
                      verticalalignment='top',
                      horizontalalignment='right',
                      bbox=TEXTBOX_PROPS)

        # Save the figure
        logger.log(logging.DEBUG, "Saving plot...")
        fig.savefig(output_filename)

        plt.close(fig)

    logger.log(logging.INFO, f"\n{'-'*10} PROBLEM 1 {'-'*10}\n")

    # ----- part a ----- #
    logger.log(logging.DEBUG, "\nStarting problem 1a...\n")

    # Read in the full 200 * 1024 records of the wave file into a timeseries object.
    logger.log(logging.DEBUG, "Reading in timeseries file...")
    filename = 'S_plus_N_20.wav'
    reader = WaveFileReader(filename)
    timeseries = reader.next_samples(200 * 1024)

    # ----- part b ----- #
    logger.log(logging.DEBUG, "\nStarting problem 1b...\n")

    # Construct a spectrogram of 1024 samples each.
    logger.log(logging.DEBUG, "Constructing 1024 pt spectrogram...")
    spectrogram = timeseries_to_spectrogram(timeseries,
                                            fft_size=1024,
                                            n_samples=1024)

    # Retrieve the single sided power spectrum and positive frequency axis for each record
    logger.log(logging.DEBUG, "Computing frequency axis and Gxx...")
    f = spectrogram.positive_frequency_axis()
    spectrogram_gxx = spectrogram.gxx()

    # Compute the mean Gxx over all 200 records
    logger.log(logging.DEBUG, "Computing mean Gxx...")
    mean_gxx = np.mean(spectrogram_gxx, axis=1)

    # Compute the maximum value (RMS) and the corresponding frequency bin.
    logger.log(logging.DEBUG, "Computing maximum frequency and corresponding Vrms...")
    max_idx = np.argmax(mean_gxx)
    max_freq = f[max_idx]
    max_gxx = mean_gxx[max_idx]
    max_rms = np.sqrt(max_gxx * spectrogram.f_res)

    # Compute the mean noise
    noise_columns = list(range(len(mean_gxx)))
    noise_columns.remove(max_idx)
    mean_root_noise = np.sqrt(np.mean(mean_gxx[noise_columns]))

    logger.log(logging.INFO, f"\nMean Gxx - peak frequency: {max_freq} Hz\n"
                             f"Mean Gxx - peak frequency Vrms: {max_rms} V\n"
                             f"Mean root Gxx noise: {mean_root_noise} V/sqrt(Hz)\n")

    plot_title = 'Assignment 2, Problem 1b\nMean $G_{xx}$, N = 200, FFT = 1024'
    filename = PLOT_DIR + os.sep + 'problem_1b_mean_gxx.png'

    generate_plot(f, mean_gxx, max_freq, max_rms, mean_root_noise, plot_title, filename)

    # ----- part c ----- #
    logger.log(logging.DEBUG, "\nStarting problem 1c...\n")

    logger.log(logging.DEBUG, f"Choosing random individual Gxx...")
    idx = random.choice(range(200))

    logger.log(logging.DEBUG, f"Computing individual Gxx Vrms for index {idx}...")
    gxx_vrms = np.sqrt(spectrogram_gxx[max_idx, idx] * spectrogram.f_res)

    logger.log(logging.DEBUG, "Computing individual Gxx mean noise...")
    noise_columns = list(range(spectrogram_gxx.shape[0]))
    noise_columns.remove(max_idx)
    mean_root_noise = np.sqrt(np.mean(spectrogram_gxx[noise_columns, idx]))

    logger.log(logging.INFO, f"\nGxx[{idx}] - peak frequency: {max_freq} Hz\n"
                             f"Gxx[{idx}] - peak frequency Vrms: {gxx_vrms} V\n"
                             f"Gxx[{idx}] - mean noise: {mean_root_noise} V/sqrt(Hz)\n")

    plot_title = 'Assignment 2, Problem 1c\nSingle $G_{xx}$[%d], FFT = 1024' % idx
    filename = PLOT_DIR + os.sep + 'problem_1c_individual_gxx.png'

    generate_plot(f, spectrogram_gxx[:, idx], max_freq, gxx_vrms, mean_root_noise, plot_title, filename)

    # ----- part d ----- #
    logger.log(logging.DEBUG, "\nStarting problem 1d...\n")

    logger.log(logging.DEBUG, "Computing Gxx for full timeseries...")
    spectrum = timeseries_to_spectrum(timeseries)
    f = spectrum.positive_frequencies()
    full_gxx = spectrum.gxx()

    # Compute the maximum value (RMS) and the corresponding frequency bin.
    logger.log(logging.DEBUG, "Computing maximum frequency and corresponding Vrms...")
    max_idx = np.argmax(full_gxx)
    max_freq = f[max_idx]
    max_rms = np.sqrt(full_gxx[max_idx] * spectrum.f_res)

    # Compute the mean noise
    noise_columns = list(range(len(full_gxx)))
    noise_columns.remove(max_idx)
    mean_root_noise = np.sqrt(np.mean(full_gxx[noise_columns]))

    logger.log(logging.INFO, f"\nFull Gxx - peak frequency: {max_freq} Hz\n"
                             f"Full Gxx - peak frequency Vrms: {max_rms} V\n"
                             f"Full Gxx - mean root noise: {mean_root_noise} V/sqrt(Hz)\n")

    plot_title = 'Assignment 2, Problem 1d\nFull timeseries $G_{xx}$, FFT = 204800'
    filename = PLOT_DIR + os.sep + 'problem_1d_full_timeseries_gxx.png'

    generate_plot(f, full_gxx, max_freq, max_rms, mean_root_noise, plot_title, filename)


def problem_2():
    def generate_psd_plot(freq, gxx, output_filename, title='', xlim=None):
        # Generate a plot of the RMS values.
        logger.log(logging.DEBUG, "Generating plot...")
        figure = plt.figure()
        psd_plot = sns.lineplot(x=freq, y=gxx)
        psd_plot.set(xlabel='frequency (Hz)',
                     ylabel=r'$PSD\ \frac{V^2}{Hz}$',
                     title=title)
        set_minor_gridlines(psd_plot)

        if xlim:
            plt.xlim(*xlim)

        # Save the figure
        logger.log(logging.DEBUG, "Saving plot...")
        figure.savefig(output_filename)

        plt.close(figure)

    logger.log(logging.INFO, f"\n{'-'*10} PROBLEM 2 {'-'*10}\n")
    zoom_xlim = (200, 600)

    # ----- part a ----- #
    logger.log(logging.DEBUG, "\nStarting problem 2a...\n")

    # Read in the full wave file into a timeseries object.
    logger.log(logging.DEBUG, "Reading in timeseries file...")
    wav_filename = 'P_plus_N_10.wav'
    reader = WaveFileReader(wav_filename)
    timeseries = reader.next_samples(reader.num_samples)

    logger.log(logging.DEBUG, "Constructing de-synchronized spectrogram...")
    desync_spectrogram = timeseries_to_spectrogram(timeseries,
                                                   fft_size=1024,
                                                   n_samples=1024,
                                                   n_records=230,
                                                   synchronization_offset=0)

    f = desync_spectrogram.positive_frequency_axis()

    logger.log(logging.DEBUG, "Computing mean Gxx of de-synchronized spectrogram...")
    mean_desync_gxx = np.mean(desync_spectrogram.gxx(), axis=1)
    filename = PLOT_DIR + os.sep + 'problem_2a_mean_desync_gxx.png'
    plot_title = 'Assignment 2, Problem 2a\nMean de-synchronized power spectrum'
    generate_psd_plot(f, mean_desync_gxx, filename, title=plot_title)

    logger.log(logging.DEBUG, "Zooming in on peak...")
    filename = PLOT_DIR + os.sep + 'problem_2a_mean_desync_gxx_zoom.png'
    generate_psd_plot(f, mean_desync_gxx, filename, title=plot_title, xlim=zoom_xlim)

    # ----- part b ----- #
    logger.log(logging.DEBUG, "\nStarting problem 2b...\n")

    logger.log(logging.DEBUG, "Constructing synchronized spectrogram...")
    sync_spectrogram = timeseries_to_spectrogram(timeseries,
                                                 fft_size=1024,
                                                 n_samples=1024,
                                                 n_records=230,
                                                 synchronization_offset=(1111 - 1024))

    logger.log(logging.DEBUG, "Computing mean Gxx of de-synchronized spectrogram...")
    mean_sync_gxx = np.mean(sync_spectrogram.gxx(), axis=1)
    f = sync_spectrogram.positive_frequency_axis()
    filename = PLOT_DIR + os.sep + 'problem_2b_mean_sync_gxx.png'
    plot_title = 'Assignment 2, Problem 2b\nMean synchronized power spectrum'
    generate_psd_plot(f, mean_sync_gxx, filename, title=plot_title)

    logger.log(logging.DEBUG, "Zooming in on peak...")
    filename = PLOT_DIR + os.sep + 'problem_2b_mean_sync_gxx_zoom.png'
    generate_psd_plot(f, mean_sync_gxx, filename, title=plot_title, xlim=zoom_xlim)

    # ---- part c ---- #
    logger.log(logging.DEBUG, "\nStarting problem 2c...\n")

    # --- Calculate synchronized spectrum
    logger.log(logging.DEBUG, "Computing mean linear spectrum from synchronized spectrogram...")
    mean_linear_sync_spectrum = Spectrum(np.mean(sync_spectrogram.data, axis=1), f_res=sync_spectrogram.f_res)

    logger.log(logging.DEBUG, "Computing Gxx of mean linear spectrum from synchronized spectrogram...")
    mean_linear_sync_spectrum_gxx = mean_linear_sync_spectrum.gxx()

    f = mean_linear_sync_spectrum.positive_frequencies()
    plot_title = 'Assignment 2, Problem 2c\nPower spectrum of mean synchronized linear spectrum'
    generate_psd_plot(f,
                      mean_linear_sync_spectrum_gxx,
                      PLOT_DIR + os.sep + 'problem_2c_mean_linear_sync_spectrum_gxx.png',
                      title=plot_title)

    logger.log(logging.DEBUG, "Zooming in on peak...")
    generate_psd_plot(f,
                      mean_linear_sync_spectrum_gxx,
                      PLOT_DIR + os.sep + 'problem_2c_mean_linear_sync_spectrum_gxx_zoom.png',
                      title=plot_title,
                      xlim=zoom_xlim)

    # --- Calculate de-synchronized spectrum
    logger.log(logging.DEBUG, "Computing mean linear spectrum from de-synchronized spectrogram...")
    mean_linear_desync_spectrum = Spectrum(np.mean(desync_spectrogram.data, axis=1), f_res=desync_spectrogram.f_res)

    logger.log(logging.DEBUG, "Computing Gxx of mean linear spectrum from synchronized spectrogram...")
    mean_linear_desync_spectrum_gxx = mean_linear_desync_spectrum.gxx()

    f = mean_linear_desync_spectrum.positive_frequencies()
    plot_title = 'Assignment 2, Problem 2c\nPower spectrum of mean de-synchronized linear spectrum'
    generate_psd_plot(f,
                      mean_linear_desync_spectrum_gxx,
                      PLOT_DIR + os.sep + 'problem_2c_mean_linear_desync_spectrum_gxx.png',
                      title=plot_title)

    logger.log(logging.DEBUG, "Zooming in on peak...")
    generate_psd_plot(f,
                      mean_linear_desync_spectrum_gxx,
                      PLOT_DIR + os.sep + 'problem_2c_mean_linear_desync_spectrum_gxx_zoom.png',
                      title=plot_title,
                      xlim=zoom_xlim)

    # ----- part d ----- #
    logger.log(logging.DEBUG, "\nStarting problem 2d...\n")

    block_size = 1024
    total_samples = 1111
    n_records = 230

    # Reshape the timeseries data into bins of n_time_samples:
    logger.log(logging.DEBUG, "Generating synchronized time samples...")
    sync_time_data = np.reshape(timeseries.data['amplitude'][:(n_records * total_samples)], (total_samples, n_records),
                                order='F')[:block_size, :]

    logger.log(logging.DEBUG, "Computing mean synchronized timeseries...")
    mean_sync_time_data = Timeseries(np.mean(sync_time_data, axis=1), timeseries.sample_rate)
    plotter = TimeseriesPlotter(timeseries=mean_sync_time_data)
    plot_title = 'Assignment 2, Problem 2d\nMean synchronized timeseries'
    logger.log(logging.DEBUG, "Generating and saving plot...")
    fig, p = plotter.plot_time_domain(filename=PLOT_DIR + os.sep + 'problem_2d_sync_time_average_timeseries.png',
                                      title=plot_title)
    plt.close(fig)

    logger.log(logging.DEBUG, "Generating de-synchronized time samples...")
    desync_time_data = np.reshape(timeseries.data['amplitude'][:(n_records * block_size)], (block_size, n_records),
                                  order='F')

    logger.log(logging.DEBUG, "Computing mean synchronized timeseries...")
    mean_desync_time_data = Timeseries(np.mean(desync_time_data, axis=1), timeseries.sample_rate)
    plotter.set_timeseries(mean_desync_time_data)
    plot_title = 'Assignment 2, Problem 2d\nMean de-synchronized timeseries'
    logger.log(logging.DEBUG, "Generating and saving plot...")
    fig, p = plotter.plot_time_domain(filename=PLOT_DIR + os.sep + 'problem_2d_desync_time_average_timeseries.png',
                                      title=plot_title)
    plt.close(fig)

    # ----- part e ----- #
    logger.log(logging.DEBUG, "\nStarting problem 2e...\n")

    logger.log(logging.DEBUG, "Generating spectrum from mean synchronized timeseries...")
    mean_sync_time_spectrum = timeseries_to_spectrum(mean_sync_time_data)
    spectrum_plotter = SpectrumPlotter(mean_sync_time_spectrum)
    plot_title = 'Assignment 2, Problem 2e\nPower spectrum of mean synchronized timeseries'
    filename = PLOT_DIR + os.sep + 'problem_2e_sync_time_average_spectrum.png'
    logger.log(logging.DEBUG, "Generating and saving plot...")
    fig, p = spectrum_plotter.plot_single_sided_power_spectral_density(filename=filename,
                                                                       title=plot_title)
    plt.close(fig)

    logger.log(logging.DEBUG, "Zooming in on peak...")
    logger.log(logging.DEBUG, "Generating and saving plot...")
    filename = PLOT_DIR + os.sep + 'problem_2e_sync_time_average_spectrum_zoom.png'
    fig, p = spectrum_plotter.plot_single_sided_power_spectral_density(filename=filename,
                                                                       title=plot_title,
                                                                       x_lim=zoom_xlim)
    plt.close(fig)


def problem_3():
    logger.log(logging.INFO, f"\n{'-'*10} PROBLEM 3 {'-'*10}\n")

    acoustic_intensity = 2.  # W/cm^2
    acoustic_intensity *= 10000.  # W/m^2
    frequency = 3000.  # Hz

    plane_wave = PlaneWave(acoustic_intensity, rho=RHO, c=C, f=frequency)
    logger.log(logging.INFO, f"Acoustic intensity: {plane_wave.acoustic_intensity} W/m^2\n"
                             f"Density: {plane_wave.medium_density} kg/m^3\n"
                             f"Speed of sound: {plane_wave.sound_speed_in_medium} m/s^2\n"
                             f"Frequency: {plane_wave.frequency} Hz\n")

    p_peak = plane_wave.peak_pressure()  # Pa
    p_peak_atm = plane_wave.peak_pressure(units=PressureUnits.atm)  # atm
    spl = plane_wave.sound_pressure_level()  # dB re 1 uPa
    spl_ubar = plane_wave.sound_pressure_level(p_ref=1, units=PressureUnits.ubar)  # dB re 1 uBar
    u_rms = plane_wave.rms_particle_velocity()  # m/s
    displacement_rms = plane_wave.rms_particle_displacement()  # m

    logger.log(logging.INFO, f"Peak pressure: {p_peak} Pa\n"
                             f"Peak pressure: {p_peak_atm} atm\n"
                             f"SPL: {spl} dB re 1 uPa\n"
                             f"SPL: {spl_ubar} dB re 1 ubar\n"
                             f"RMS particle velocity: {u_rms} m/s\n"
                             f"RMS particle displacement: {displacement_rms} m")


def problem_5():

    logger.log(logging.INFO, f"\n{'-'*10} PROBLEM 5 {'-'*10}\n")

    power = 1.0  # W
    frequency = 100.  # Hz
    diameter = 0.05  # m
    spherical_wave = SphericalWave(power=power, rho=RHO, c=C, f=frequency)

    radius = diameter/2  # m
    logger.log(logging.INFO, f"Sphere radius: {radius} m\n"
                             f"Acoustic power: {spherical_wave.power} W\n"
                             f"Medium density: {spherical_wave.medium_density} kg/m^3\n"
                             f"Speed of sound: {spherical_wave.sound_speed_in_medium} m/s^2\n"
                             f"Frequency: {spherical_wave.frequency} Hz\n")

    surface_peak_pressure = spherical_wave.peak_pressure(r=radius)  # Pa
    surface_peak_velocity = spherical_wave.peak_particle_velocity(r=radius)  # m/s
    surface_peak_displacement = spherical_wave.peak_particle_displacement(r=radius)  # m
    spl_100m = spherical_wave.sound_pressure_level(r=100)  # dB re 1 uPa
    spl_1m = spherical_wave.sound_pressure_level(r=1)  # dB re 1 u Pa

    logger.log(logging.INFO, f"Peak surface pressure: {surface_peak_pressure} Pa\n"
                             f"Peak surface velocity: {surface_peak_velocity} m/s\n"
                             f"Peak surface displacement: {surface_peak_displacement} m\n"
                             f"SPL @ 100m: {spl_100m} dB re 1 uPa\n"
                             f"SPL @ 1m: {spl_1m} dB re 1 uPa")


if __name__ == '__main__':
    """
    Note: Problems 4 and 6 are derivation problems, and do not require a code solution.
    """

    problems = [problem_1, problem_2, problem_3, problem_5]
    for problem in problems:
        problem()
