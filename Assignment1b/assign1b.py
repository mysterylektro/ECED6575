"""
    -*- coding: utf-8 -*-
    Time    : 2021-05-18 5:14 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: assign1b.py
    
    {Description}
    -------------
    
"""

import os
import logging
from signal_analysis_tools.spectrogram import *
from signal_analysis_tools.utilities import setup_logging

# Setup output directories.
PLOT_DIR = os.sep.join([os.path.dirname(__file__), 'plots'])
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


def problem_1(spectrum16: Spectrum,
              spectrum65536: Spectrum,
              logger=None,
              prob=1,
              overwrite=False):

    timeseries_plotter = TimeseriesPlotter()
    spectrum_plotter = SpectrumPlotter()

    plot_filename = f'{PLOT_DIR}{os.sep}problem_{prob}_spectrum_n16.png'
    if not os.path.exists(plot_filename) or overwrite:
        logger.log(logging.DEBUG, "Plotting N = 16 spectrum...")
        spectrum_plotter.set_spectrum(spectrum16)
        fig, plot = spectrum_plotter.plot_spectrum(x_label='frequency (Hz)',
                                                   y_label='amplitude (V)',
                                                   title=f'Assignment 1b, Problem {prob}\nReal and '
                                                         f'Imaginary spectrum, n=16',
                                                   filename=plot_filename)
        plt.close(fig)

    plot_filename = f'{PLOT_DIR}{os.sep}problem_{prob}_spectrum_n65536.png'
    if not os.path.exists(plot_filename) or overwrite:
        logger.log(logging.DEBUG, "Plotting N = 65536 spectrum...")
        spectrum_plotter.set_spectrum(spectrum65536)
        fig, plot = spectrum_plotter.plot_spectrum(x_label='frequency (Hz)',
                                                   y_label='amplitude (V)',
                                                   title=f'Assignment 1b, Problem {prob}\nReal and '
                                                         f'Imaginary spectrum, n=65536',
                                                   filename=plot_filename)
        plt.close(fig)

    logger.log(logging.DEBUG, "Generating timeseries for N = 65536 spectrum...")
    timeseries = spectrum_to_timeseries(spectrum65536)

    assert all(np.greater_equal(np.abs(timeseries.amplitude(only_real=False).real),
                                np.abs(timeseries.amplitude(only_real=False).imag)))

    logger.log(logging.INFO, f"Problem {prob}: all real timeseries values are greater than or "
                             f"equal to their imaginary counterparts.\n")

    plot_filename = f'{PLOT_DIR}{os.sep}problem_{prob}_timeseries_n65536.png'
    if not os.path.exists(plot_filename) or overwrite:
        logger.log(logging.DEBUG, "Plotting timeseries for N = 65536 timeseries...")
        timeseries_plotter.set_timeseries(timeseries)
        fig, plot = timeseries_plotter.plot_time_domain(x_label='time (s)',
                                                        y_label='amplitude (V)',
                                                        title=f'Assignment 1b, Problem {prob}\nReal '
                                                              f'timeseries, n=65536',
                                                        filename=plot_filename,
                                                        stats=['mean', 'var', 'std', 'std_ratio'])
        plt.close(fig)

    plot_filename = f'{PLOT_DIR}{os.sep}problem_{prob}_timeseries_histogram_n65536.png'
    if not os.path.exists(plot_filename) or overwrite:
        logger.log(logging.DEBUG, "Plotting histogram with bin model = sqrt for N = 65536 timeseries...")
        fig, plot = timeseries_plotter.plot_histogram(x_label='amplitude (V)',
                                                      y_label='probability density function',
                                                      title=f'Assignment 1b, Problem {prob}\nTimeseries histogram, '
                                                            f'n=65536',
                                                      filename=plot_filename)

        plt.close(fig)

    plot_filename = f'{PLOT_DIR}{os.sep}problem_{prob}_timeseries_normal_prob_plot_n65536.png'
    if not os.path.exists(plot_filename) or overwrite:
        logger.log(logging.DEBUG, "Plotting normal probability plot for N = 65536 timeseries...")
        fig, plot, stats = timeseries_plotter.plot_normal_probability(x_label='amplitude (V)',
                                                                      title=f'Assignment 1b, Problem {prob}\n'
                                                                            f'Timeseries normal probability plot,'
                                                                            f' n=65536',
                                                                      filename=plot_filename)
        plt.close(fig)


def assign1b(problems=None):
    # Setup logging
    logger = setup_logging(assignment_name='1b')

    if problems is None:
        problems = list(range(1, 9))

    logger.log(logging.INFO, "---------- Starting Assignment 1B ----------\n")

    # Set random seed for reproducibility
    np.random.seed(100)

    ts_plotter = TimeseriesPlotter()
    spectrum_plotter = SpectrumPlotter()

    """
    Problem 1
    """
    if 1 in problems:
        logger.log(logging.INFO, "\n---------- Starting problem 1 ----------\n")

        logger.log(logging.DEBUG, "Generating random phase spectrum for N = 16 and N = 65536...")
        random_phase_spectrum16 = generate_spectrum(n=16)
        random_phase_spectrum65536 = generate_spectrum(n=65536)
        problem_1(random_phase_spectrum16, random_phase_spectrum65536, logger=logger, prob=1)

    """
    Problem 2
    """
    if 2 in problems:
        logger.log(logging.INFO, "\n---------- Starting problem 2 ----------\n")

        logger.log(logging.DEBUG, "Generating zero phase spectrum for N = 16 and N = 65536...")
        zero_phase_spectrum16 = generate_spectrum(n=16, phase=0)
        zero_phase_spectrum65536 = generate_spectrum(n=65536, phase=0)
        problem_1(zero_phase_spectrum16, zero_phase_spectrum65536, logger=logger, prob=2)

    """
    Problem 3
    """
    if 3 in problems:
        logger.log(logging.INFO, "\n---------- Starting problem 3 ----------\n")

        logger.log(logging.DEBUG, "Generating pink magnitude spectrum for N = 16 and N = 65536...")
        pink_magnitude_spectrum16 = generate_spectrum(n=16, magnitude=pink)
        pink_magnitude_spectrum65536 = generate_spectrum(n=65536, magnitude=pink)
        problem_1(pink_magnitude_spectrum16, pink_magnitude_spectrum65536, logger=logger, prob=3)

    """
    Problem 4
    """
    if 4 in problems:
        if any(i not in problems for i in [1, 2, 3]):
            raise ValueError("Must complete problems 1, 2, and 3 in order to complete problem 4")

        logger.log(logging.INFO, "\n---------- Starting problem 4 ----------\n")

        logger.log(logging.DEBUG, "Playing timeseries for each N = 65536 spectra...")
        for spectrum in [random_phase_spectrum65536, zero_phase_spectrum65536, pink_magnitude_spectrum65536]:
            timeseries = spectrum_to_timeseries(spectrum)
            timeseries.set_sample_rate(44100)
            playback_timeseries(timeseries)

    """
    Problem 5
    """
    if 5 in problems:
        if 1 not in problems:
            raise ValueError("Must complete problem 1 before completing problem 5")
        logger.log(logging.INFO, "\n---------- Starting problem 5 ----------\n")

        timeseries = spectrum_to_timeseries(random_phase_spectrum65536)
        timeseries.set_sample_rate(44100)

        logger.log(logging.DEBUG, "Starting playback and record of timeseries of random phase spectrum...")
        recorded_timeseries = play_and_record_timeseries(timeseries)
        # Remove DC offset
        recorded_timeseries.data['amplitude'] = recorded_timeseries.data['amplitude'] - recorded_timeseries.mean()

        logger.log(logging.DEBUG, "Plotting recorded timeseries...")
        ts_plotter.set_timeseries(recorded_timeseries)
        fig, plot = ts_plotter.plot_time_domain(x_label='time (s)',
                                                y_label='amplitude (α V)',
                                                title='Assignment 1b, Problem 5\nRecorded white noise playback',
                                                filename=f'{PLOT_DIR}{os.sep}problem_5_white_noise_playback.png')
        plt.close(fig)

        logger.log(logging.DEBUG, "Generating spectrum from recorded data...")
        recorded_spectrum = timeseries_to_spectrum(recorded_timeseries)

        spectrum_plotter.set_spectrum(recorded_spectrum)
        logger.log(logging.DEBUG, "Plotting spectrum from recorded data...")
        fig, plot = spectrum_plotter.plot_magnitude(x_label='frequency (Hz)',
                                                    y_label='linear magnitude (α V)',
                                                    x_lim=[0, 20000],
                                                    positive_only=True,
                                                    title='Assignment 1b, Problem 5\n'
                                                          'Spectrum of recorded white noise playback',
                                                    filename=f'{PLOT_DIR}{os.sep}problem_5_white_noise_playback_'
                                                             f'spectrum.png')
        plt.semilogy()
        fig.savefig(f'{PLOT_DIR}{os.sep}problem_5_white_noise_playback_spectrum_semilogy.png')
        plt.close(fig)

        # Duplicate timeseries.
        logger.log(logging.DEBUG, "Concatenating random phase timeseries into two duplicates...")
        amplitude = np.concatenate([timeseries.amplitude(), timeseries.amplitude()])
        timeseries = Timeseries(amplitude, timeseries.sample_rate)

        logger.log(logging.DEBUG, "Starting playback and record of duplicated timeseries of random phase spectrum...")
        recorded_timeseries = play_and_record_timeseries(timeseries)

        # Remove DC offset
        recorded_timeseries.data['amplitude'] = recorded_timeseries.data['amplitude'] - recorded_timeseries.mean()

        logger.log(logging.DEBUG, "Plotting recorded timeseries...")
        ts_plotter.set_timeseries(recorded_timeseries)
        fig, plot = ts_plotter.plot_time_domain(x_label='time (s)',
                                                y_label='amplitude (α V)',
                                                title='Assignment 1b, Problem 5\n'
                                                      'Double-length recorded white noise playback',
                                                filename=f'{PLOT_DIR}{os.sep}problem_5_double_white_noise_playback.png')
        plt.close(fig)

        logger.log(logging.DEBUG, "Generating subset from recorded data...")
        timeseries_subset = recorded_timeseries.subset(0.8, 0.8 + timeseries.duration() / 2, zero_mean=True)

        logger.log(logging.DEBUG, "Plotting subset from recorded data...")
        ts_plotter.set_timeseries(timeseries_subset)
        fig, plot = ts_plotter.plot_time_domain(x_label='time (s)',
                                                y_label='amplitude (α V)',
                                                title='Assignment 1b, Problem 5\nSubset of double-length '
                                                      'recorded white noise playback',
                                                filename=f'{PLOT_DIR}{os.sep}problem_5_double_white_noise_'
                                                         f'playback_subset.png')
        plt.close(fig)

        logger.log(logging.DEBUG, "Generating spectrum from recorded subset...")
        subset_spectrum = timeseries_to_spectrum(timeseries_subset)

        logger.log(logging.DEBUG, "Plotting positive frequency content from recorded subset spectrum...")
        spectrum_plotter = SpectrumPlotter(subset_spectrum)
        fig, plot = spectrum_plotter.plot_magnitude(x_label='frequency (Hz)',
                                                    y_label='linear magnitude (α V)',
                                                    x_lim=[0, 20000],
                                                    positive_only=True,
                                                    title='Assignment 1b, Problem 5\n'
                                                          'Spectrum of recorded white noise subset',
                                                    filename=f'{PLOT_DIR}{os.sep}problem_5_double_white_noise_playback_'
                                                             f'subset_spectrum.png')
        plt.semilogy()
        fig.savefig(f'{PLOT_DIR}{os.sep}problem_5_double_white_noise_playback_subset_spectrum_semilogy.png')
        plt.close(fig)

    """
    Problem 6
    """
    if 6 in problems:
        logger.log(logging.INFO, "\n---------- Starting problem 6 ----------\n")

        filename = 'TRAC1_noise_time.csv'
        basename = os.path.basename(filename)
        directory = 'Assign 1b - Linear Spectrum and Averaging'

        logger.log(logging.DEBUG, f"Loading in file {filename}")
        ts = timeseries_from_csv(f'.{os.sep}{directory}{os.sep}{filename}')

        logger.log(logging.DEBUG, f"Generating spectrum of file {filename}")
        spectrum = timeseries_to_spectrum(ts)

        logger.log(logging.DEBUG, f"Plotting single sided power spectral density of file {filename}")
        spectrum_plotter = SpectrumPlotter(spectrum)
        fig, plot = spectrum_plotter.plot_gxx(title='Assignment 1b, Problem 6\n'
                                                                                    f'Single sided power spectral '
                                                                                    f'density of {basename}',
                                              filename=f'{PLOT_DIR}{os.sep}problem_6_'
                                                                                       f'gxx_{basename}.png')

        plt.close(fig)

        logger.log(logging.INFO, "Calculating mean square value, double sided integral value, "
                                 "and single sided integral value...\n")
        mean_square_value = ts.mean_square()
        double_sided_integral = np.sum(spectrum.sxx() * spectrum.f_res)
        single_sided_integral = np.sum(spectrum.gxx() * spectrum.f_res)

        logger.log(logging.INFO, f'Mean square value: {mean_square_value}\n'
                                 f'Double sided integral: {double_sided_integral}\n'
                                 f'Single sided integral: {single_sided_integral}\n')

    """
    Problem 7
    """
    if 7 in problems:
        logger.log(logging.INFO, "\n---------- Starting problem 7 ----------\n")

        filename = 'TRAC3_sin100_time.csv'
        basename = os.path.basename(filename)
        directory = 'Assign 1b - Linear Spectrum and Averaging'

        logger.log(logging.DEBUG, f"Loading in file {filename}")
        ts = timeseries_from_csv(f'.{os.sep}{directory}{os.sep}{filename}')

        logger.log(logging.DEBUG, f"Generating spectrum of file {filename}")
        spectrum = timeseries_to_spectrum(ts)

        logger.log(logging.DEBUG, f"Plotting single sided power spectral density of file {filename}")
        spectrum_plotter.set_spectrum(spectrum)
        fig, plot = spectrum_plotter.plot_gxx(title='Assignment 1b, Problem 7\n'
                                                                                    f'Single sided power spectral '
                                                                                    f'density of {basename}',
                                              filename=f'{PLOT_DIR}{os.sep}problem_7_'
                                                                                       f'gxx_{basename}.png')

        plt.close(fig)

        logger.log(logging.INFO, "Calculating mean square value, double sided integral value, "
                                 "and single sided integral value...\n")
        mean_square_value = ts.rms() ** 2
        double_sided_integral = np.sum(spectrum.sxx() * spectrum.f_res)
        single_sided_integral = np.sum(spectrum.gxx() * spectrum.f_res)

        logger.log(logging.INFO, f'Mean square value: {mean_square_value}\n'
                                 f'Double sided integral: {double_sided_integral}\n'
                                 f'Single sided integral: {single_sided_integral}\n')

        logger.log(logging.INFO, f"Calculating rms of {filename} and "
                                 f"rms estimate from single sided power spectral density...\n")

        rms = ts.rms()
        gxx_rms = np.sqrt(np.max(spectrum.gxx()) * spectrum.f_res)

        logger.log(logging.INFO, f'Timeseries RMS: {rms}\nGxx RMS: {gxx_rms}\n')

    """
    Problem 8
    """
    if 8 in problems:
        logger.log(logging.INFO, "\n---------- Starting problem 8 ----------\n")

        logger.log(logging.DEBUG, "Starting timeseries recording...")
        ts = record_timeseries(duration=2.0, sample_rate=44100)

        logger.log(logging.DEBUG, "Plotting timeseries recording...")
        ts_plotter.set_timeseries(ts)
        fig, plot = ts_plotter.plot_time_domain(x_label='time (s)',
                                                y_label=r'$amplitude\ (\alpha\ V)$',
                                                title='Assignment 1b, Problem 8\nRecorded whistling through microphone',
                                                filename=f'{PLOT_DIR}{os.sep}problem_8_recorded_signal_timeseries.png')
        plt.close(fig)

        logger.log(logging.DEBUG, "Generating spectrum from timeseries...")
        spectrum = timeseries_to_spectrum(ts)

        logger.log(logging.DEBUG, "Plotting single sided power spectral density...")
        spectrum_plotter.set_spectrum(spectrum)
        fig, plot = spectrum_plotter.plot_gxx(title='Assignment 1b, Problem 8\n'
                                                                                    f'Single sided power spectral '
                                                                                    f'density of recorded signal',
                                              filename=f'{PLOT_DIR}{os.sep}problem_8_'
                                                                                       f'gxx_recorded_signal.png')
        logger.log(logging.DEBUG, "Setting y axis to semilogy...")
        plt.semilogy()

        logger.log(logging.DEBUG, "Saving semilogy plot...")
        fig.savefig(f'{PLOT_DIR}{os.sep}problem_8_gxx_recorded_signal_semilogy.png')
        plt.close(fig)

    logger.log(logging.INFO, "\nAssignment 1b complete!\n")


if __name__ == '__main__':
    """
    Select the problems you wish to solve. Note: Some problems require the completion of others
    in order to complete. 
    """
    p = [1, 2, 3, 4, 5, 6, 7, 8]
    assign1b(problems=p)
