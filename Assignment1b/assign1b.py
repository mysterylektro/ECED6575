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
from signal_analysis_tools.spectrogram import *
from signal_analysis_tools.utilities import setup_logging

# Setup output directories.
PLOT_DIR = os.sep.join([os.path.dirname(__file__), 'plots'])
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


def problem_1(spectrum16: Spectrum,
              spectrum65536: Spectrum,
              prob=1):
    timeseries_plotter = TimeseriesPlotter()
    spectrum_plotter = SpectrumPlotter()

    spectrum_plotter.set_spectrum(spectrum16)
    fig, plot = spectrum_plotter.plot_spectrum(x_label='frequency (Hz)',
                                               y_label='amplitude (V)',
                                               title=f'Assignment 1b, Problem {prob}\nReal and '
                                                     f'Imaginary spectrum, n=16',
                                               filename=f'{PLOT_DIR}{os.sep}problem_{prob}_spectrum_n16.png')
    plt.close(fig)

    spectrum_plotter.set_spectrum(spectrum65536)
    fig, plot = spectrum_plotter.plot_spectrum(x_label='frequency (Hz)',
                                               y_label='amplitude (V)',
                                               title=f'Assignment 1b, Problem {prob}\nReal and '
                                                     f'Imaginary spectrum, n=65536',
                                               filename=f'{PLOT_DIR}{os.sep}problem_{prob}_spectrum_n65536.png')
    plt.close(fig)

    timeseries = spectrum_to_timeseries(spectrum65536)
    assert all(np.greater_equal(np.abs(timeseries.amplitude(only_real=False).real),
                                np.abs(timeseries.amplitude(only_real=False).imag)))

    timeseries_plotter.set_timeseries(timeseries)
    fig, plot = timeseries_plotter.plot_time_domain(x_label='time (s)',
                                                    y_label='amplitude (V)',
                                                    title=f'Assignment 1b, Problem {prob}\nReal '
                                                          f'timeseries, n=65536',
                                                    filename=f'{PLOT_DIR}{os.sep}problem_{prob}_'
                                                             f'timeseries_n65536.png',
                                                    stats=['mean', 'var', 'std', 'std_ratio'])
    plt.close(fig)

    fig, plot = timeseries_plotter.plot_histogram(x_label='amplitude (V)',
                                                  y_label='probability density function',
                                                  title=f'Assignment 1b, Problem {prob}\nTimeseries histogram, '
                                                        f'n=65536',
                                                  filename=f'{PLOT_DIR}{os.sep}problem_{prob}_timeseries_histogram'
                                                           f'_n65536.png')

    plt.close(fig)

    fig, plot, stats = timeseries_plotter.plot_normal_probability(x_label='amplitude (V)',
                                                                  title=f'Assignment 1b, Problem {prob}\n'
                                                                        f'Timeseries normal probability plot,'
                                                                        f' n=65536',
                                                                  filename=f'{PLOT_DIR}{os.sep}problem_{prob}'
                                                                           f'_timeseries_normal_prob_plot_'
                                                                           f'n65536.png')
    plt.close(fig)


def assign1b():
    # Setup logging
    logger = setup_logging(assignment_name='1b')

    # Set random seed for reproducibility
    np.random.seed(100)

    ts_plotter = TimeseriesPlotter()
    spectrum_plotter = SpectrumPlotter()

    """
    Problem 1
    """
    random_phase_spectrum16 = generate_spectrum(n=16)
    random_phase_spectrum65536 = generate_spectrum(n=65536)
    problem_1(random_phase_spectrum16, random_phase_spectrum65536, prob=1)

    """
    Problem 2
    """
    zero_phase_spectrum16 = generate_spectrum(n=16, phase=0)
    zero_phase_spectrum65536 = generate_spectrum(n=65536, phase=0)
    problem_1(zero_phase_spectrum16, zero_phase_spectrum65536, prob=2)

    """
    Problem 3
    """
    pink_magnitude_spectrum16 = generate_spectrum(n=16, magnitude=pink)
    pink_magnitude_spectrum65536 = generate_spectrum(n=65536, magnitude=pink)
    problem_1(pink_magnitude_spectrum16, pink_magnitude_spectrum65536, prob=3)

    """
    Problem 4
    """
    for spectrum in [random_phase_spectrum65536, zero_phase_spectrum65536, pink_magnitude_spectrum65536]:
        timeseries = spectrum.to_timeseries()
        timeseries.set_sample_rate(44100)
        playback_timeseries(timeseries)

    """
    Problem 5
    """

    timeseries = spectrum_to_timeseries(random_phase_spectrum65536)
    timeseries.set_sample_rate(44100)
    recorded_timeseries = play_and_record_timeseries(timeseries)
    ts_plotter.set_timeseries(recorded_timeseries)
    fig, plot = ts_plotter.plot_time_domain(x_label='time (s)',
                                            y_label='amplitude (α V)',
                                            title='Assignment 1b, Problem 5\nRecorded white noise playback',
                                            filename=f'{PLOT_DIR}{os.sep}problem_5_white_noise_playback.png')
    plt.close(fig)

    recorded_spectrum = timeseries_to_spectrum(recorded_timeseries)
    spectrum_plotter.set_spectrum(recorded_spectrum)

    fig, plot = spectrum_plotter.plot_magnitude(x_label='frequency (Hz)',
                                                y_label='linear magnitude (α V)',
                                                x_lim=[0, 20000],
                                                y_lim=[0, 10],
                                                positive_only=True,
                                                title='Assignment 1b, Problem 5\n'
                                                      'Spectrum of recorded white noise playback',
                                                filename=f'{PLOT_DIR}{os.sep}problem_5_white_noise_playback_'
                                                         f'spectrum.png')
    plt.close(fig)

    # Duplicate timeseries.
    amplitude = np.concatenate([timeseries.amplitude(), timeseries.amplitude()])
    t = np.arange(len(amplitude)) / timeseries.sample_rate
    timeseries = Timeseries(t, amplitude, timeseries.sample_rate)
    recorded_timeseries = play_and_record_timeseries(timeseries)
    ts_plotter.set_timeseries(recorded_timeseries)
    fig, plot = ts_plotter.plot_time_domain(x_label='time (s)',
                                            y_label='amplitude (α V)',
                                            title='Assignment 1b, Problem 5\n'
                                                  'Double-length recorded white noise playback',
                                            filename=f'{PLOT_DIR}{os.sep}problem_5_double_white_noise_playback.png')
    plt.close(fig)

    timeseries_subset = recorded_timeseries.subset(0.8, 0.8 + timeseries.duration() / 2, zero_mean=True)
    ts_plotter.set_timeseries(timeseries_subset)
    fig, plot = ts_plotter.plot_time_domain(x_label='time (s)',
                                            y_label='amplitude (α V)',
                                            title='Assignment 1b, Problem 5\nSubset of double-length '
                                                  'recorded white noise playback',
                                            filename=f'{PLOT_DIR}{os.sep}problem_5_double_white_noise_'
                                                     f'playback_subset.png')
    plt.close(fig)

    subset_spectrum = timeseries_to_spectrum(timeseries_subset)
    spectrum_plotter = SpectrumPlotter(subset_spectrum)
    fig, plot = spectrum_plotter.plot_magnitude(x_label='frequency (Hz)',
                                                y_label='linear magnitude (α V)',
                                                x_lim=[0, 20000],
                                                y_lim=[0, 10],
                                                positive_only=True,
                                                title='Assignment 1b, Problem 5\n'
                                                      'Spectrum of recorded white noise subset',
                                                filename=f'{PLOT_DIR}{os.sep}problem_5_double_white_noise_playback_'
                                                         f'subset_spectrum.png')
    plt.close(fig)

    """
    Problem 6
    """

    filename = 'TRAC1_noise_time.csv'
    basename = os.path.basename(filename)
    directory = 'Assign 1b - Linear Spectrum and Averaging'
    ts = timeseries_from_csv(f'.{os.sep}{directory}{os.sep}{filename}')
    spectrum = timeseries_to_spectrum(ts)
    spectrum_plotter = SpectrumPlotter(spectrum)
    fig, plot = spectrum_plotter.plot_single_sided_power_spectral_density(title='Assignment 1b, Problem 6\n'
                                                                                f'Single sided power spectral density'
                                                                                f' of {basename}',
                                                                          filename=f'{PLOT_DIR}{os.sep}problem_6_gxx_'
                                                                                   f'{basename}.png')

    plt.close(fig)

    mean_square_value = ts.rms() ** 2
    double_sided_integral = np.sum(spectrum.double_sided_power_spectral_density() * spectrum.bin_size)
    single_sided_integral = np.sum(spectrum.single_sided_power_spectral_density() * spectrum.bin_size)

    print(f'Mean square value: {mean_square_value}\n'
          f'Double sided integral: {double_sided_integral}\n'
          f'Single sided integral: {single_sided_integral}')

    """
    Problem 7
    """

    filename = 'TRAC3_sin100_time.csv'
    basename = os.path.basename(filename)
    directory = 'Assign 1b - Linear Spectrum and Averaging'
    ts = timeseries_from_csv(f'.{os.sep}{directory}{os.sep}{filename}')
    spectrum = timeseries_to_spectrum(ts)
    spectrum_plotter.set_spectrum(spectrum)

    fig, plot = spectrum_plotter.plot_single_sided_power_spectral_density(title='Assignment 1b, Problem 7\n'
                                                                                f'Single sided power spectral density'
                                                                                f' of {basename}',
                                                                          filename=f'{PLOT_DIR}{os.sep}problem_7_gxx_'
                                                                                   f'{basename}.png')

    plt.close(fig)

    mean_square_value = ts.rms() ** 2
    double_sided_integral = np.sum(spectrum.double_sided_power_spectral_density() * spectrum.bin_size)
    single_sided_integral = np.sum(spectrum.single_sided_power_spectral_density() * spectrum.bin_size)

    print(f'Mean square value: {mean_square_value}\n'
          f'Double sided integral: {double_sided_integral}\n'
          f'Single sided integral: {single_sided_integral}')

    print(f'Timeseries RMS: {ts.rms()}')
    print(f'Gxx RMS: {np.sqrt(np.max(spectrum.single_sided_power_spectral_density()) * spectrum.bin_size)}')

    """
    Problem 8
    """

    ts = record_timeseries(duration=2.0, sample_rate=44100)
    ts_plotter.set_timeseries(ts)
    fig, plot = ts_plotter.plot_time_domain(x_label='time (s)', y_label=r'$amplitude\ (\alpha\ V)$',
                                            filename=f'{PLOT_DIR}{os.sep}problem_8_recorded_signal_timeseries.png')
    plt.close(fig)

    spectrum = timeseries_to_spectrum(ts)
    spectrum_plotter.set_spectrum(spectrum)
    fig, plot = spectrum_plotter.plot_single_sided_power_spectral_density(title='Assignment 1b, Problem 8\n'
                                                                                f'Single sided power spectral density '
                                                                                f'of recorded signal',
                                                                          filename=f'{PLOT_DIR}{os.sep}problem_8_gxx_'
                                                                                   f'recorded_signal.png')
    plt.semilogy()
    fig.savefig('{PLOT_DIR}{os.sep}problem_8_gxx_recorded_signal_semilogy.png')
    plt.close(fig)


if __name__ == '__main__':
    assign1b()
