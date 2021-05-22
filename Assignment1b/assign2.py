"""
    -*- coding: utf-8 -*-
    Time    : 2021-05-18 5:14 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: assign2.py
    
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


def assign2():
    def problem_1(spectrum16: Spectrum,
                  spectrum65536: Spectrum,
                  prob=1):
        spectrum_analyzer = SpectrumAnalyzer(spectrum16)
        fig, plot = spectrum_analyzer.plot_spectrum(x_label='frequency (Hz)',
                                                    y_label='amplitude (V)',
                                                    title=f'Assignment 1b, Problem {prob}\nReal and '
                                                          f'Imaginary spectrum, n=16',
                                                    filename=f'{PLOT_DIR}{os.sep}problem_{prob}_spectrum_n16.png')
        plt.close(fig)

        spectrum_analyzer.set_spectrum(spectrum65536)
        fig, plot = spectrum_analyzer.plot_spectrum(x_label='frequency (Hz)',
                                                    y_label='amplitude (V)',
                                                    title=f'Assignment 1b, Problem {prob}\nReal and '
                                                          f'Imaginary spectrum, n=65536',
                                                    filename=f'{PLOT_DIR}{os.sep}problem_{prob}_spectrum_n65536.png')
        plt.close(fig)

        timeseries = spectrum65536.to_timeseries()
        assert all(np.greater_equal(np.abs(timeseries.amplitude(only_real=False).real),
                                    np.abs(timeseries.amplitude(only_real=False).imag)))

        timeseries_analyzer = TimeseriesAnalyzer(timeseries)
        fig, plot = timeseries_analyzer.plot_time_domain(x_label='time (s)',
                                                         y_label='amplitude (V)',
                                                         title=f'Assignment 1b, Problem {prob}\nReal '
                                                               f'timeseries, n=65536',
                                                         filename=f'{PLOT_DIR}{os.sep}problem_{prob}_'
                                                                  f'timeseries_n65536.png',
                                                         stats=['mean', 'var', 'std', 'std_ratio'])
        plt.close(fig)

        fig, plot = timeseries_analyzer.plot_histogram(x_label='amplitude (V)',
                                                       y_label='probability density function',
                                                       title=f'Assignment 1b, Problem {prob}\nTimeseries histogram, '
                                                             f'n=65536',
                                                       filename=f'{PLOT_DIR}{os.sep}problem_{prob}_timeseries_histogram'
                                                                f'_n65536.png')

        plt.close(fig)

        fig, plot, stats = timeseries_analyzer.plot_normal_probability(x_label='amplitude (V)',
                                                                       title=f'Assignment 1b, Problem {prob}\n'
                                                                             f'Timeseries normal probability plot,'
                                                                             f' n=65536',
                                                                       filename=f'{PLOT_DIR}{os.sep}problem_{prob}'
                                                                                f'_timeseries_normal_prob_plot_'
                                                                                f'n65536.png')
        plt.close(fig)

    logger = setup_logging(assignment_name='1b')

    # Set random seed for reproducibility
    np.random.seed(100)

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
    ts_analyzer = TimeseriesAnalyzer(Timeseries(np.array([0]), np.array([0]), 1.0))
    for spectrum in [random_phase_spectrum65536, zero_phase_spectrum65536, pink_magnitude_spectrum65536]:
        ts_analyzer.set_timeseries(spectrum.to_timeseries())
        ts_analyzer.playback_timeseries(sample_rate=44100)

    """
    Problem 5
    """


if __name__ == '__main__':
    assign2()
