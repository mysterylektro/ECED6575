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
import numpy as np
from scipy.fft import ifft
from typing import Union
import seaborn as sns
from Assignment1a.assign1 import plot_timeseries

# Setup output directories.
PLOT_DIR = os.sep.join([os.path.dirname(__file__), 'plots'])
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Setup plotting styles.
sns.plotting_context("paper")
sns.set_style("darkgrid")


def pink(n):
    """

    Args:
        n:

    Returns:

    """
    return 1. / np.sqrt(np.arange(n) + 1)


def random_phase(n: int):
    """

    Args:
        n:

    Returns:

    """
    return np.random.uniform(0, 2 * np.pi, (n,))


def generate_spectrum(n: int = 65536,
                      magnitude: Union[float, callable] = 1.0,
                      phase: Union[float, callable] = random_phase,
                      fs: float = 0.0,
                      dc_offset: float = 0.0,
                      spectral_density=True):
    if fs == 0.0:
        f_res = 1.0
    else:
        f_res = fs / n

    # Calculate positive frequency magnitude
    if type(magnitude) is float or type(magnitude) is int:
        a = np.full(n // 2 - 1, fill_value=magnitude)
    elif callable(magnitude):
        try:
            a = magnitude(n // 2 - 1)
        except ValueError:
            raise ValueError("magnitude function must accept only one input argument: the number of elements.")
    else:
        raise ValueError("Magnitude must be either a float or a callable function")

    if type(phase) is float or type(phase) is int:
        theta = np.full(n // 2 - 1, fill_value=phase)
    elif callable(phase):
        try:
            theta = phase(n // 2 - 1)
        except ValueError:
            raise ValueError("phase function must accept only one input argument: the number of elements.")
    else:
        raise ValueError("phase must be either a float or a callable function")

    # Calculate positive frequency magnitude
    offset = np.array([dc_offset + 0j])
    nyquist_freq = np.array([0 + 0j])

    if spectral_density:
        a /= f_res
        offset /= f_res

    pos_freq = a * np.exp(1j * theta)
    neg_freq = np.flip(np.conj(pos_freq))

    # Create a complex value for the frequency domain.
    spectrum = np.concatenate([offset, pos_freq, nyquist_freq, neg_freq])

    # Construct the timeseries from the inverse fourier transform
    timeseries = ifft(spectrum * n * f_res)

    return timeseries, spectrum


def assign2():
    """
    Problem 1
    """

    random_phase_unit_magnitude = generate_spectrum()

    fig, random_phase_unit_magnitude_timeseries = plot_timeseries(x=np.arange(len(random_phase_unit_magnitude[0])),
                                                                  y=random_phase_unit_magnitude[0].real,
                                                                  x_label='time',
                                                                  y_label='amplitude',
                                                                  title='Timeseries\nRandom Phase, Magnitude = 1')

    fig.savefig(f'{PLOT_DIR}{os.sep}timeseries_random_phase_mag_1.png')

    fig, random_phase_unit_magnitude_spectrum = plot_timeseries(x=np.arange(len(random_phase_unit_magnitude[1])),
                                                                y=np.abs(random_phase_unit_magnitude[1]),
                                                                x_label='frequency',
                                                                y_label='amplitude',
                                                                title='Spectrum\nRandom Phase, Magnitude = 1')

    fig.savefig(f'{PLOT_DIR}{os.sep}spectrum_random_phase_mag_1.png')

    """
    Phase = 0
    """

    zero_phase_unit_magnitude = generate_spectrum(phase=0)

    fig, zero_phase_unit_magnitude_timeseries = plot_timeseries(x=np.arange(len(zero_phase_unit_magnitude[0])),
                                                                y=zero_phase_unit_magnitude[0].real,
                                                                x_label='time',
                                                                y_label='amplitude',
                                                                title='Timeseries\nZero Phase, Magnitude = 1')

    fig.savefig(f'{PLOT_DIR}{os.sep}timeseries_zero_phase_mag_1.png')

    fig, zero_phase_unit_magnitude_spectrum = plot_timeseries(x=np.arange(len(zero_phase_unit_magnitude[1])),
                                                              y=np.abs(zero_phase_unit_magnitude[1]),
                                                              x_label='frequency',
                                                              y_label='amplitude',
                                                              title='Spectrum\nZero Phase, Magnitude = 1')

    fig.savefig(f'{PLOT_DIR}{os.sep}spectrum_zero_phase_mag_1.png')

    """
    Magnitude = pink
    """

    random_phase_pink_magnitude = generate_spectrum(magnitude=pink)

    fig, random_phase_pink_magnitude_timeseries = plot_timeseries(x=np.arange(len(random_phase_pink_magnitude[0])),
                                                                  y=random_phase_pink_magnitude[0].real,
                                                                  x_label='time',
                                                                  y_label='amplitude',
                                                                  title='Timeseries\nRandom Phase, Magnitude = Pink')

    fig.savefig(f'{PLOT_DIR}{os.sep}timeseries_random_phase_mag_pink.png')

    fig, random_phase_pink_magnitude_spectrum = plot_timeseries(x=np.arange(len(random_phase_pink_magnitude[1])),
                                                                y=np.abs(random_phase_pink_magnitude[1].real),
                                                                x_label='frequency',
                                                                y_label='amplitude',
                                                                title='Spectrum\nRandom Phase, Magnitude = Pink')

    fig.savefig(f'{PLOT_DIR}{os.sep}spectrum_random_phase_mag_pink.png')


if __name__ == '__main__':
    assign2()
