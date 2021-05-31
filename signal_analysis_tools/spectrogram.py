"""
    -*- coding: utf-8 -*-
    Time    : 2021-05-21 9:25 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: spectrogram.py
    
    {Description}
    -------------
    
"""

from signal_analysis_tools.timeseries import *
from scipy import fft
from typing import Union
import pandas as pd


class Spectrum:
    """
    Base class for storing Spectrum data. This class provides built-in functions for
    computing statistic on the underlying data.
    """
    def __init__(self,
                 frequencies: np.array,
                 amplitude: np.array,
                 bin_size: float,
                 name: str = ''):
        """

        Args:
            frequencies: frequency values (in Hz) for each frequency bin
            amplitude: measured values for each frequency bin (in sampled units)
            bin_size: the size (in Hz) of each frequency bin.
            name: a descriptor for the Spectrum.
        """
        self.data = np.array([*zip(frequencies, amplitude)], dtype=[('frequency', frequencies.dtype),
                                                                    ('amplitude', amplitude.dtype)])
        self.bin_size = bin_size
        self.name = name

    def frequency(self) -> np.array:
        """

        Returns: An array containing the frequency values (in Hz) of each bin.

        """
        return self.data['frequency']

    def sample_rate(self) -> float:
        """

        Returns: the sample rate of the corresponding timeseries.

        """
        return self.num_samples() * self.bin_size

    def duration(self) -> float:
        """

        Returns: The duration of the corresponding timeseries.

        """
        return self.num_samples() / self.sample_rate()

    def amplitude(self) -> np.array:
        """

        Returns: The linear amplitude of the spectrum for each frequency bin.

        """
        return self.data['amplitude']

    def magnitude(self) -> np.array:
        """

        Returns: The linear magnitude of the spectrum for each frequency bin

        """
        return np.abs(self.data['amplitude'])

    def num_samples(self) -> int:
        """

        Returns: The number of samples in the spectrum.

        """
        return len(self.data['frequency'])

    def nyquist(self) -> complex:
        """

        Returns: The amplitude of the Nyquist frequency.

        """
        return self.data['amplitude'][self.num_samples() // 2]

    def dc_offset(self) -> complex:
        """

        Returns: The amplitude of the 0 Hz component

        """
        return self.data['amplitude'][0]

    def positive_content(self) -> np.array:
        """

        Returns: The spectral content of the positive frequencies, including the DC offset and Nyquist.

        """
        return self.data['amplitude'][0:self.num_samples() // 2 + 1]

    def negative_content(self) -> np.array:
        """

        Returns: The spectral content of the negative frequencies, excluding the DC offset and Nyquist.

        """
        return self.data['amplitude'][self.num_samples() // 2 + 1:]

    def positive_frequencies(self) -> np.array:
        """

        Returns: The frequency axis (in Hz) of the positive frequency content.

        """
        return self.frequency()[0: self.num_samples() // 2 + 1]

    def single_sided_power_spectral_density(self) -> np.array:
        """

        Returns: The single-sided power spectral density of the spectrum.

        """
        duration = self.duration()
        positive_content = self.positive_content()
        weights = np.full(len(positive_content), fill_value=2)
        weights[0], weights[-1] = 1, 1
        return ((weights / duration) * np.conj(positive_content) * positive_content).real

    def double_sided_power_spectral_density(self) -> np.array:
        """

        Returns: The double-sided power spectral density of the spectrum.

        """
        return ((1. / self.duration()) * np.conj(self.data['amplitude']) * self.data['amplitude']).real


def pink(n) -> np.array:
    """
    This function returns the magnitude of each frequency bin for a pink spectrum.

    Args:
        n: number of frequency bins.

    Returns:

    """
    return 1. / np.sqrt(np.arange(n) + 1)


def random_phase(n: int) -> np.array:
    """
    This function returns a random phase for each frequency bin

    Args:
        n: number of frequency bins

    Returns:

    """
    return np.random.uniform(0, 2 * np.pi, (n,))


def generate_spectrum(n: int = 65536,
                      magnitude: Union[float, callable] = 1.0,
                      phase: Union[float, callable] = random_phase,
                      fs: float = 0.0,
                      dc_offset: float = 0.0,
                      spectral_density=True):
    """
    This function generates a Spectrum class based on a variable number of inputs.

    Args:
        n: The total number of frequency bins (positive + negative + DC + Nyquist)
        magnitude: Either a single value to apply to all frequency bins,
                   or a function that calculates the magnitude for each bin.
        phase: Either a single value to apply to all frequency bins,
               or a function that calculates the phase for each bin.
        fs: Sampling frequency of the corresponding timeseries.
        dc_offset: Value for the 0Hz component
        spectral_density: A boolean to indicate the bins are in a per Hz value or not.

    Returns: Spectrum

    """
    if fs == 0.0:
        f_res = 1.0
    else:
        f_res = fs / n

    # Calculate positive frequency magnitude
    if type(magnitude) is float or type(magnitude) is int:
        a = np.full(n // 2 - 1, fill_value=magnitude, dtype=float)
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

    if spectral_density:
        a /= f_res
        dc_offset /= f_res

    pos_freq = a * np.exp(1j * theta)

    # Create a complex value for the frequency domain.
    spectrum = np.zeros((n,), dtype=complex)
    spectrum[0] = dc_offset + 0j
    spectrum[1:n // 2] = pos_freq
    spectrum[n // 2] = 0 + 0j
    spectrum[n // 2 + 1:] = np.flip(np.conj(pos_freq))

    # Create frequency axis
    frequencies = np.arange(n) * f_res

    return Spectrum(frequencies, spectrum, f_res)


def timeseries_to_spectrum(timeseries: Timeseries) -> Spectrum:
    """
    Helper function to create the corresponding spectrum from a timeseries class.

    Args:
        timeseries: The timeseries in question.

    Returns: The corresponding Spectrum

    """
    spectrum = fft.fft(timeseries.data['amplitude'] / timeseries.sample_rate)
    frequency_resolution = 1 / timeseries.duration()
    frequency_axis = np.arange(len(timeseries.data['amplitude'])) * frequency_resolution
    return Spectrum(frequency_axis, spectrum, frequency_resolution)


def spectrum_to_timeseries(spectrum: Spectrum) -> Timeseries:
    """
    Helper function to create the corresponding timeseries from a Spectrum class.

    Args:
        spectrum: The spectrum in question.

    Returns: the corresponding Timeseries

    """
    amplitude = fft.ifft(spectrum.data['amplitude'] * spectrum.num_samples() * spectrum.bin_size)
    sample_rate = spectrum.sample_rate()
    time_axis = np.arange(spectrum.num_samples()) / sample_rate
    return Timeseries(time_axis, amplitude, sample_rate)


class SpectrumPlotter:
    """
    This class is a helper class to generate assignment-quality spectrum plots.
    """
    # Setup common graphics properties.
    TEXTBOX_PROPS = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.75}

    # Setup plotting styles.
    sns.plotting_context("paper")
    sns.set_style("darkgrid")

    def __init__(self, spectrum: Spectrum = None):
        """
        Initialize the plotting class.

        Args:
            spectrum: a Spectrum class to generate plots for.
        """
        self.spectrum = spectrum

    def set_spectrum(self, spectrum: Spectrum):
        """

        Args:
            spectrum: set the internal Spectrum class for plotting.

        Returns:

        """
        self.spectrum = spectrum

    def plot_spectrum(self,
                      x_label='frequency (Hz)',
                      y_label='linear amplitude (V)',
                      title='',
                      y_lim=None,
                      x_lim=None,
                      filename=None):
        """
        This function will generate a frequency domain plot and return the axis and figure.
        Both the real and imaginary components of the spectrum will be plotted.

        Args:
            x_label: Desired label for the x axis (frequency)
            y_label: Desired label for the y axis (amplitude)
            title: Desired title for the plot
            y_lim: A manual override to set the y axis limits
            x_lim: A manual override to set the x axis limits
            filename: If provided, will save the figure to the filename path.


        Returns: a Tuple of (axis, figure)

        """

        frequency = self.spectrum.frequency()
        frequency -= (np.max(frequency) - self.spectrum.bin_size) / 2

        amplitude = self.spectrum.amplitude()
        amplitude = np.roll(amplitude, len(amplitude) // 2 - 1)

        df = pd.DataFrame({'frequency': frequency,
                           'real': amplitude.real,
                           'imaginary': amplitude.imag})
        df = df.melt('frequency', var_name='spectrum', value_name='values')

        fig = plt.figure()
        spectrum_plot = sns.lineplot(data=df,
                                     x='frequency',
                                     y='values',
                                     hue='spectrum')
        spectrum_plot.set(xlabel=x_label, ylabel=y_label, title=title)
        set_minor_gridlines(spectrum_plot)

        spectrum_plot.get_legend().get_frame().update(self.TEXTBOX_PROPS)

        if y_lim is not None:
            plt.ylim(y_lim)

        if x_lim is not None:
            plt.xlim(x_lim)

        plt.tight_layout()

        if filename:
            fig.savefig(filename)

        return fig, spectrum_plot

    def plot_magnitude(self,
                       x_label='frequency (Hz)',
                       y_label='linear magnitude (V)',
                       title='',
                       y_lim=None,
                       x_lim=None,
                       positive_only=False,
                       filename=None):
        """
       This function will generate a frequency domain plot and return the axis and figure.
       Only the magnitude of the spectrum will be plotted.

       Args:
           x_label: Desired label for the x axis (frequency)
           y_label: Desired label for the y axis (amplitude)
           title: Desired title for the plot
           y_lim: A manual override to set the y axis limits
           x_lim: A manual override to set the x axis limits
           positive_only: A boolean to indicate whether to plot the negative content as well.
           filename: If provided, will save the figure to the filename path.


       Returns: a Tuple of (axis, figure)

       """

        if positive_only:
            frequency = self.spectrum.positive_frequencies()
            amplitude = self.spectrum.positive_content()
        else:
            frequency = self.spectrum.frequency()
            frequency -= (np.max(frequency) - self.spectrum.bin_size) / 2
            amplitude = self.spectrum.amplitude()
            amplitude = np.roll(amplitude, len(amplitude) // 2 - 1)

        fig = plt.figure()
        spectrum_plot = sns.lineplot(x=frequency, y=np.abs(amplitude))
        spectrum_plot.set(xlabel=x_label, ylabel=y_label, title=title)
        set_minor_gridlines(spectrum_plot)

        if y_lim is not None:
            plt.ylim(y_lim)

        if x_lim is not None:
            plt.xlim(x_lim)

        plt.tight_layout()

        if filename:
            fig.savefig(filename)

        return fig, spectrum_plot

    def plot_single_sided_power_spectral_density(self,
                                                 x_label='frequency (Hz)',
                                                 y_label=r'$G_{xx}\ (\frac{V^{2}}{Hz})$',
                                                 title='',
                                                 y_lim=None,
                                                 x_lim=None,
                                                 filename=None):
        """
        This function will plot the single-sided power spectral density of the Spectrum.

        Args:
            x_label: Desired label for the x axis (frequency)
            y_label: Desired label for the y axis (Gxx)
            title: Desired title for the plot
            y_lim: A manual override to set the y axis limits
            x_lim: A manual override to set the x axis limits
            filename: If provided, will save the figure to the filename path.

        Returns: a Tuple of (axis, figure)

        """

        f = self.spectrum.positive_frequencies()
        y = self.spectrum.single_sided_power_spectral_density()

        fig = plt.figure()
        psd_plot = sns.lineplot(x=f, y=y)
        psd_plot.set(xlabel=x_label, ylabel=y_label, title=title)
        set_minor_gridlines(psd_plot)

        if y_lim is not None:
            plt.ylim(y_lim)

        if x_lim is not None:
            plt.xlim(x_lim)

        plt.tight_layout()

        if filename:
            fig.savefig(filename)

        return fig, psd_plot
