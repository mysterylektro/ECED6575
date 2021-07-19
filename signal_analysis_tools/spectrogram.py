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
from scipy.signal import windows
from typing import Union
import pandas as pd


class Spectrum:
    """
    Base class for storing Spectrum data. This class provides built-in functions for
    computing statistic on the underlying data.
    """

    def __init__(self,
                 amplitudes: np.array,
                 f_res: float,
                 name: str = '',
                 window=None):
        """

        Args:
            amplitudes: measured values for each frequency bin (in sampled units)
            f_res: the size (in Hz) of each frequency bin.
            name: a descriptor for the Spectrum.
        """
        self.f_res = f_res
        self.name = name
        if window is None:
            self.window = np.ones(len(amplitudes))
        else:
            self.window = window

        freq_axis = np.arange(len(amplitudes)) * self.f_res
        self.data = np.array([*zip(freq_axis, amplitudes)], dtype=[('frequency', freq_axis.dtype),
                                                                   ('amplitude', amplitudes.dtype)])

    def effective_frequency_resolution(self):
        return self.f_res * len(self.window) * np.sum(self.window ** 2) / np.sum(self.window) ** 2

    def energy_window_correction(self):
        return len(self.window) / np.sum(self.window ** 2)

    def amplitude_window_correction(self):
        return (len(self.window) / np.sum(self.window))**2

    def frequency(self) -> np.array:
        """

        Returns: An array containing the frequency values (in Hz) of each bin.

        """
        return self.data['frequency']

    def sample_rate(self) -> float:
        """

        Returns: the sample rate of the corresponding timeseries.

        """
        return self.num_samples() * self.f_res

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

    def gxx(self, fixed_energy=True, no_correction=False) -> np.array:
        """

        Returns: The single-sided power spectral density of the spectrum.

        """
        positive_content = self.positive_content()
        weights = np.full(len(positive_content), fill_value=2)
        weights[0], weights[-1] = 1, 1
        if no_correction:
            correction_factor = 1
        else:
            correction_factor = self.energy_window_correction() if fixed_energy else self.amplitude_window_correction()
        return np.abs((correction_factor * weights / self.duration()) * np.conj(positive_content) * positive_content)

    def sxx(self, fixed_energy=True) -> np.array:
        """

        Returns: The double-sided power spectral density of the spectrum.

        """
        correction_factor = self.energy_window_correction() if fixed_energy else self.amplitude_window_correction()
        return np.abs((correction_factor / self.duration()) * np.conj(self.data['amplitude']) * self.data['amplitude'])


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

    return Spectrum(spectrum, f_res)


def timeseries_to_spectrum(timeseries: Timeseries, window=None) -> Spectrum:
    """
    Helper function to create the corresponding spectrum from a timeseries class.

    Args:
        timeseries: The timeseries in question.

    Returns: The corresponding Spectrum

    """
    if window is None:
        window = np.ones(len(timeseries.data['amplitude']))

    spectrum = fft.fft(timeseries.data['amplitude'] * window / timeseries.sample_rate)
    frequency_resolution = 1 / timeseries.duration()
    return Spectrum(spectrum, frequency_resolution, window=window)


def spectrum_to_timeseries(spectrum: Spectrum) -> Timeseries:
    """
    Helper function to create the corresponding timeseries from a Spectrum class.

    Args:
        spectrum: The spectrum in question.

    Returns: the corresponding Timeseries

    """
    amplitude = fft.ifft(spectrum.data['amplitude'] * spectrum.num_samples() * spectrum.f_res)
    sample_rate = spectrum.sample_rate()
    return Timeseries(amplitude, sample_rate)


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
        frequency -= (np.max(frequency) - self.spectrum.f_res) / 2

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
            frequency -= (np.max(frequency) - self.spectrum.f_res) / 2
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

    def plot_gxx(self,
                 x_label='frequency (Hz)',
                 y_label=r'$G_{xx}\ (\frac{V^{2}}{Hz})$',
                 title='',
                 y_lim=None,
                 x_lim=None,
                 y_log=False,
                 filename=None,
                 text=None,
                 **kwargs):
        """
        This function will plot the single-sided power spectral density of the Spectrum.

        Args:
            x_label: Desired label for the x axis (frequency)
            y_label: Desired label for the y axis (Gxx)
            title: Desired title for the plot
            y_lim: A manual override to set the y axis limits
            x_lim: A manual override to set the x axis limits
            y_log: A manual switch to plot semilogy axis
            filename: If provided, will save the figure to the filename path.
            text: A string of text to display on the plot.

        Returns: a Tuple of (axis, figure)

        """

        f = self.spectrum.positive_frequencies()
        y = self.spectrum.gxx()

        fig = plt.figure()

        psd_plot = sns.lineplot(x=f, y=y, **kwargs)
        psd_plot.set(xlabel=x_label, ylabel=y_label, title=title)
        set_minor_gridlines(psd_plot)

        if y_lim is not None:
            plt.ylim(y_lim)

        if x_lim is not None:
            plt.xlim(x_lim)

        if y_log:
            plt.semilogy()

        if text is not None:
            psd_plot.text(0.95, 0.95, text,
                          transform=psd_plot.transAxes,
                          fontsize=14,
                          verticalalignment='top',
                          horizontalalignment='right',
                          bbox=self.TEXTBOX_PROPS)

        plt.tight_layout()

        if filename:
            fig.savefig(filename)

        return fig, psd_plot

    def plot_root_gxx(self,
                      x_label='frequency (Hz)',
                      y_label=r'$sqrt{PSD}\ (\frac{V}{\sqrt{Hz}})$',
                      title='',
                      y_lim=None,
                      x_lim=None,
                      y_log=False,
                      filename=None,
                      **kwargs):
        """
        This function will plot the single-sided power spectral density of the Spectrum.

        Args:
            x_label: Desired label for the x axis (frequency)
            y_label: Desired label for the y axis (Gxx)
            title: Desired title for the plot
            y_lim: A manual override to set the y axis limits
            x_lim: A manual override to set the x axis limits
            y_log: A manual switch to plot semilogy axis
            filename: If provided, will save the figure to the filename path.

        Returns: a Tuple of (axis, figure)

        """

        f = self.spectrum.positive_frequencies()
        y = np.sqrt(self.spectrum.gxx())

        fig = plt.figure()

        psd_plot = sns.lineplot(x=f, y=y, **kwargs)
        psd_plot.set(xlabel=x_label, ylabel=y_label, title=title)
        set_minor_gridlines(psd_plot)

        if y_lim is not None:
            plt.ylim(y_lim)

        if x_lim is not None:
            plt.xlim(x_lim)

        if y_log:
            plt.semilogy()

        plt.tight_layout()

        if filename:
            fig.savefig(filename)

        return fig, psd_plot

    def plot_db_root_gxx(self,
                         x_label='frequency (Hz)',
                         y_label=r'$sqrt{PSD}\ (dB re \mu \frac{V}{\sqrt{Hz}})$',
                         title='',
                         y_lim=None,
                         x_lim=None,
                         y_log=False,
                         filename=None,
                         reference=1e-6,
                         **kwargs):
        """
        This function will plot the single-sided power spectral density of the Spectrum.

        Args:
            x_label: Desired label for the x axis (frequency)
            y_label: Desired label for the y axis (Gxx)
            title: Desired title for the plot
            y_lim: A manual override to set the y axis limits
            x_lim: A manual override to set the x axis limits
            y_log: A manual switch to plot semilogy axis
            filename: If provided, will save the figure to the filename path.
            reference: Value to refer the dBs to.

        Returns: a Tuple of (axis, figure)

        """

        f = self.spectrum.positive_frequencies()
        y = 20.*np.log10(np.sqrt(self.spectrum.gxx()) / reference)

        fig = plt.figure()

        psd_plot = sns.lineplot(x=f, y=y, **kwargs)
        psd_plot.set(xlabel=x_label, ylabel=y_label, title=title)
        set_minor_gridlines(psd_plot)

        if y_lim is not None:
            plt.ylim(y_lim)

        if x_lim is not None:
            plt.xlim(x_lim)

        if y_log:
            plt.semilogy()

        plt.tight_layout()

        if filename:
            fig.savefig(filename)

        return fig, psd_plot


class Spectrogram:

    def __init__(self,
                 amplitudes: np.array,
                 f_res: float,
                 t_res: float):
        """

        Args:
            amplitudes: must be a complex array of shape (nBins, nRecords)
                        where bins = frequency bins, and records = time records.
                        This assumes the positive content of the spectrum only,
                        with the negative content calculable using the complex conjugate.
            f_res: The frequency resolution (Hz) of axis 0
            t_res: The time resolution (s) of axis 1
        """

        self.data = amplitudes
        self.f_res = f_res
        self.t_res = t_res
        self.start_time = 0
        self.end_time = self.data.shape[1] * self.t_res + self.start_time

    def positive_frequency_axis(self):
        return np.linspace(0, self.data.shape[0] * self.f_res / 2 + 1, int(self.data.shape[0] / 2 + 1), endpoint=False)

    def frequency_axis(self):
        return np.concatenate(self.positive_frequency_axis(), -1 * self.positive_frequency_axis()[::-1][1:-1])

    def time_axis(self):
        return np.linspace(self.start_time, self.end_time, self.data.shape[1], endpoint=True)

    def gxx(self, window=None):
        if window is not None:
            correction = energy_correction(window, len(self.positive_frequency_axis()) * 2)
        else:
            correction = 1

        positive_data = self.data[:int(self.data.shape[0] // 2 + 1), :]
        record_duration = 1 / self.f_res
        weights = np.full_like(positive_data, fill_value=2, dtype=float)
        weights[0, :], weights[-1, :] = 1, 1
        return ((weights / record_duration) * np.conj(positive_data) * positive_data).real * correction

    def rms(self, window=None):
        gxx = self.gxx(window=window)
        return np.sum(gxx, axis=1) / gxx.shape[1]

    def rms_v_time(self, window=None):
        gxx = self.gxx(window=window)
        return np.sqrt(np.sum(gxx*self.f_res, axis=0))


def energy_correction(window, n_samples):
    window = windows.get_window(window, n_samples)
    return len(window) / np.sum(window ** 2)


def create_spectrogram_image(spectrogram: Spectrogram, max_f=None, window=None):

    with sns.axes_style("ticks"):
        gxx = 10*np.log10(spectrogram.gxx(window=window))
        x_axis, y_axis = spectrogram.positive_frequency_axis(), spectrogram.time_axis()

        # Toss the 0 Hz bin:
        gxx = gxx[1:, :]
        x_axis = x_axis[1:]

        xmin, xmax, ymin, ymax = x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]

        # Find closest frequency to max_f
        if max_f is None:
            max_f = xmax * 0.8

        max_f = x_axis[x_axis <= max_f][-1]
        end_index = int(np.argwhere(x_axis == max_f))

        # Toss last 80%
        gxx = gxx[:end_index, :]
        x_axis = x_axis[:end_index]

        xmin, xmax, ymin, ymax = x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]
        fig, ax = plt.subplots(1)
        im = ax.imshow(np.transpose(gxx), extent=[xmin, xmax, ymax, ymin], cmap='plasma')
        ax.grid(False, 'both')
        ax.set_aspect(xmax / ymax)
        ax.set_ylabel('time (s)')
        ax.set_xlabel('frequency (Hz)')

        ax.set_aspect(xmax / ymax)

        cbar = plt.colorbar(im)
        cbar.ax.tick_params(axis='y', direction='in')
        cbar.set_label(r'PSD ($\frac{V^2}{Hz}$)', rotation=270, labelpad=20)

        plt.tight_layout()

    return fig, ax


def timeseries_to_spectrogram(timeseries: Timeseries,
                              fft_size: int = 1024,
                              n_samples: int = 1024,
                              n_records: int = None,
                              synchronization_offset: int = 0,
                              overlap: float = 0.0,
                              window: str = None):
    """
    # TODO: Introduce overlap + windowing

    Args:
        n_records:
        synchronization_offset:
        timeseries:
        fft_size:
        n_samples:
        overlap:
        window: str

    Returns:

    """

    if overlap >= 1:
        raise ValueError("Overlap percentage must be less than 100%")
    if overlap < 0:
        raise ValueError("Overlap percentage must be equal or greater than 0%")

    total_samples = n_samples + synchronization_offset
    overlap_samples = n_samples * (1-overlap)

    if n_records is None:
        n_records = int(((len(timeseries.data) - total_samples) // overlap_samples)) + 1

    start_samples = [int(i * total_samples * (1 - overlap)) for i in range(n_records)]
    time_data = np.zeros((fft_size, n_records), dtype=float)

    # Reshape the timeseries data into bins of n_time_samples:
    input_data = np.reshape(np.array(list(zip(*(timeseries.data['amplitude'][i:i+total_samples] for i in start_samples)))),
                            (total_samples, n_records), order='F')
    # input_data = np.reshape(timeseries.data['amplitude'][:(n_records * total_samples)], (total_samples, n_records),
    #                         order='F')
    time_data[:n_samples, :] = input_data[:n_samples, :]

    if window is None:
        window = np.ones(n_samples)
    else:
        window = windows.get_window(window, n_samples)

    # Apply window
    time_data *= np.transpose(np.tile(window, (time_data.shape[1], 1)))

    spectrogram_data = fft.fft(time_data / timeseries.sample_rate, fft_size, axis=0)
    t_res = timeseries.duration() / (int(((len(timeseries.data) - total_samples) // overlap_samples)) + 1)
    f_res = timeseries.sample_rate / fft_size

    return Spectrogram(spectrogram_data, f_res, t_res)
