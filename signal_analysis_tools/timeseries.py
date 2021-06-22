"""
    -*- coding: utf-8 -*-
    Time    : 2021-05-21 8:30 a.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: timeseries.py
    
    {Description}
    -------------
    
"""

import numpy as np
import seaborn as sns
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.stats import probplot, norm
from scipy.signal import resample
from signal_analysis_tools.utilities import set_minor_gridlines

SPEAKERS = 5  # Speakers identified using sd.query_devices()
HEADSET_MICROPHONE = 1  # Headset microphone identified using sd.query_devices()


class Timeseries:
    """
    Base class for storing Timeseries data. This class provides built-in functions for
    computing statistics on the underlying data.
    """

    def __init__(self,
                 samples: np.array,
                 sample_rate: float,
                 time_offset: float = 0.0,
                 units: str = 'V',
                 name: str = ''):
        """
        Initialize a Timeseries class. If a sample rate is provided, the time axis will be recomputed based on that
        value. If the sample rate is not provided, it will be computed based on the provided time axis.

        Args:
            samples: measured values for each sample (in sampled units)
            sample_rate: how often the samples are sampled (in Hz)
            name: a descriptor for the timeseries.
        """

        self.sample_rate = sample_rate
        self.time_offset = time_offset
        self.units = units
        self.name = name

        time_axis = np.arange(len(samples)) / self.sample_rate + self.time_offset
        self.data = np.array([*zip(time_axis, samples)], dtype=[('time', time_axis.dtype),
                                                                ('amplitude', samples.dtype)])

    def update_time_axis(self):
        self.data['time'] = np.arange(self.num_samples()) / self.sample_rate + self.time_offset

    def set_sample_rate(self, sample_rate: float):
        """
        Sets the sample rate and recomputes the time axis.

        Args:
            sample_rate: how often the samples are sampled (in Hz)

        Returns:

        """
        self.sample_rate = sample_rate
        self.update_time_axis()

    def set_time_offset(self, time_offset: float):
        self.time_offset = time_offset
        self.update_time_axis()

    def duration(self) -> float:
        """

        Returns: the duration of the timeseries (in seconds)

        """
        return self.num_samples() / self.sample_rate

    def num_samples(self) -> int:
        """

        Returns: The number of recorded samples

        """
        return len(self.data['amplitude'])

    def time(self) -> np.array:
        """

        Returns: An array of the time values (in seconds) for each sample.

        """
        return self.data['time']

    def amplitude(self, only_real=True) -> np.array:
        """

        Args:
            only_real: A boolean to return the real part of the timeseries if the measured amplitude is complex.

        Returns: the measured sample values (in units) for the timeseries.

        """
        if only_real and np.iscomplexobj(self.data['amplitude']):
            return self.data['amplitude'].real
        else:
            return self.data['amplitude']

    def mean_square(self) -> float:
        """

        Returns: The mean of the square value (in units) of the measured amplitudes.

        """
        return np.mean(self.amplitude() ** 2)

    def rms(self) -> float:
        """

        Returns: The root-mean square value (in units) of the measured amplitudes.

        """
        return np.abs(np.sqrt(np.mean(self.amplitude() ** 2)))

    def std(self) -> float:
        """

        Returns: The standard deviation (in units) of the measured amplitudes.

        """
        return np.std(self.amplitude())

    def var(self) -> float:
        """

        Returns: The variance (in units) of the measured amplitudes.

        """
        return np.var(self.amplitude())

    def max(self) -> float:
        """

        Returns: The maximum amplitude (in units) of the measured amplitudes.

        """
        return np.max(np.abs(self.amplitude()))

    def stddev_ratio(self) -> float:
        """

        Returns: The ratio of the maximum amplitude and standard deviation of the measured amplitudes.

        """
        return self.max() / self.std()

    def time_resolution(self) -> float:
        """

        Returns: The time resolution (in seconds) of the measured amplitudes.

        """
        return 1 / self.sample_rate

    def mean(self) -> float:
        """

        Returns: The mean value (in units) of the measured amplitudes.

        """
        return np.mean(self.amplitude())

    def get_sample_rate(self) -> float:
        """

        Returns: The sample rate (in Hz) of the measured amplitudes.

        """
        return self.sample_rate

    def subset(self, start_time, end_time, zero_mean=False) -> 'Timeseries':
        """
        This function returns a new timeseries that is a subset of itself.

        Args:
            start_time: The time (in seconds) to begin the subset.
            end_time: The time (in seconds) to end the subset.
            zero_mean: A boolean value to indicate whether or not to remove any DC offset in the subset.

        Returns: Timeseries class containing the subset data.

        """
        start_idx, end_idx = np.searchsorted(self.data['time'], (start_time, end_time), side="left")
        data = self.data['amplitude'][start_idx:end_idx]
        if zero_mean:
            data -= np.mean(data)

        return Timeseries(data, self.sample_rate)

    def zero_mean(self) -> 'Timeseries':
        return Timeseries(self.data['amplitude'] - self.mean(), self.sample_rate)

    def zero_pad(self, n, start=False, end=True):
        data = self.data['amplitude']
        zeroes = np.zeros(n, dtype=data.dtype)
        if start:
            data = np.concatenate((zeroes, data))
        if end:
            data = np.concatenate((data, zeroes))

        return Timeseries(data, self.sample_rate)


class TimeseriesPlotter:
    """
    This class is a helper class to generate assignment-quality timeseries plots.
    """

    # Setup common graphics properties.
    TEXTBOX_PROPS = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.75}

    # Setup plotting styles.
    sns.plotting_context("paper")
    sns.set_style("darkgrid")

    # Available histogram bin-width models.
    BIN_MODELS = {'fd': 'Freedman-Diaconis choice',
                  'doane': 'Doane\'s formula',
                  'scott': 'Scott\'s normal reference rule',
                  'stone': 'Stone\'s formula',
                  'rice': 'Rice rule',
                  'sturges': 'Sturge\'s formula',
                  'sqrt': 'Square-root choice'}

    # LaTeX strings for a variety of statistical values obtainable through the Timeseries class.
    STATS_TEXT = {'mean': r'$\mu=%.2f\ V$',
                  'num_samples': r'$N=%d$',
                  'sample_rate': r'$f_{s}=%.2f\ Hz$',
                  'std': r'$\sigma=%.2f\ V$',
                  'var': r'$\sigma^2=%.2f\ V^2$',
                  'max': r'$V_{max}=%.2f\ V$',
                  'std_ratio': r'$\frac{V_{max}}{\sigma}=%.2f$'}

    def __init__(self, timeseries: Timeseries = None):
        """
        Initialize the plotting class.

        Args:
            timeseries: a Timeseries class to plot values for.
        """
        self.timeseries = timeseries
        if timeseries is not None:
            self.STATS_FUNC = {'mean': self.timeseries.mean,
                               'num_samples': self.timeseries.num_samples,
                               'sample_rate': self.timeseries.get_sample_rate,
                               'std': self.timeseries.std,
                               'var': self.timeseries.var,
                               'max': self.timeseries.max,
                               'std_ratio': self.timeseries.stddev_ratio}

    def set_timeseries(self, timeseries: Timeseries):
        """

        Args:
            timeseries: set the internal Timeseries class for plotting.

        Returns:

        """
        self.timeseries = timeseries
        self.STATS_FUNC = {'mean': self.timeseries.mean,
                           'num_samples': self.timeseries.num_samples,
                           'sample_rate': self.timeseries.get_sample_rate,
                           'std': self.timeseries.std,
                           'var': self.timeseries.var,
                           'max': self.timeseries.max,
                           'std_ratio': self.timeseries.stddev_ratio}

    def plot_time_domain(self,
                         x_label='time (s)',
                         y_label='amplitude (V)',
                         title='',
                         y_lim=None,
                         x_lim=None,
                         stats=None,
                         filename=None,
                         **kwargs):
        """
        This function will generate a time domain plot and return the axis and figure.

        Args:
            x_label: Desired label for the x axis (time)
            y_label: Desired label for the y axis (amplitude)
            title: Desired title for the plot
            y_lim: A manual override to set the y axis limits
            x_lim: A manual override to set the x axis limits
            stats: A list of statistics keys to display on the plot
            filename: If provided, will save the figure to the filename path.

        Returns: a Tuple of (axis, figure)

        """
        x, y = self.timeseries.time(), self.timeseries.amplitude()
        fig = plt.figure()
        line_plot = sns.lineplot(x=x, y=y, **kwargs)
        line_plot.set(xlabel=x_label, ylabel=y_label, title=title)
        set_minor_gridlines(line_plot)

        if y_lim is not None:
            plt.ylim(y_lim)

        if x_lim is not None:
            plt.xlim(x_lim)

        if stats:
            text_string = '\n'.join([t % f() for t, f in zip([self.STATS_TEXT.get(s) for s in stats],
                                                             [self.STATS_FUNC.get(s) for s in stats])])
            line_plot.text(0.05, 0.95, text_string,
                           transform=line_plot.transAxes,
                           fontsize=14, verticalalignment='top', bbox=self.TEXTBOX_PROPS)

        plt.tight_layout()

        if filename:
            fig.savefig(filename)

        return fig, line_plot

    def plot_histogram(self,
                       bin_model='sqrt',
                       x_label='amplitude (V)',
                       y_label='probability density function',
                       title='',
                       y_lim=None,
                       x_lim=None,
                       filename=None):
        """
        This function will generate a histogram plot of the Timeseries distribution and return the axis and figure.

        Args:
            bin_model: A string indicating the type of bin model to use.
                       Accepted values are described in np.histogram_bin_edges
            x_label: Desired label for the x axis (amplitude)
            y_label: Desired label for the y axis (probability)
            title: Desired title for the plot
            y_lim: A manual override to set the y axis limits
            x_lim: A manual override to set the x axis limits
            filename: If provided, will save the figure to the filename path.

        Returns: a Tuple of (axis, figure)

        """

        fig = plt.figure()
        histogram_plot = sns.histplot(data=self.timeseries.amplitude(), stat='density', bins=bin_model)
        histogram_plot.set(xlabel=x_label, ylabel=y_label, title=title)
        set_minor_gridlines(histogram_plot)

        if y_lim is not None:
            plt.ylim(y_lim)

        if x_lim is not None:
            plt.xlim(x_lim)

        plt.tight_layout()

        if filename:
            fig.savefig(filename)

        return fig, histogram_plot

    def plot_normal_probability(self,
                                x_label='amplitude (V)',
                                y_label='probability',
                                title='',
                                stats=True,
                                filename=None):
        """
        This function will generate a normal probability plot of the Timeseries distribution
        and return the axis and figure.

        Args:
            x_label: Desired label for the x axis (time)
            y_label: Desired label for the y axis (time)
            title: Desired title for the plot
            stats: A boolean to indicate whether to display the statistics for the line of best fit.
            filename: If provided, will save the figure to the filename path.

        Returns: a Tuple of Tuples: ((axis, figure), (slope, intercept, r))

        """

        stddev = self.timeseries.std()
        (quantiles, values), (slope, intercept, r) = probplot(self.timeseries.amplitude(), dist='norm')

        fig, ax = plt.subplots()
        ax.plot(values, quantiles, '+b')
        ax.plot(quantiles * slope + intercept, quantiles, 'r')

        percentiles = np.array([0.01, 0.02, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.98, 0.99])
        ticks_quantile = [norm.ppf(p) for p in percentiles]
        plt.yticks(ticks_quantile, percentiles)

        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title)

        plt.ylim(norm.ppf(0.01), norm.ppf(0.99))
        plt.xlim(-3 * stddev, 3 * stddev)

        set_minor_gridlines(ax)

        if stats:
            text_string = '\n'.join([r'$m=%.2f\ V$' % slope,
                                     r'$b=%.2f\ V$' % intercept,
                                     r'$r=%.2f$' % r])
            ax.text(0.05, 0.95, text_string,
                    transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', bbox=self.TEXTBOX_PROPS)

        plt.tight_layout()

        if filename:
            fig.savefig(filename)

        return fig, ax, (slope, intercept, r)


def timeseries_from_csv(filename, *args, **kwargs) -> Timeseries:
    """
    Helper function to load a .CSV file into a Timeseries class

    Args:
        filename: path to the .CSV file
        *args: Additional arguments for the np.genfromtxt function
        **kwargs: Additional keyword arguments for the np.genfromtxt function

    Returns: A Timeseries class with the .CSV data loaded.

    """
    data = np.genfromtxt(filename, usecols=[0, 1], names=['time', 'amplitude'], delimiter=',', *args, **kwargs)
    sample_rate = 1 / np.mean(data['time'][1:] - data['time'][0:-1])
    return Timeseries(data['amplitude'], sample_rate=sample_rate)


def playback_timeseries(timeseries: Timeseries, sample_rate=None) -> None:
    """
    A helper function to playback a Timeseries class through the default sound output device.

    Args:
        timeseries: The Timeseries class to play back
        sample_rate: The sample rate to playback the timeseries data at.
                     If not specified, will use the native timeseries sample rate.

    Returns: None

    """
    duration = timeseries.duration()
    if sample_rate:
        data = resample(timeseries.amplitude(), int(np.floor(duration * sample_rate)))
    else:
        sample_rate = timeseries.sample_rate
        data = timeseries.amplitude()

    normalized_data = data / np.max(data)

    # Play sound in speakers.
    sd.play(normalized_data, sample_rate, blocking=True)


def play_and_record_timeseries(timeseries: Timeseries, sample_rate=None) -> Timeseries:
    """
    A helper function to playback and record a Timeseries class through the default sound input and output devices.

    Args:
        timeseries: The Timeseries class to play back
        sample_rate: The sample rate to playback the timeseries data at.
                     If not specified, will use the native timeseries sample rate.

    Returns: a Timeseries class containing the recorded data.

    """

    if sample_rate:
        data = resample(timeseries.amplitude(), int(np.floor(timeseries.duration() * sample_rate)))
    else:
        sample_rate = timeseries.sample_rate
        data = timeseries.amplitude()

    normalized_data = (data / np.max(data)).astype(np.float32)

    recorded_data = sd.playrec(normalized_data, samplerate=sample_rate, channels=1, blocking=True)
    return Timeseries(recorded_data, sample_rate)


def record_timeseries(duration: float = 1.0, prompt: bool = True, sample_rate: float = 44100) -> Timeseries:
    """
    A helper function to record timeseries data through the default sound input device.

    Args:
        duration: The length of time to record
        prompt: A boolean to prompt the user for a trigger to begin recording.
        sample_rate: The sample rate at which to record.

    Returns: a Timeseries class containing the recorded data.

    """
    if prompt:
        input(f"Press enter when ready to record for {duration} seconds at {sample_rate} samples per second...")

    recorded_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, blocking=True)

    return Timeseries(recorded_data, sample_rate)
