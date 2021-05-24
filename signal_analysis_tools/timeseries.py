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


class Timeseries:

    def __init__(self,
                 time_axis: np.array,
                 amplitude: np.array,
                 sample_rate: float,
                 name: str = ''):

        self.data = np.array([*zip(time_axis, amplitude)], dtype=[('time', time_axis.dtype),
                                                                  ('amplitude', amplitude.dtype)])
        self.sample_rate = sample_rate
        self.name = name

    def set_sample_rate(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.data['time'] = np.arange(self.num_samples()) / self.sample_rate

    def duration(self):
        return self.num_samples() / self.sample_rate

    def num_samples(self):
        return len(self.data['time'])

    def time(self):
        return self.data['time']

    def amplitude(self, only_real=True):
        if only_real and np.iscomplexobj(self.data['amplitude']):
            return self.data['amplitude'].real
        else:
            return self.data['amplitude']

    def mean_square(self):
        return np.mean(self.amplitude() ** 2)

    def rms(self):
        return np.sqrt(np.mean(self.amplitude() ** 2))

    def std(self):
        return np.std(self.amplitude())

    def var(self):
        return np.var(self.amplitude())

    def max(self):
        return np.max(np.abs(self.amplitude()))

    def stddev_ratio(self):
        return self.max() / self.std()

    def time_resolution(self):
        return 1 / self.sample_rate

    def mean(self):
        return np.mean(self.amplitude())

    def get_sample_rate(self):
        return self.sample_rate

    def subset(self, start_time, end_time, zero_mean=False):
        start_idx, end_idx = np.searchsorted(self.data['time'], (start_time, end_time), side="left")
        data = self.data['amplitude'][start_idx:end_idx]
        if zero_mean:
            data -= np.mean(data)
        time_axis = np.arange(len(data)) / self.sample_rate + self.data['time'][start_idx]
        return Timeseries(time_axis, data, self.sample_rate)


class TimeseriesPlotter:
    # Setup common graphics properties.
    TEXTBOX_PROPS = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.75}

    # Setup plotting styles.
    sns.plotting_context("paper")
    sns.set_style("darkgrid")

    # Histogram bin-width models.
    BIN_MODELS = {'fd': 'Freedman-Diaconis choice',
                  'doane': 'Doane\'s formula',
                  'scott': 'Scott\'s normal reference rule',
                  'stone': 'Stone\'s formula',
                  'rice': 'Rice rule',
                  'sturges': 'Sturge\'s formula',
                  'sqrt': 'Square-root choice'}

    STATS_TEXT = {'mean': r'$\mu=%.2f\ V$',
                  'num_samples': r'$N=%d$',
                  'sample_rate': r'$f_{s}=%.2f\ Hz$',
                  'std': r'$\sigma=%.2f\ V$',
                  'var': r'$\sigma^2=%.2f\ V^2$',
                  'max': r'$V_{max}=%.2f\ V$',
                  'std_ratio': r'$\frac{V_{max}}{\sigma}=%.2f$'}

    def __init__(self, timeseries: Timeseries = None):
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
        self.timeseries = timeseries
        self.STATS_FUNC = {'mean': self.timeseries.mean,
                           'num_samples': self.timeseries.num_samples,
                           'sample_rate': self.timeseries.get_sample_rate,
                           'std': self.timeseries.std,
                           'var': self.timeseries.var,
                           'max': self.timeseries.max,
                           'std_ratio': self.timeseries.stddev_ratio}

    def plot_time_domain(self,
                         x_label='',
                         y_label='',
                         title='',
                         y_lim=None,
                         x_lim=None,
                         stats=None,
                         filename=None):

        x, y = self.timeseries.time(), self.timeseries.amplitude()
        fig = plt.figure()
        line_plot = sns.lineplot(x=x, y=y)
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

        if filename:
            fig.savefig(filename)

        return fig, line_plot

    def plot_histogram(self,
                       bin_model='sqrt',
                       x_label='',
                       y_label='probability density function',
                       title='',
                       y_lim=None,
                       x_lim=None,
                       filename=None):

        fig = plt.figure()
        histogram_plot = sns.histplot(data=self.timeseries.amplitude(), stat='density', bins=bin_model)
        histogram_plot.set(xlabel=x_label, ylabel=y_label, title=title)
        set_minor_gridlines(histogram_plot)

        if y_lim is not None:
            plt.ylim(y_lim)

        if x_lim is not None:
            plt.xlim(x_lim)

        if filename:
            fig.savefig(filename)

        return fig, histogram_plot

    def plot_normal_probability(self,
                                x_label='',
                                y_label='probability',
                                title='',
                                stats=True,
                                filename=None):

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

        if filename:
            fig.savefig(filename)

        return fig, ax, (slope, intercept, r)


def timeseries_from_csv(filename, *args, **kwargs):
    data = np.genfromtxt(filename, usecols=[0, 1], names=['time', 'amplitude'], delimiter=',')
    sample_rate = 1 / np.mean(data['time'][1:] - data['time'][0:-1])
    return Timeseries(data['time'], data['amplitude'], sample_rate)


def playback_timeseries(timeseries: Timeseries, sample_rate=None):
    duration = timeseries.duration()
    if sample_rate:
        data = resample(timeseries.amplitude(), int(np.floor(duration * sample_rate)))
    else:
        sample_rate = timeseries.sample_rate
        data = timeseries.amplitude()

    normalized_data = data / np.max(data)

    # Play sound in speakers.
    sd.play(normalized_data, sample_rate)
    sd.sleep(int(duration * 1000))
    sd.stop()


def play_and_record_timeseries(timeseries: Timeseries, sample_rate=None):

    if sample_rate:
        data = resample(timeseries.amplitude(), int(np.floor(timeseries.duration() * sample_rate)))
    else:
        sample_rate = timeseries.sample_rate
        data = timeseries.amplitude()

    normalized_data = (data / np.max(data)).astype(np.float32)

    device_out = 5  # Speakers
    device_in = 1  # Headset microphone
    with sd.Stream(device=(device_in, device_out), channels=1,
                   samplerate=sample_rate, blocksize=len(normalized_data)) as stream:
        stream.write(normalized_data)
        recorded_data = stream.read(stream.read_available)

    t = np.arange(len(recorded_data[0])) / sample_rate
    return Timeseries(t, recorded_data[0], sample_rate)


def record_timeseries(duration=1.0, prompt=True, sample_rate=None):
    if prompt:
        input(f"Press enter when ready to record for {duration} seconds at {sample_rate} samples per second...")
    device_in = 1
    with sd.InputStream(device=device_in, channels=1, samplerate=sample_rate, blocksize=int(duration * sample_rate)) as stream:
        sd.sleep(int(duration * 1000))
        recorded_data = stream.read(stream.read_available)
    time_axis = np.arange(0, len(recorded_data[0])) / sample_rate
    return Timeseries(time_axis, recorded_data[0], sample_rate)
