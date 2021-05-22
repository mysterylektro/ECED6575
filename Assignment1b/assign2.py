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
# from Assignment1a.assign1 import set_minor_gridlines, plot_histogram, normal_probability_plot, setup_logging

# Setup output directories.
PLOT_DIR = os.sep.join([os.path.dirname(__file__), 'plots'])
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Setup common graphics properties.
TEXTBOX_PROPS = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.75}


def generate_plots(phase=(random_phase, 'Random Phase'),
                   magnitude=(1, 'Flat Magnitude'),
                   problem=1):
    timeseries_title = f'Real Timeseries\n{phase[1]}, {magnitude[1]}'
    spectrum_title = f'Spectrum\n{phase[1]}, {magnitude[1]}'

    data_16 = generate_spectrum(n=16, phase=phase[0], magnitude=magnitude[0])
    data = generate_spectrum(phase=phase[0], magnitude=magnitude[0])

    phase_filename = phase[1].lower().replace(" ", "_")
    magnitude_filename = magnitude[1].lower().replace(" ", "_")

    # Create spectrum plot for N = 16

    fig = plt.figure()
    spectrum_data = data_16[['frequency', 'real_spectrum', 'imaginary_spectrum']].melt('frequency', var_name='Spectrum', value_name='vals')
    spectrum_plot = sns.lineplot(data=spectrum_data,
                                 x='frequency',
                                 y='vals',
                                 hue='Spectrum')
    spectrum_plot.set(xlabel='frequency bin', ylabel='amplitude', title=spectrum_title)
    for t, l in zip(spectrum_plot.get_legend().texts, ['Real', 'Imaginary']):
        t.set_text(l)
    spectrum_plot.get_legend().get_frame().update(TEXTBOX_PROPS)

    fig.savefig(f'{PLOT_DIR}{os.sep}problem_{problem}_spectrum_n16_{phase_filename}_{magnitude_filename}.png')
    plt.close(fig)

    # Create timeseries plot
    fig = plt.figure()
    timeseries_plot = sns.lineplot(data=data,
                                   x='time',
                                   y='real_timeseries')
    timeseries_plot.set(xlabel='time (s)', ylabel='amplitude (V)', title=timeseries_title)
    set_minor_gridlines(timeseries_plot)
    mean, var, stddev = np.mean(data['real_timeseries']), np.var(data['real_timeseries']), np.std(data['real_timeseries'])
    stddev_ratio = np.max(data['real_timeseries']) / stddev

    text_string = '\n'.join([r'$\mu=%.2f\ V$' % mean,
                             r'$\sigma=%.2f\ V$' % stddev,
                             r'$\sigma^2=%.2f\ V^2$' % var,
                             r'$\frac{V_{max}}{\sigma}=%.2f$' % stddev_ratio])

    timeseries_plot.text(0.05, 0.95, text_string,
                         transform=timeseries_plot.transAxes,
                         fontsize=14, verticalalignment='top', bbox=TEXTBOX_PROPS)

    fig.savefig(f'{PLOT_DIR}{os.sep}problem_{problem}_timeseries_{phase_filename}_{magnitude_filename}.png')
    plt.close(fig)

    fig, histogram_plot = plot_histogram(data['real_timeseries'],
                                         x_label='amplitude (V)',
                                         y_label='probability density function',
                                         title=timeseries_title,
                                         bin_model='scott')

    fig.savefig(f'{PLOT_DIR}{os.sep}problem_{problem}_histogram_{phase_filename}_{magnitude_filename}.png')
    plt.close(fig)

    fig, prob_plot, stats = normal_probability_plot(data['real_timeseries'], stddev=stddev)
    prob_plot.set(title=timeseries_title, xlabel='amplitude (V)')
    fig.savefig(f'{PLOT_DIR}{os.sep}problem_{problem}_normal_probability_{phase_filename}_{magnitude_filename}.png')
    plt.close(fig)

    return data


def assign2():

    logger = setup_logging(assignment_name='1b')

    # Set random seed for reproducibility
    np.random.seed(100)

    prob1_data = generate_plots(phase=(random_phase, 'Random Phase'), magnitude=(1, 'Flat Magnitude'), problem=1)
    prob2_data = generate_plots(phase=(0, 'Zero Phase'), magnitude=(1, 'Flat Magnitude'), problem=2)
    prob3_data = generate_plots(phase=(random_phase, 'Random Phase'), magnitude=(pink, 'Pink Noise'), problem=3)


if __name__ == '__main__':
    assign2()
