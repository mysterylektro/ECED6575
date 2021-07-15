"""
    -*- coding: utf-8 -*-
    Time    : 2021-06-26 11:48 a.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: exam.py
    
    {Description}
    -------------
    
"""

from signal_analysis_tools.utility_import import *


# Setup output directories.
PLOT_DIR = os.sep.join([os.path.dirname(__file__), 'plots'])
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Setup plotting styles.
sns.plotting_context("paper")
sns.set_style("darkgrid")


def problem_1():
    # a)

    cal_bravo = 2.7  # Pa / V
    cal_alpha = -1.8  # Pa / V
    filenames = ['./rcvr_alpha_run 002.wav',
                 './rcvr_alpha_run 001.wav',
                 './rcvr_alpha_run 003.wav',
                 './rcvr_bravo_run 002.wav',
                 './rcvr_bravo_run 001.wav',
                 './rcvr_bravo_run 003.wav']

    alpha_fig, alpha_ax = plt.subplots(1, 1)
    bravo_fig, bravo_ax = plt.subplots(1, 1)

    bravo_ax.set_ylim(-2, 2)
    alpha_ax.set_ylim(-2, 2)

    alpha_ax.set_ylabel('amplitude (Pa)')
    bravo_ax.set_ylabel('amplitude (Pa)')
    alpha_ax.set_xlabel('time (s)')
    bravo_ax.set_xlabel('time (s)')

    timeseries_data = {}

    for filename in filenames:
        run = filename[-7:-4]
        reader = WaveFileReader()
        reader.set_filename(filename)
        ts = reader.full_wavefile()
        if 'alpha' in filename:
            cal_factor = cal_alpha
            ax = alpha_ax
            name = 'alpha ' + run
        else:
            cal_factor = cal_bravo
            ax = bravo_ax
            name = 'bravo ' + run

        if 'run 002' in filename:
            cal_factor *= 3.16

        if 'run 003' in filename:
            cal_factor /= 3.16

        ts = Timeseries(ts.amplitude() * cal_factor, sample_rate=ts.sample_rate)
        timeseries_data[name] = ts
        ax.plot(ts.data['time'], ts.amplitude(), label=name, linewidth=0.25)

    alpha_ax.legend()
    bravo_ax.legend()

    alpha_fig.savefig('./problem1_a_alpha.png')
    bravo_fig.savefig('./problem1_a_bravo.png')

    # b)

    # i)
    ts = timeseries_data['bravo 001']
    ts = Timeseries(ts.amplitude()[:2**15], sample_rate=ts.sample_rate)

    print(f'i) {2 ** 15 / ts.sample_rate} seconds')

    spectrum = timeseries_to_spectrum(ts)
    spectrum_plotter = SpectrumPlotter()
    spectrum_plotter.set_spectrum(spectrum)
    fig, p = spectrum_plotter.plot_gxx(y_label=r'$G_{bb}\ (\frac{Pa^{2}}{Hz})$', x_lim=(0, 500), y_log=True)
    fig.savefig('problem1b_ii.png')

    print(f'iii) RMS of timeseries: {ts.rms()} Pa')
    print(f'iii) RMS of Gbb: {np.sqrt(np.sum(spectrum.gxx() * spectrum.f_res))} Pa')
    print(f'iv) Gbb at zero-frequency: {spectrum.gxx()[0]} Pa^2/Hz')
    print(f'v) dF for this record: {spectrum.f_res} Hz')
    print(f'vi) Gbb for first non-zero frequency: {spectrum.gxx()[1]} Pa^2/Hz')
    peak_index = np.argmax(spectrum.gxx())
    print(f'vii) Peak value for Gbb is: {np.sqrt(spectrum.gxx()[peak_index]*spectrum.f_res)} Pa RMS. Frequency value is: {peak_index * spectrum.f_res} Hz')

    # c)
    alpha_ts = timeseries_data['alpha 001']
    bravo_ts = timeseries_data['bravo 001']

    n_records = alpha_ts.num_samples() // 4096
    alpha_spectrogram = timeseries_to_spectrogram(alpha_ts,
                                            fft_size = 4096,
                                            n_samples = 4096,
                                            n_records = n_records,
                                            synchronization_offset = 0)

    bravo_spectrogram = timeseries_to_spectrogram(bravo_ts,
                                            fft_size = 4096,
                                            n_samples = 4096,
                                            n_records = n_records,
                                            synchronization_offset = 0)

    print(f'c) i) Each record is {4096 / alpha_ts.sample_rate} seconds')
    print(f'c) ii) Each frequency domain has a dF of {alpha_spectrogram.f_res} Hz')
    print(f'c) iii) Each Max number of 4096-point records is {n_records}')
    print(f'c) iv) total duration would need to be {1/5} seconds, = {1/5 * alpha_ts.sample_rate} samples per FFT')

    mean_gbb = np.mean(bravo_spectrogram.gxx(), axis=1)
    f = bravo_spectrogram.positive_frequency_axis()
    fig = plt.figure()
    psd_plot = sns.lineplot(x=f, y=mean_gbb)
    psd_plot.set(xlabel='frequency (Hz)', ylabel=r'$Mean\ G_{bb}\ (\frac{Pa^{2}}{Hz})$')
    set_minor_gridlines(psd_plot)
    plt.xlim(0, 500)
    plt.semilogy()
    fig.savefig('problem1c_v.png')

    print(f'c) vi) Mean Gbb at zero-frequency: {mean_gbb[0]} Pa^2/Hz')
    peak_index = np.argmax(mean_gbb)
    print(f'c) vii) Peak value for mean Gbb is: {np.sqrt(mean_gbb[peak_index] * bravo_spectrogram.f_res)} Pa RMS.\n'
          f'Frequency value is: {peak_index * bravo_spectrogram.f_res} Hz')

    fig = plt.figure()

    for run in ['bravo 001', 'bravo 002', 'bravo 003']:
        ts = timeseries_data[run]
        n_records = ts.num_samples() // 4096
        spectrogram = timeseries_to_spectrogram(ts,
                                                fft_size=4096,
                                                n_samples=4096,
                                                n_records=n_records,
                                                synchronization_offset=0)
        f = spectrogram.positive_frequency_axis()
        mean_gbb = np.mean(spectrogram.gxx(), axis=1)
        psd_plot = sns.lineplot(x=f, y=mean_gbb, label=run)

    set_minor_gridlines(psd_plot)
    plt.xlabel('frequency (Hz)')
    plt.ylabel(r'$Mean\ G_{bb}\ (\frac{Pa^{2}}{Hz})$')
    plt.legend()
    plt.xlim(0, 500)
    plt.semilogy()
    fig.savefig(f'problem1d.png')


def problem_2():
    alpha = {'a)': 3.41e-3,  # dB / km
             'b)': 5.7e-3,  # dB / km
             'c)': 0.849}  # dB / km
    d = 10.  # km
    for q, a in alpha.items():
        transmission_loss = a * d + 20*np.log10(d*1000)
        print(q + f' TL = {transmission_loss} dB')

    print(f'Q2: d) Low frequency = lower attenutaion, high frequency = more attenuation.')


def problem_3():

    f = 30000  # Hz
    SL = 215  # dB re 1 uPa
    d = 218  # m
    alpha = 6.14 * 0.001   # dB/m
    EL = 98.79  # dB re 1 uPa

    # Assume spherical spreading
    TL = alpha * d + 20*np.log10(d)

    # EL = SNR + NL
    # EL = SL + TS - 2TL
    TS = EL - SL + 2*TL

    print(f'Target Strength of Mine: {TS} dB')
    # Scattering cross-section:
    cross_section = 10 ** (TS / 10) * 4 * np.pi
    print(f'Target cross section: {cross_section} m^2')
    effective_radius = np.sqrt(cross_section / np.pi)
    print(f'Effective mine radius: {effective_radius} m')
    print(f'One way transmission loss: {TL} dB')


if __name__ == '__main__':
    problems = [problem_1, problem_2, problem_3]
    for problem in problems:
        problem()
