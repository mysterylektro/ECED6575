"""
    -*- coding: utf-8 -*-
    Time    : 2021-06-15 8:45 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: assign3a.py
    
    {Description}
    -------------
    
"""

from signal_analysis_tools.utility_import import *
from scipy.signal.windows.windows import flattop

# Setup output directories.
PLOT_DIR = os.sep.join([os.path.dirname(__file__), 'plots'])
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Setup plotting styles.
sns.plotting_context("paper")
sns.set_style("darkgrid")

logger = setup_logging('3a')


def problem_1():
    logger.log(logging.INFO, '\n---- Starting problem 1 ----\n')
    # Inputs
    transmit_area = 0.1 ** 2  # m^2
    power = 400.  # W
    frequency = 100000.  # Hz
    rho = 1000.  # kg/m^3
    c = 1500.  # m/s
    alpha = 4.14e-3  # Np / m
    target_distance = 30.  # m
    target_cross_section = 80e-4  # m^2
    scattering_loss_per_m = 0.1 / 30.  # %

    parameters = {'Transducer area (m^2)': transmit_area,
                  'Transmit power (W)': power,
                  'Pulse frequency (Hz)': frequency,
                  'Medium density (kg/m^3)': rho,
                  'Speed of sound (m/s)': c,
                  'Absorption coefficient (Np/m)': alpha,
                  'Target range (m)': target_distance,
                  'Target backscatter cross-section': target_cross_section,
                  'Scattering loss (%/m)': scattering_loss_per_m}

    logger.log(logging.INFO, '--- Input Parameters ---\n')
    logger.log(logging.INFO, '\n'.join([f'{key}: {val}' for key, val in parameters.items()]) + '\n')

    # Calculated inputs
    logger.log(logging.DEBUG, 'Computing wavelength and transmit directivity...')
    wavelength = c / frequency  # m
    transmit_directivity = 4 * np.pi * transmit_area / wavelength ** 2
    logger.log(logging.INFO, f'Wavelength (m): {wavelength}\nTransmit directivity factor: {transmit_directivity}')

    # Calculate acoustic intensity at target range
    logger.log(logging.DEBUG, 'Computing acoustic intensity incident on target...')
    scattering_loss = 1 - scattering_loss_per_m * target_distance
    absorption_loss = np.exp(-2 * alpha * target_distance)
    incident_acoustic_intensity = power * scattering_loss * absorption_loss * transmit_directivity / (
            4 * np.pi * target_distance ** 2)  # W/m^2
    logger.log(logging.INFO, f'Incident acoustic intensity (W/m^2): {incident_acoustic_intensity}')
    reflected_power = incident_acoustic_intensity * target_cross_section
    logger.log(logging.INFO, f'Reflected power (W): {reflected_power}')

    logger.log(logging.DEBUG, 'Computing received acoustic intensity from target...')
    received_acoustic_intensity = reflected_power * scattering_loss * absorption_loss / (
            4 * np.pi * target_distance ** 2)
    logger.log(logging.INFO, f'Received acoustic intensity (W/m^2): {received_acoustic_intensity}')

    logger.log(logging.DEBUG, 'Computing mean received pressure...')
    mean_received_pressure = np.sqrt(received_acoustic_intensity * rho * c)
    logger.log(logging.INFO, f'Received RMS mean pressure (Pa): {mean_received_pressure}')


def problem_2():
    logger.log(logging.INFO, '\n---- Starting problem 2 ----\n')

    SL = 200  # dB
    TL = 53  # dB
    NL = 71  # dB
    SNR = 10  # dB

    parameters = {'SL (dB)': SL,
                  'TL (dB)': TL,
                  'NL (dB)': NL,
                  'SNR (dB)': SNR}

    logger.log(logging.INFO, '--- Input Parameters ---\n')
    logger.log(logging.INFO, '\n'.join([f'{key}: {val}' for key, val in parameters.items()]) + '\n')

    logger.log(logging.DEBUG, 'Computing target strength...')
    TS = SNR - SL + 2 * TL + NL  # dB
    logger.log(logging.INFO, f'TS: {TS} dB')

    logger.log(logging.DEBUG, 'Computing cross-section...')
    cross_section = 10 ** (TS / 10) * 4 * np.pi
    logger.log(logging.INFO, f'Cross section: {cross_section} m^2')


def problem_3():
    logger.log(logging.INFO, '\n---- Starting problem 3 ----\n')

    freq = 100000.  # Hz
    efficiency = 0.75  # %
    power = 200  # W
    Dt = 100.
    range_resolution = 0.15  # m
    c = 1500.  # m/s
    SNR = 20  # dB
    NL = -130.  # dB re 1 W/m^2 / Hz
    absorption_coefficient = 0.02  # dB / m
    Dr = 200.
    r = 200.  # m
    BW = c / (2 * range_resolution)  # Bandwidth required in order to resolve 0.15 m

    parameters = {'Pulse centre frequency (Hz)': freq,
                  'Power efficiency': efficiency,
                  'Supplied power (W)': power,
                  'Transmit directivity factor': Dt,
                  'Pulse range resolution (m)': range_resolution,
                  'Speed of sound (m/s)': c,
                  'Required SNR (dB)': SNR,
                  'Ambient noise (dB re 1 W/m^2)': NL,
                  'Absorption coefficient (dB/m)': absorption_coefficient,
                  'Receiver directivity factor': Dr,
                  'Target range (m)': r,
                  'Pulse bandwidth (Hz)': BW}

    logger.log(logging.INFO, '--- Input Parameters ---\n')
    logger.log(logging.INFO, '\n'.join([f'{key}: {val}' for key, val in parameters.items()]) + '\n')

    logger.log(logging.DEBUG, 'Computing source level, total noise, and one-way transmission loss...')
    source_level = 10 * np.log10(power * efficiency * Dt / (4 * np.pi))  # dB re 1 W/m^2
    total_noise = NL + 10 * np.log10(BW) - 10 * np.log10(Dr)  # dB re 1 W/m^2
    transmission_loss = absorption_coefficient * r + 20 * np.log10(r)

    logger.log(logging.INFO, '\n' + '\n'.join([f'SL (dB): {source_level}',
                                               f'AN (dB): {total_noise}',
                                               f'TL (dB): {transmission_loss}']) + '\n')

    logger.log(logging.DEBUG, 'Computing target strength...')
    TS = SNR - source_level + total_noise + 2 * transmission_loss
    logger.log(logging.INFO, f'\nTS: {TS}\n')

    logger.log(logging.DEBUG, 'Computing minimum target size...')
    cross_section = 10 ** (TS / 10) * 4 * np.pi
    radius = np.sqrt(cross_section / np.pi)

    logger.log(logging.INFO, '\n' + '\n'.join([f'Cross section (m^2): {cross_section}',
                                               f'Radius (m): {radius}']))


def problem_4():
    logger.log(logging.INFO, '\n---- Starting problem 4 ----\n')

    filename = './Cal2_3V_06_sub2.wav'
    logger.log(logging.DEBUG, f'Reading in {filename}...')
    reader = WaveFileReader(filename=filename)
    cal_ts = reader.full_wavefile()

    logger.log(logging.DEBUG, f'Plotting {filename} as timeseries...')
    ts_plotter = TimeseriesPlotter(cal_ts)
    fig, p = ts_plotter.plot_time_domain(title='Assignment 3a, Problem 4\nCal2 3V 06 sub2 timeseries',
                                         filename=f'{PLOT_DIR}{os.sep}problem_4_timeseries_plot.png',
                                         linewidth=0.25,
                                         stats=['rms'])

    plt.close(fig)

    logger.log(logging.DEBUG, f'Applying flattop window...')
    fft_window = flattop(cal_ts.num_samples())
    windowed_cal_ts = Timeseries(cal_ts.amplitude() * fft_window, cal_ts.sample_rate)

    logger.log(logging.DEBUG, f'Plotting windowed {filename} timeseries...')
    ts_plotter.set_timeseries(windowed_cal_ts)
    fig, p = ts_plotter.plot_time_domain(title='Assignment 3a, Problem 4\nCal2 3V 06 sub2 windowed timeseries',
                                         filename=f'{PLOT_DIR}{os.sep}problem_4_windowed_timeseries_plot.png',
                                         linewidth=0.25,
                                         stats=['rms'])
    plt.close(fig)

    time_rms = cal_ts.rms()
    windowed_time_rms = windowed_cal_ts.rms()
    logger.log(logging.INFO, f'\nTimeseries RMS: {time_rms} (V)\nWindowed Timeseries RMS: {windowed_time_rms} (V)')

    """
    Adding the window to the spectrum will implicitly calculate the fixed-energy window correction
    and apply it when calculating the Gxx. The fixed-amplitude window correction is applied when the
    user explicitly asks for it via keyword.
    """

    logger.log(logging.DEBUG, f'Generating spectrum from windowed timeseries...')
    cal_spec = timeseries_to_spectrum(cal_ts, window=fft_window)
    freq_rms = np.sqrt(np.sum(cal_spec.gxx()) * cal_spec.f_res)

    logger.log(logging.INFO, f'Fixed-energy window correction: {np.sqrt(cal_spec.energy_window_correction())}')
    logger.log(logging.INFO, f'Fixed-amplitude window correction: {np.sqrt(cal_spec.amplitude_window_correction())}')
    logger.log(logging.INFO, f'Gxx (windowed) RMS: {freq_rms} (V)')

    amplitudes = np.sqrt(cal_spec.gxx(fixed_energy=False) * cal_spec.f_res)
    max_amplitude_index = np.argmax(amplitudes)
    max_amplitude, max_freq = amplitudes[max_amplitude_index], cal_spec.f_res * max_amplitude_index

    logger.log(logging.INFO, f'Gxx Peak RMS: {max_amplitude} (V)\nGxx Peak RMS Frequency: {max_freq} (Hz)')
    logger.log(logging.INFO, f'Calibration Factor (250 Hz): {30. / max_amplitude} (Pa/V)')

    logger.log(logging.DEBUG, f'Plotting spectrum from windowed timeseries...')
    spec_plotter = SpectrumPlotter(cal_spec)
    filename = f'{PLOT_DIR}{os.sep}problem_4_spectrum_plot_flattop_plot.png'
    text_string = '\n'.join([r'$G_{xx}\ V_{rms}\ =\ %.2f\ (V)$' % freq_rms,
                             r'$G_{xx}\ f_{peak}\ =\ %2.f\ (Hz)$' % max_freq,
                             r'$G_{xx}\ f_{peak\ rms} = %.2f\ (V)$' % max_amplitude])
    fig, p = spec_plotter.plot_gxx(y_log=True,
                                   text=text_string,
                                   filename=filename,
                                   linewidth=0.25)

    plt.close(fig)

    # Zoom in for detail
    logger.log(logging.DEBUG, f'Plotting zoomed in spectrum from windowed timeseries...')
    filename = f'{PLOT_DIR}{os.sep}problem_4_spectrum_plot_flattop_plot_zoom.png'
    fig, p = spec_plotter.plot_gxx(y_log=True,
                                   x_lim=(0, 1000),
                                   text=text_string,
                                   filename=filename,
                                   linewidth=1)

    plt.close(fig)


def problem_5():
    logger.log(logging.INFO, '\n---- Starting problem 4 ----\n')

    filename = './Boom_F1B2_6.wav'

    logger.log(logging.DEBUG, f'Reading in {filename}...')
    reader = WaveFileReader(filename=filename)
    boom_ts = reader.full_wavefile()

    logger.log(logging.DEBUG, f'Playing {filename} at 44100 Hz...')
    playback_timeseries(boom_ts, sample_rate=44100)

    # Negative calibration factor because B&K Hydrophone inverts pressure.
    logger.log(logging.DEBUG, f'Calibrating timeseries...')
    calibration_factor = -30 / 8477.332654980304  # Pa-rms per V-rms from Q4
    calibrated_boom_ts = Timeseries(boom_ts.amplitude() * calibration_factor, boom_ts.sample_rate)

    logger.log(logging.INFO, f'Maximum pressure observed (Pa): {calibrated_boom_ts.max()}')

    logger.log(logging.DEBUG, f'Plotting timeseries...')
    ts_plotter = TimeseriesPlotter(calibrated_boom_ts)
    fig, p = ts_plotter.plot_time_domain(title='Assignment 3a, Problem 5\nBoom F1B2 6 calibrated timeseries',
                                         y_label='amplitude (Pa)',
                                         text=r'$Pa_{max}=%.2f$ Pa' % calibrated_boom_ts.max(),
                                         filename=f'{PLOT_DIR}{os.sep}problem_5_timeseries_plot.png')

    plt.close(fig)

    # Find boom duration
    logger.log(logging.DEBUG, f'Computing timeseries derivative...')
    delta_ts = Timeseries(calibrated_boom_ts.amplitude()[1:] - calibrated_boom_ts.amplitude()[:-1],
                          calibrated_boom_ts.sample_rate)

    # Manual interrogation
    max_indices = [11722, 18288]
    boom_duration = (max_indices[1] - max_indices[0]) / delta_ts.sample_rate
    logger.log(logging.INFO, f'Sonic boom duration (s): {boom_duration}')

    logger.log(logging.DEBUG, f'Plotting timeseries derivative...')
    ts_plotter.set_timeseries(delta_ts)
    fig, p = ts_plotter.plot_time_domain(title='Assignment 3a, Problem 5\nBoom F1B2 6 calibrated timeseries derivative',
                                         y_label='amplitude derivative(Pa)',
                                         text=r'$T_{boom}=%.2f$ s' % boom_duration,
                                         filename=f'{PLOT_DIR}{os.sep}problem_5_timeseries_derivative_plot.png')

    plt.close(fig)

    logger.log(logging.DEBUG, f'Computing spectrum...')
    boom_spectrum = timeseries_to_spectrum(calibrated_boom_ts)

    logger.log(logging.DEBUG, f'Plotting power spectral density...')
    spec_plotter = SpectrumPlotter(boom_spectrum)
    fig, p = spec_plotter.plot_root_gxx(title='Assignment 3a, Problem 5\nBoom F1B2 6 power spectral density',
                                        filename=f'{PLOT_DIR}{os.sep}problem_5_psd_plot.png',
                                        y_label=r'$\sqrt{PSD}\ (\frac{Pa}{\sqrt{Hz}})$')
    plt.xlim(0, 200)
    fig.savefig(f'{PLOT_DIR}{os.sep}problem_5_psd_plot_zoom.png')
    plt.close(fig)

    logger.log(logging.DEBUG, f'Plotting power spectral density (log scale)...')
    fig, p = spec_plotter.plot_db_root_gxx(title='Assignment 3a, Problem 5\nBoom F1B2 6 power spectral density',
                                           filename=f'{PLOT_DIR}{os.sep}problem_5_psd_plot_db.png',
                                           y_label=r'$\sqrt{PSD}\ (dB\ re\ \frac{20\ \mu Pa}{\sqrt{Hz}})$',
                                           reference=2e-5)
    plt.xlim(0, 200)
    plt.ylim(60, 120)
    fig.savefig(f'{PLOT_DIR}{os.sep}problem_5_psd_plot_db_zoom.png')
    plt.close(fig)


if __name__ == '__main__':
    problems = [problem_1, problem_2, problem_3, problem_4, problem_5]
    for p in problems:
        p()
