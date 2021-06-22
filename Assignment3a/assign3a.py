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
from scipy.signal.windows.windows import flattop, hann

# Setup output directories.
PLOT_DIR = os.sep.join([os.path.dirname(__file__), 'plots'])
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Setup plotting styles.
sns.plotting_context("paper")
sns.set_style("darkgrid")

logger = setup_logging('3a')


def problem_1():

    # Inputs
    transmit_area = 0.1**2  # m^2
    power = 400.  # W
    frequency = 100000.  # Hz
    rho = 1000.  # kg/m^3
    c = 1500.  # m/s
    alpha = 4.14e-3  # Np / m
    target_distance = 30.  # m
    target_cross_section = 80e-4  # m^2
    scattering_loss_per_m = 0.9/30.  # %

    # Calculated inputs
    wavelength = c/frequency  # m
    transmit_directivity = 4 * np.pi * transmit_area / wavelength**2

    # Calculate acoustic intensity at target range
    scattering_loss = scattering_loss_per_m * target_distance
    absorption_loss = np.exp(-2 * alpha * target_distance)
    incident_acoustic_intensity = power * scattering_loss * absorption_loss * transmit_directivity / (4 * np.pi * target_distance**2)  # W/m^2
    reflected_power = incident_acoustic_intensity * target_cross_section

    received_acoustic_intensity = reflected_power * scattering_loss * absorption_loss / (4 * np.pi * target_distance**2)
    mean_received_pressure = np.sqrt(received_acoustic_intensity * rho * c)

    print(f'Received acoustic intensity: {received_acoustic_intensity} W / m^2')
    print(f'Mean received pressure: {mean_received_pressure} Pa')


def problem_2():
    SL = 200  # dB
    TL = 53  # dB
    NL = 71  # dB
    SNR = 10  # dB

    TS = SNR - SL + 2*TL + NL   # dB

    print(f'TS: {TS} dB')

    cross_section = 10**(TS/10) * 4 * np.pi
    print(f'Cross section: {cross_section} m^2')


def problem_3():
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

    BW = c/(2*range_resolution)  # Bandwidth required in order to resolve 0.15 m

    source_level = 10*np.log10(power * efficiency * Dt / (4 * np.pi))  # dB re 1 W/m^2
    total_noise = NL + 10*np.log10(BW) - 10*np.log10(Dr)  # dB re 1 W/m^2
    transmission_loss = absorption_coefficient * r + 20*np.log10(r)

    TS = SNR - source_level + total_noise + 2*transmission_loss
    cross_section = 10**(TS/10) * 4 * np.pi
    radius = np.sqrt(cross_section/np.pi)

    print(f'TS: {TS}')
    print(f'radius: {radius} m')


def problem_4():
    filename = './Cal2_3V_06_sub2.wav'
    reader = WaveFileReader(filename=filename)
    cal_ts = reader.full_wavefile()
    ts_plotter = TimeseriesPlotter(cal_ts)
    ts_plotter.plot_time_domain(title='Assignment 3a, Problem 4\nCal2 3V 06 sub2 timeseries',
                                filename=f'{PLOT_DIR}{os.sep}problem_4_timeseries_plot.png',
                                linewidth=0.25)

    spectrum_window = flattop(cal_ts.num_samples())

    """
    Adding the window to the spectrum will implicitly calculate the fixed-energy window correction
    and apply it when calculating the Gxx. The fixed-amplitude window correction is applied when the
    user explicitly asks for it via keyword.
    """

    cal_spec = timeseries_to_spectrum(cal_ts, window=spectrum_window)
    spec_plotter = SpectrumPlotter(cal_spec)
    spec_plotter.plot_gxx(y_log=True,
                          filename=f'{PLOT_DIR}{os.sep}problem_4_spectrum_plot_flattop_plot.png',
                          linewidth=0.25)

    spec_plotter.plot_gxx(x_lim=(0, 1000),
                          y_log=True,
                          filename=f'{PLOT_DIR}{os.sep}problem_4_spectrum_plot_flattop_plot_zoom.png',
                          linewidth=0.25)

    freq_rms = np.sqrt(np.sum(cal_spec.gxx()) * cal_spec.f_res)
    time_rms = cal_ts.rms()
    print(f'Freq RMS: {freq_rms}')
    print(f'Time RMS: {time_rms}')

    amplitudes = np.sqrt(cal_spec.gxx(fixed_energy=False) * cal_spec.f_res)
    print(f"250 Hz V: {np.max(amplitudes)}")


def problem_5():
    filename = './Boom_F1B2_6.wav'
    reader = WaveFileReader(filename=filename)
    boom_ts = reader.full_wavefile()
    playback_timeseries(boom_ts, sample_rate=44100)

    calibration_factor = -30 / 8477.332654980304  # Pa-rms per V-rms from Q4
    calibrated_boom_ts = Timeseries(boom_ts.amplitude() * calibration_factor, boom_ts.sample_rate)
    ts_plotter = TimeseriesPlotter(calibrated_boom_ts)
    fig, p = ts_plotter.plot_time_domain(title='Assignment 3a, Problem 5\nBoom F1B2 6 calibrated timeseries',
                                         y_label='amplitude (Pa)')
    p.text(0.05, 0.95, r'$Pa_{max}=%.2f$ Pa' % calibrated_boom_ts.max(),
           transform=p.transAxes,
           fontsize=14, verticalalignment='top', bbox=ts_plotter.TEXTBOX_PROPS)
    fig.savefig(f'{PLOT_DIR}{os.sep}problem_5_timeseries_plot.png')
    plt.close(fig)

    boom_spectrum = timeseries_to_spectrum(calibrated_boom_ts)
    spec_plotter = SpectrumPlotter(boom_spectrum)
    fig, p = spec_plotter.plot_root_gxx(title='Assignment 3a, Problem 5\nBoom F1B2 6 power spectral density',
                                        filename=f'{PLOT_DIR}{os.sep}problem_5_psd_plot.png',
                                        y_label=r'$\sqrt{PSD}\ (\frac{Pa}{\sqrt{Hz}})$')
    plt.xlim(0, 1000)
    fig.savefig(f'{PLOT_DIR}{os.sep}problem_5_psd_plot_zoom.png')
    plt.close(fig)

    fig, p = spec_plotter.plot_db_root_gxx(title='Assignment 3a, Problem 5\nBoom F1B2 6 power spectral density',
                                           filename=f'{PLOT_DIR}{os.sep}problem_5_psd_plot_db.png',
                                           y_label=r'$amplitude\ (dB\ re\ \frac{20\ \mu Pa}{\sqrt{Hz}})$',
                                           reference=2e-5)
    plt.xlim(0, 1000)
    plt.ylim(20, 120)
    fig.savefig(f'{PLOT_DIR}{os.sep}problem_5_psd_plot_db_zoom.png')
    plt.close(fig)


if __name__ == '__main__':
    problems = [problem_1, problem_2, problem_3, problem_4, problem_5]
    for p in problems:
        p()
