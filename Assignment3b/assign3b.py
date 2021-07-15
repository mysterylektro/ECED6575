"""
    -*- coding: utf-8 -*-
    Time    : 2021-07-08 10:21 a.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: assign3b.py
    
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

logger = setup_logging('3b')
UBAR_TO_PA = 0.1


def problem_1():
    logger.log(logging.INFO, '----- Starting problem 1 -----\n')

    I = 0.0001  # W/m^2
    theta = 30.  # deg
    v_sand = 1.74e3  # m/s
    rho_sand = 1.98e3  # kg/m^3
    v_water = 1500.  # m/s
    rho_water = 1000  # kg/m^3

    logger.log(logging.INFO, f'Input Parameters:\n'
                             f'I = {I} W/m^2\n'
                             f'Incident Angle = {theta} deg\n'
                             f'Speed of Sound in Sand: {v_sand} m/s\n'
                             f'Density of Sand: {rho_sand} kg/m^3\n'
                             f'Speed of Sound in Water: {v_water} m/s\n'
                             f'Density of Water: {rho_water} kg/m^3')

    n = v_water / v_sand
    mn = rho_sand / rho_water

    R1 = v_water*rho_water
    R2 = v_sand*rho_sand

    incident_angle = np.deg2rad(theta)
    transmitted_angle = np.arcsin(np.sin(incident_angle)/n)

    reflection_coefficient = ((R2*np.cos(incident_angle) - R1*np.cos(transmitted_angle)) /
                              (R2*np.cos(incident_angle) + R1*np.cos(transmitted_angle)))**2

    transmission_coefficient = ((4*R1*R2*np.cos(incident_angle)*np.cos(transmitted_angle)) /
                                (R2*np.cos(incident_angle) + R2*np.cos(transmitted_angle))**2)

    incident_pressure = np.sqrt(2*I*R1)
    reflected_pressure = incident_pressure * np.sqrt(reflection_coefficient)
    transmitted_pressure = incident_pressure * np.sqrt(transmission_coefficient)

    reflected_intensity = reflected_pressure**2 / (2*R1)
    transmitted_intensity = transmitted_pressure ** 2 / (2 * R2)

    critical_angle = np.rad2deg(np.arcsin(n))

    logger.log(logging.INFO, f'\nCalculated Variables:\n'
                             f'R1 = {R1} kg/m^2s\n'
                             f'R2 = {R2} kg/m^2s\n'
                             f'Incident pressure: {incident_pressure} Pa\n'
                             f'Reflected pressure: {reflected_pressure} Pa\n'
                             f'Transmitted pressure: {transmitted_pressure} Pa\n'
                             f'Reflected intensity: {reflected_intensity} W/m^2\n'
                             f'Transmitted intensity: {transmitted_intensity} W/m^2\n'
                             f'Critical angle: {critical_angle} deg\n')


def problem_2():
    logger.log(logging.INFO, '----- Starting problem 2 -----\n')

    # a)

    radius = 0.5  # m
    cross_section = np.pi*radius**2
    TS = 10*np.log10(cross_section/(4*np.pi))

    logger.log(logging.INFO, f'a) Target strength = {TS} dB')

    # b)

    cross_section = 30 * 0.0001  # m^2
    TS = 10*np.log10(cross_section/(4*np.pi))

    logger.log(logging.INFO, f'b) Target strength = {TS} dB')

    # c)
    c = 1500  # m/s
    rho = 1000  # kg/m^3
    f = 30000  # Hz
    power = 100  # W
    directivity_factor = 400*np.pi
    r = 100  # m

    logger.log(logging.INFO, f'c) Input parameters:\n'
                             f'Speed of sound in Water: {c} m/s\n'
                             f'Signal frequency: {f} Hz\n'
                             f'Source power: {power} W\n'
                             f'Source directivity: {directivity_factor}\n'
                             f'Target range: {r} m\n')

    w = 2 * np.pi * f
    kappa = w/c
    radius = np.pi*30/kappa  # m
    cross_section = np.pi*radius**2

    logger.log(logging.INFO, f'Calculated parameters:\n'
                             f'Angular frequency: {w} rad/s\n'
                             f'Kappa: {kappa}\n'
                             f'Target radius: {radius} m\n'
                             f'Target cross-section: {cross_section} m**2\n')

    TS = 10*np.log10(cross_section/(4*np.pi))
    SL = 10*np.log10(power * directivity_factor / (4 * np.pi))
    # Assume spherical spreading
    TL = 20*np.log10(r)

    RL = SL - TL + TS - TL

    logger.log(logging.INFO, f'Target Strength: {TS} dB\n'
                             f'Source Level: {SL} dB\n'
                             f'Transmission Loss: {TL} dB\n'
                             f'Receive Level: {RL} dB\n')

    pressure_level = 2*RL

    logger.log(logging.INFO, f'Pressure level: {pressure_level} dB re 1 uPa @ 1m\n')


def problem_3():

    logger.log(logging.INFO, '----- Starting problem 3 -----\n')

    f = 80.  # Hz
    v_rms = 9.01e-6  # V
    calibration = 10**(110.9/10.)  # ubar / V
    d = 1000  # m
    c = 1500  # m/s
    rho = 1000  # kg/m^3

    logger.log(logging.INFO, f'Input parameters:\n'
                             f'Speed of sound in Water: {c} m/s\n'
                             f'Density of Water: {rho} m/s\n'
                             f'Signal frequency: {f} Hz\n'
                             f'Receive voltage: {v_rms} Vrms\n'
                             f'Calibration factor: {calibration} ubar/V\n'
                             f'Target range: {d} m\n')

    receive_pressure = v_rms * calibration  # uBar
    receive_pressure_pa = receive_pressure * UBAR_TO_PA
    receive_intensity = receive_pressure**2 / (c*rho)
    TL = 20*np.log10(d)

    logger.log(logging.INFO, f'Calculated parameters:\n'
                             f'Receive pressure (uBar): {receive_pressure}\n'
                             f'Receive pressure (Pa): {receive_pressure_pa}\n'
                             f'Receive intensity (W/m^2): {receive_intensity}\n'
                             f'Transmission loss (dB): {TL}\n')

    # dB re Pa:
    SL = 10*np.log10(receive_pressure_pa) + TL
    logger.log(logging.INFO, f'Source level (dB re 1 Pa): {SL}')

    # dB re 1 W/m^2
    SL = 10*np.log10(receive_intensity) + TL
    logger.log(logging.INFO, f'Source level (dB re 1 W/m^2): {SL}')

    # dB re 1 uBar
    SL = 10*np.log10(receive_pressure) + TL
    logger.log(logging.INFO, f'Source level (dB re 1 uBar): {SL}')


def problem_4():
    f = 10000  # Hz
    c = 1500  # m/s
    wavelength = c / f  # m
    duration = 20e-3  # s
    peak_power = 100000  # W
    array_horizontal_spacing = 50. * wavelength  # m
    array_vertical_spacing = 20. * wavelength  # m
    TS = 20  # dB
    SNR = 20  # dB
    surface_scattering_strength = -30  # dB
    absorption_coefficient = 1  # dB/m
    ambient_noise_level = -90  # dB re 1 W/m^2

    rms_power = peak_power / np.sqrt(2)
    signal_bandwidth = 1/duration
    DI = 10*np.log10(4*np.pi*array_vertical_spacing*array_horizontal_spacing/wavelength**2)

    SL = 10*np.log10(rms_power/(4*np.pi))
    NL = ambient_noise_level + 10*np.log10(signal_bandwidth) - DI

    theta_3db = 2 * np.arcsin(0.44 * wavelength / array_horizontal_spacing)
    phi_3db = 2 * np.arcsin(0.44 * wavelength / array_vertical_spacing)

    r = 2 / (theta_3db * c * duration)

    RVB = 10*np.log10(r*theta_3db*c*duration/2)
    TL = 20*np.log10(r) + absorption_coefficient*r


    # Where does f(r) == 78.49 dB


def problem_5():

    f = 500  # Hz
    r = 50000  # m
    SL = 56  # dB re 1 pW/m/Hz
    NL = 6  # dB re 1 pW/m/Hz
    DI = 20  # dB
    T = 3600  # s

    TL = 20*np.log10(r)

    # Noise level reduced by DI
    NL -= DI

    # Source level reduced by TL
    SL -= TL
    SNR = SL - NL
    d = 6.3  # From roc curve
    B = d / T / 10**(SNR/5.)

    dummy = 1


def problem_6():

    reader = WaveFileReader(filename='./Boom_F1B2_6.wav')
    timeseries = reader.full_wavefile()

    spectrogram = timeseries_to_spectrogram(timeseries,
                                            fft_size=512,
                                            n_samples=512,
                                            overlap=0.75,
                                            window='hann')

    gxx = 10*np.log10(spectrogram.gxx())
    fig, ax = plt.subplots(1)
    x_axis, y_axis = spectrogram.positive_frequency_axis(), spectrogram.time_axis()
    xmin, xmax, ymin, ymax = x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]
    ax.imshow(np.transpose(gxx), extent=[xmin, xmax, ymin, ymax], cmap='Greys_r')
    ax.grid(False, 'both')
    ax.set_aspect(xmax / ymax)
    ax.invert_yaxis()
    ax.set_ylabel('time (s)')
    ax.set_xlabel('frequency (Hz)')


def problem_7():

    timeseries = record_timeseries(5.0)
    spectrogram = timeseries_to_spectrogram(timeseries,
                                            fft_size=4096,
                                            n_samples=4096,
                                            overlap=0.75,
                                            window='hann')

    gxx = 10 * np.log10(spectrogram.gxx())
    fig, ax = plt.subplots(1)
    x_axis, y_axis = spectrogram.positive_frequency_axis(), spectrogram.time_axis()
    xmin, xmax, ymin, ymax = x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]
    ax.imshow(np.transpose(gxx), extent=[xmin, xmax, ymax, ymin], cmap='Greys_r')
    ax.grid(False, 'both')
    ax.set_aspect(xmax / ymax)
    ax.set_ylabel('time (s)')
    ax.set_xlabel('frequency (Hz)')

    ax.set_xlim(0, 6500)
    ax.set_aspect(6500 / ymax)

    fig.savefig(f'{PLOT_DIR}\\recorded_spectrogram.png')


if __name__ == '__main__':
    # problem_1()
    # problem_2()
    # problem_3()
    # problem_4()
    # problem_5()
    # problem_6()
    problem_7()
