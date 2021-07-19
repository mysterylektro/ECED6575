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

    I = 1  # W/cm^2
    theta = 30.  # deg
    v_sand = 1740  # m/s
    rho_sand = 1980.  # kg/m^3
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
    R1 = v_water*rho_water
    R2 = v_sand*rho_sand

    incident_angle = np.deg2rad(theta)
    cos_theta = np.cos(incident_angle)
    sin_theta = np.sin(incident_angle)

    transmitted_angle = np.arcsin(sin_theta/n)

    critical_angle = np.rad2deg(np.arcsin(n))

    r_coeff = ((R2 * cos_theta - R1 * np.cos(transmitted_angle)) / (R2 * cos_theta + R1 * np.cos(transmitted_angle)))**2
    t_coeff = 4 * R1 * R2 * cos_theta * np.cos(transmitted_angle) / (R2 * cos_theta + R1 * np.cos(transmitted_angle))**2

    # Alternate method:
    mn = rho_sand / rho_water
    quad = np.sqrt(n**2 - sin_theta**2)
    a_r = ((mn * cos_theta - quad) / (mn * cos_theta + quad))**2
    a_t = 4 * mn * cos_theta * quad / (mn * cos_theta + quad)**2

    transmitted_intensity = t_coeff * I
    reflected_intensity = r_coeff * I

    logger.log(logging.INFO, f'\nCalculated Variables:\n'
                             f'Refraction index: {n}\n'
                             f'Reflection coefficient: {r_coeff}\n'
                             f'Transmission coefficient: {t_coeff}\n'
                             f'Reflected intensity: {reflected_intensity} W/cm^2\n'
                             f'Transmitted intensity: {transmitted_intensity} W/cm^2\n'
                             f'Transmitted angle: {np.rad2deg(transmitted_angle)} deg\n'
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

    EL = SL - TL + TS - TL

    logger.log(logging.INFO, f'Target Strength: {TS} dB\n'
                             f'Source Level: {SL} dB\n'
                             f'Transmission Loss: {TL} dB\n'
                             f'Receive Level: {EL} dB\n')

    pressure_level = EL + 10*np.log10(rho*c)

    logger.log(logging.INFO, f'Pressure level: {pressure_level} dB re 1 uPa @ 1m\n')


def problem_3():

    logger.log(logging.INFO, '----- Starting problem 3 -----\n')

    f = 80.  # Hz
    v_rms = 9.01e-6  # V
    calibration = 110.9  # dB V/ ubar
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

    TL = 20*np.log10(d)

    logger.log(logging.INFO, f'Calculated parameters:\n'
                             f'Transmission loss (dB): {TL}\n')

    # dB re 1 uBar
    SL = 20*np.log10(v_rms) + calibration + TL
    logger.log(logging.INFO, f'Source level (dB re 1 uBar): {SL}')

    # dB re Pa:
    SL = 20*np.log10(v_rms) + calibration + 20*np.log10(0.1) + TL
    logger.log(logging.INFO, f'Source level (dB re 1 Pa): {SL}')

    # dB re 1 W/m^2
    SL -= 10*np.log10(rho*c)
    logger.log(logging.INFO, f'Source level (dB re 1 W/m^2): {SL}')


def problem_4():
    logger.log(logging.INFO, '----- Starting problem 4 -----\n')

    f = 10000  # Hz
    c = 1500  # m/s
    wavelength = c / f  # m
    duration = 20e-3  # s
    peak_power = 100000  # W
    array_horizontal_length = 50. * wavelength  # m
    array_vertical_length = 20. * wavelength  # m
    TS = 20  # dB
    ERR = 20  # dB
    surface_scattering_strength = -30  # dB
    absorption_coefficient = 1  # dB/m
    ambient_noise_level = -90  # dB re 1 W/m^2

    logger.log(logging.INFO, f'Input parameters:\n'
                             f'Signal Frequency: {f} Hz\n'
                             f'Speed of Sound: {c} m/s\n'
                             f'Signal wavelength: {wavelength} m\n'
                             f'Signal duration: {duration} s\n'
                             f'Peak signal power: {peak_power} W\n'
                             f'Horizontal array spacing: {array_horizontal_length} m\n'
                             f'Vertical array spacing: {array_vertical_length} m\n'
                             f'Target strength: {TS} dB\n'
                             f'Echo-to-reverb level: {ERR} dB\n'
                             f'Surface scattering strength: {surface_scattering_strength} dB\n'
                             f'Absorption coefficient: {absorption_coefficient} dB re 1 W/m^2\n')

    rms_power = peak_power / np.sqrt(2)
    signal_bandwidth = 1/duration
    DI = 10*np.log10(4*np.pi*array_vertical_length*array_horizontal_length/wavelength**2)

    SL = 10*np.log10(rms_power/(4*np.pi))
    NL = ambient_noise_level + 10*np.log10(signal_bandwidth) - DI

    theta_3db = 2 * np.arcsin(0.44 * wavelength / array_horizontal_length)
    phi_3db = 2 * np.arcsin(0.44 * wavelength / array_vertical_length)

    # Alternate formula
    # theta_3db = np.deg2rad(2 * 25.3 * wavelength / array_horizontal_length)
    # phi_3db = np.deg2rad(2 * 25.3 * wavelength / array_vertical_length)

    r = 10**((TS - ERR - surface_scattering_strength)/10) * (2/(c*duration*theta_3db))

    TL = 20*np.log10(r) + absorption_coefficient*r/1000  # I think it should be in dB / km
    SNR = SL + DI - 2*TL - NL + TS

    logger.log(logging.INFO, f'Calculated parameters:\n'
                             f'RMS power: {rms_power} W\n'
                             f'Signal bandwidth: {signal_bandwidth} Hz\n'
                             f'Array directivity: {DI} dB\n'
                             f'Signal source level: {SL} dB\n'
                             f'Noise level: {NL} dB\n'
                             f'Horizontal beamwidth: {theta_3db} rads\n'
                             f'Vertical beamwidth: {phi_3db} rads\n'
                             f'Range at defined ERR level: {r} m\n'
                             f'Transmission loss at defined ERR level: {TL} dB\n'
                             f'SNR at defined ERR level: {SNR} dB\n'
                             f'Ambient noise limited: {SNR < ERR}\n'
                             f'Reverb limited: {SNR >= ERR}\n')


def problem_5():

    logger.log(logging.INFO, '----- Starting problem 5 -----\n')

    f = 500  # Hz
    r = 50000  # m
    SL = 56  # dB re 1 pW/m/Hz
    NL = 6  # dB re 1 pW/m/Hz
    DI = 20  # dB
    T = 3600  # s

    logger.log(logging.INFO, f'Input parameters:\n'
                             f'Signal Frequency: {f} Hz\n'
                             f'Target range: {r} m\n'
                             f'Signal source level: {SL} dB re 1 pW/m/Hz\n'
                             f'Ambient noise level: {NL} dB re 1 pW/m/Hz\n'
                             f'Sonar array DI: {DI} dB\n'
                             f'Maximum integration time: {T} s\n')

    TL = 20*np.log10(r)

    # Noise level reduced by DI
    NL -= DI

    # Receive level reduced by TL
    RL = SL - TL
    SNR = RL - NL
    d = 6.3  # From roc curve
    B = d / T / 10**(SNR/5.)

    logger.log(logging.INFO, f'Calculated parameters:\n'
                             f'Transmission loss: {TL} dB\n'
                             f'Noise level: {NL} dB\n'
                             f'Receive level: {RL} dB\n'
                             f'SNR: {SNR} dB\n'
                             f'Detection index: {d}\n'
                             f'Bandwidth: {B} Hz\n')


def problem_6():

    filename = 'T4_C5_3L_dec4a.wav'

    reader = WaveFileReader(filename=f'./{filename}')
    timeseries = reader.full_wavefile()

    fft_size = 512
    overlap = 0.75
    window = 'hann'

    spectrogram = timeseries_to_spectrogram(timeseries,
                                            fft_size=fft_size,
                                            n_samples=fft_size,
                                            overlap=overlap,
                                            window=window)

    rms = spectrogram.rms(window=window)
    fig, ax = plt.subplots(1)
    x_axis = spectrogram.positive_frequency_axis()
    ax.plot(x_axis, rms)
    set_minor_gridlines(ax)
    ax.set_title(f'Assignment 3b, Problem 6\n{filename[:-4]} Mean PSD vs frequency')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'Mean PSD ($\frac{V^2}{Hz}$)')
    ax.semilogy()
    ax.set_xlim(0, x_axis[-1]*0.8)
    ax.set_ylim(1e-9, 1e-5)
    fig.savefig(f'{PLOT_DIR}\\{filename[:-4]}_spectrogram_nfft_{fft_size}_overlap_{overlap}_window_{window}_mean_psd'.replace('.', '_') + '.png')

    plt.close(fig)

    rms_v_time = np.sqrt(spectrogram.rms_v_time(window=window))
    fig, ax = plt.subplots(1)
    x_axis = spectrogram.time_axis()
    ax.plot(x_axis, rms_v_time)
    set_minor_gridlines(ax)
    ax.set_title(f'Assignment 3b, Problem 6\n{filename[:-4]} RMS vs time')
    ax.set_xlabel('time (s)')
    ax.set_ylabel(r'RMS (V)')
    fig.savefig(
        f'{PLOT_DIR}\\{filename[:-4]}_spectrogram_nfft_{fft_size}_overlap_{overlap}_window_{window}_rms_vs_time'.replace(
            '.', '_') + '.png')

    plt.close(fig)

    fig, ax = create_spectrogram_image(spectrogram, window=window)
    plot_filename = f'{PLOT_DIR}\\{filename[:-4]}_spectrogram_nfft_{fft_size}_overlap_{overlap}_window_{window}'.replace('.', '_') + '.png'
    fig.savefig(plot_filename)
    plt.close(fig)


def problem_7():

    timeseries = record_timeseries(5.0)
    fft_size = 4096
    overlap = 0.75
    window = 'hann'
    spectrogram = timeseries_to_spectrogram(timeseries,
                                            fft_size=fft_size,
                                            n_samples=fft_size,
                                            overlap=overlap,
                                            window=window)

    fig, ax = create_spectrogram_image(spectrogram, max_f=6500)
    plot_filename = f'{PLOT_DIR}\\recorded_spectrogram_nfft_{fft_size}_overlap_{overlap}_window_{window}'.replace('.', '_') + '.png'
    fig.savefig(plot_filename)


if __name__ == '__main__':
    problems = [problem_1, problem_2, problem_3, problem_4, problem_5, problem_6, problem_7]
    for p in problems:
        p()
