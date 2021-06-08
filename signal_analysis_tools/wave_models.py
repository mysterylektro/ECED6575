"""
    -*- coding: utf-8 -*-
    Time    : 2021-06-08 9:41 a.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: wave_models
    
    {Description}
    -------------
    
"""
from abc import abstractmethod

import numpy as np
from enum import Enum


class PressureUnits(Enum):
    Pa = 0
    ubar = 1
    atm = 2


class PressureWave:
    conversions = {PressureUnits.Pa: 1,
                   PressureUnits.atm: 9.87e-6,
                   PressureUnits.ubar: 10}

    def __init__(self, **kwargs):
        self.power = kwargs.get('power', 0)  # W
        self.medium_density = kwargs.get('rho', 1000.0)  # kg/m^3
        self.sound_speed_in_medium = kwargs.get('c', 1481.0)  # m/s
        self.frequency = kwargs.get('f', 100.0)  # Hz

    def set_power(self, power: float):
        self.power = power

    def convert_pressure(self, p_pa: float, unit: PressureUnits):
        return p_pa * self.conversions.get(unit)

    @abstractmethod
    def phase_angle(self, r: float = 1.0):
        pass

    @abstractmethod
    def characteristic_sai(self, r: float = 1.0):
        pass

    @abstractmethod
    def get_acoustic_intensity(self, r: float):
        pass

    def wavelength(self):
        return self.sound_speed_in_medium / self.frequency

    def wave_number(self):
        return 2 * np.pi / self.wavelength()

    def peak_pressure(self, r: float = 1.0, units: PressureUnits = PressureUnits.Pa):
        p_peak = np.sqrt(2 * self.get_acoustic_intensity(r) * self.medium_density * self.sound_speed_in_medium)
        return self.convert_pressure(p_peak, units)

    def rms_pressure(self, r: float = 1.0, units: PressureUnits = PressureUnits.Pa):
        p_rms = np.sqrt(self.get_acoustic_intensity(r) * self.medium_density * self.sound_speed_in_medium)
        return self.convert_pressure(p_rms, units)

    def peak_particle_velocity(self, r: float = 1.0):
        return self.peak_pressure(r)/self.characteristic_sai(r)

    def rms_particle_velocity(self, r: float = 1.0):
        return self.rms_pressure(r)/self.characteristic_sai(r)

    def peak_particle_displacement(self, r: float = 1.0):
        return self.peak_particle_velocity(r)/(2*np.pi*self.frequency)

    def rms_particle_displacement(self, r: float = 1.0):
        return self.rms_particle_velocity(r)/(2*np.pi*self.frequency)

    def sound_pressure_level(self, r: float = 1.0, p_ref: float = 1e-6, units: PressureUnits = PressureUnits.Pa):
        p_rms = self.rms_pressure(r, units=units)
        return 20*np.log10(p_rms/p_ref)


class PlaneWave(PressureWave):
    def __init__(self, acoustic_intensity, **kwargs):
        self.acoustic_intensity = acoustic_intensity
        super().__init__(**kwargs)

    def get_acoustic_intensity(self, r: float):
        return self.acoustic_intensity

    def characteristic_sai(self, r: float = 1.0):
        return self.medium_density * self.sound_speed_in_medium

    def phase_angle(self, r: float = 1.0):
        return 0


class SphericalWave(PressureWave):
    def __init__(self, power, **kwargs):
        super().__init__(power=power, **kwargs)

    def get_acoustic_intensity(self, r: float):
        return self.power / (4*np.pi*r**2)

    def reactance(self, r: float = 1.0):
        kr = self.wave_number() * r
        return kr / (1 + kr**2)

    def resistance(self, r: float = 1.0):
        kr = self.wave_number() * r
        return kr**2 / (1 + kr**2)

    def phase_angle(self, r: float = 1.0):
        return np.arctan2(self.reactance(r), self.resistance(r))

    def characteristic_sai(self, r: float = 1.0):
        return self.medium_density * self.sound_speed_in_medium * np.cos(self.phase_angle(r))
