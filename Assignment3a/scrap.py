"""
    -*- coding: utf-8 -*-
    Time    : 2021-06-20 3:20 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: scrap.py
    
    {Description}
    -------------
    
"""

import numpy as np
from scipy.fft import fft, ifft
from scipy.signal.windows import flattop

np.random.seed(1000)

n = 2**18
duration = 1.0
freq_res = 1 / duration
sample_rate = duration / n

# Generate a spectrum, and use that to generate a white noise timeseries:

phase = np.random.uniform(0, 2 * np.pi, (n//2 - 1,))
# phase = np.full(n // 2 - 1, fill_value=0)
theta = np.full(n // 2 - 1, fill_value=phase)
pos_freq = np.exp(1j * theta)

s = np.zeros((n,), dtype=complex)
s[1:n // 2] = pos_freq
s[n // 2 + 1:] = np.flip(np.conj(pos_freq))

timeseries = ifft(s * n * freq_res)
# timeseries = np.roll(timeseries, n//2)
# timeseries = np.sin(np.linspace(0, 8*np.pi, n, endpoint=True))

print(f"Timeseries RMS: {np.abs(np.sqrt(np.mean(timeseries**2.)))}")

# Calculate the timeseries RMS from spectrum

s = fft(timeseries * sample_rate)
pos_freq = s[0:n//2 + 1]

weights = np.full(len(pos_freq), fill_value=2)
weights[0], weights[-1] = 1, 1
gxx = np.abs(weights / duration * np.conj(pos_freq) * pos_freq)

print(f"RMS from Gxx integral: {np.sqrt(np.sum(gxx * freq_res))}")

# Calculate the RMS from spectrum with flattop window applied

window = flattop(len(timeseries))

print(f"Timeseries * Window RMS: {np.abs(np.sqrt(np.mean((window*timeseries)**2)))}")

s = fft(window * timeseries * sample_rate)
pos_freq = s[0:n//2 + 1]
gxx = np.abs(weights / duration * np.conj(pos_freq) * pos_freq)

print(f"RMS from Gxx integral (no window correction): {np.sqrt(np.sum(gxx * freq_res))}")

# Correct for window

# Proper correction = RMS of timeseries data / RMS of windowed timeseries data
proper_correction = np.abs(np.sqrt(np.mean(timeseries**2))) / np.abs(np.sqrt(np.mean((window*timeseries)**2)))

# Energy correction factor computed from Parseval's theorem
# weighting = np.abs(timeseries / np.max(timeseries))
# Aw = sum(weighting) / np.sum(window*weighting)
Aw = len(window) / np.sum(window**2)

print(f'Window RMS: {np.sqrt(np.mean(window**2))}')

print(f"Proper correction factor: {proper_correction}")
print(f"Calculated correction factor: {np.sqrt(Aw)}")
print(f"RMS from Gxx integral (window correction): {np.sqrt(np.sum(gxx * Aw * freq_res))}")
