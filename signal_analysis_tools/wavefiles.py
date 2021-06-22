"""
    -*- coding: utf-8 -*-
    Time    : 2021-06-01 5:43 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: wavefiles.py
    
    {Description}
    -------------
    
"""

import wave
import struct
from signal_analysis_tools.timeseries import Timeseries
import numpy as np


class WaveFileReader:
    sizes = {1: 'B', 2: 'h', 4: 'i'}

    def __init__(self, filename=None):
        super().__init__()
        self.filename = None
        self.num_samples = None
        self.sample_rate = None
        self.sample_width = None
        self.num_channels = None
        self.format_size = None
        self.pos = 0
        self.EOF = False

        if filename is not None:
            self.set_filename(filename)

    def set_filename(self, filename):
        self.filename = filename
        with wave.open(self.filename, 'r') as f:
            nchannels, sampwidth, framerate, nframes, comptype, compname = f.getparams()
            self.num_samples = nframes // nchannels
            self.sample_rate = framerate
            self.num_channels = nchannels
            self.sample_width = sampwidth
            self.format_size = self.sizes.get(sampwidth)

    def validate_params(self):
        if self.filename is None:
            return False
        if self.EOF:
            raise EOFError

        return True

    def reset(self):
        self.pos = 0
        self.EOF = False

    def next_samples(self, num_samples: int) -> Timeseries:
        if not self.validate_params():
            raise ValueError("Invalid wavefile parameters")

        with wave.open(self.filename, 'r') as f:
            f.setpos(self.pos)
            samples = f.readframes(num_samples)
            num_samples_read = len(samples) // self.sample_width // self.num_channels
            if num_samples_read < num_samples:
                self.EOF = True
            samples = np.array(struct.unpack(f'<{num_samples_read}{self.format_size}', samples))
            data_out = Timeseries(samples, self.sample_rate, time_offset=self.pos // self.sample_width // self.num_channels)
            self.pos += num_samples_read

        return data_out

    def full_wavefile(self) -> Timeseries:
        return self.next_samples(self.num_samples)