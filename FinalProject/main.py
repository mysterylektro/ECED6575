"""
    -*- coding: utf-8 -*-
    Time    : 2021-07-18 2:53 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: main.py
    
    {Description}
    -------------
    
"""
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from views import *
import pyqtgraph.opengl as gl
import sys
from matplotlib import pyplot as plt
import numpy as np
from lib.tactical.arrays import *
import time


class SimulationMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(SimulationMainWindow, self).__init__(*args, **kwargs)
        self.model = SimulationModel()
        uic.loadUi('simulation_mainwindow.ui', self)
        self.setWindowTitle('Deformed Array Beam Pattern Analysis')
        self.traditionalView.setWindowTitle('Traditional View')
        self.correctedView.setWindowTitle('Corrected View')
        self.arrayView.update_array_locations(list(self.model.array.elements.keys()))
        self.controlTree.parameter.sigTreeStateChanged.connect(self.update_parameters)

        self.controlTree.parameter.param('Start').sigActivated.connect(self.start_simulation)
        self.controlTree.parameter.param('Stop').sigActivated.connect(self.pause_simulation)

        self.parameter_mapping = {'Speed of sound': self.model.set_speed_of_sound,
                                  'Frequency': self.model.set_frequency,
                                  'Pulse length': self.model.set_pulse_length,
                                  'Element Spacing': self.model.array.set_spacing,
                                  'Number of elements': self.model.array.set_num_elements,
                                  'Curvature': self.model.array.set_curvature,
                                  'Max curvature': self.model.set_max_curvature,
                                  'Undulation rate': self.model.set_undulation_rate}

        self.model.array.set_phi_resolution(-1*np.pi, np.pi, 360)
        self.model.array.set_theta_resolution(0, 0, 1)
        self.simulation_thread = None
        self.array_curvature_simulator = None

    def update_parameters(self, param, changes):
        for change in changes:
            if change[0].name() in self.parameter_mapping:
                self._update_param(change[0].name(), change[2])

        self.update_array_view()

    def _update_param(self, param_name, value):
        self.parameter_mapping[param_name](value)

    def update_array_view(self):
        locations = list(self.model.array.elements.keys())
        self.arrayView.update_array_locations(locations)
        bp = self.model.array.calculate_beam_pattern(np.pi/2, 0, self.model.frequency)
        x = self.model.array.phis * 180 / np.pi
        self.correctedView.azimuthal_plot_curve.setData(x, bp[:,0])

    def pause_simulation(self):
        if self.array_curvature_simulator is not None:
            self.array_curvature_simulator.running = not self.array_curvature_simulator.running

    def start_simulation(self):
        self.simulation_thread = QtCore.QThread()
        self.array_curvature_simulator = ArrayCurvatureSimulator(self.model.array,
                                                                 self.model.max_curvature,
                                                                 self.model.undulation_rate)

        self.array_curvature_simulator.moveToThread(self.simulation_thread)
        self.simulation_thread.started.connect(self.array_curvature_simulator.simulate)
        self.array_curvature_simulator.finished.connect(self.simulation_thread.quit)
        self.array_curvature_simulator.finished.connect(self.array_curvature_simulator.deleteLater)
        self.simulation_thread.finished.connect(self.simulation_thread.deleteLater)
        self.array_curvature_simulator.progress.connect(self.update_array_view)
        self.array_curvature_simulator.running = True
        self.simulation_thread.start()


class ArrayGLViewer(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opts['distance'] = 30
        self.show()
        self.grid = gl.GLGridItem()
        self.addItem(self.grid)
        self.scatter_plot = None
        self.beam_pattern = None
        self.surf = None

    def update_array_locations(self, locations):
        pos = np.array(locations)
        size = np.ones(len(locations))*10
        if self.scatter_plot is None:
            self.scatter_plot = gl.GLScatterPlotItem(pos=pos, size=size)
            self.addItem(self.scatter_plot)
        else:
            self.scatter_plot.setData(pos=pos, size=size)


class ArrayCurvatureSimulator(QtCore.QObject):
    progress = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()

    def __init__(self, array, max_curvature, undulation_rate):
        super().__init__()
        self.max_curvature = max_curvature
        self.undulation_rate = undulation_rate
        self.array = array

    def simulate(self):
        t = 0
        while self.running:
            self.array.set_curvature(self.max_curvature * np.sin(2 * np.pi * self.undulation_rate * t))
            self.progress.emit()
            time.sleep(0.02)
            t += 0.02
        self.finished.emit()


class SimulationModel(QtCore.QObject):

    progress = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.array = CurvedArray(64)
        self.speed_of_sound = 1500.
        self.frequency = 2000.
        self.pulse_length = 1.0
        self.undulation_rate = 0.1
        self.max_curvature = 0.1
        self.running = True

    def set_max_curvature(self, max_curvature):
        self.max_curvature = max_curvature

    def set_undulation_rate(self, undulation_rate):
        self.undulation_rate = undulation_rate

    def set_speed_of_sound(self, speed_of_sound):
        self.speed_of_sound = speed_of_sound

    def set_frequency(self, frequency):
        self.frequency = frequency

    def set_pulse_length(self, pulse_length):
        self.pulse_length = pulse_length


def main():
    app = pg.mkQApp()
    m = SimulationMainWindow()
    # m.showFullScreen()
    m.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
