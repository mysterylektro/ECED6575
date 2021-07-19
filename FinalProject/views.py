"""
    -*- coding: utf-8 -*-
    Time    : 2021-07-18 3:04 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: views.py
    
    {Description}
    -------------
    
"""

from pyqtgraph.parametertree import Parameter, ParameterTree, parameterTypes
import pyqtgraph.opengl as gl
import pyqtgraph as pg


class SimulationControlParameterTree(ParameterTree):

    params = [{'name': 'Speed of sound', 'type': 'float', 'value': 1500., 'decimals': 6, 'suffix': 'm/s'},
              {'name': 'Signal parameters', 'type': 'group', 'children': [
                  {'name': 'Frequency', 'type': 'float', 'value': 2000., 'decimals': 6, 'suffix': 'Hz'},
                  {'name': 'Pulse length', 'type': 'float', 'value': 1., 'suffix': 's'}
              ]},
              {'name': 'Array parameters', 'type': 'group', 'children': [
                    {'name': 'Number of elements', 'type': 'int', 'value': 64, 'limits': (2, 256)},
                    {'name': 'Element Spacing', 'type': 'float', 'value': 0.125, 'suffix': 'm', 'siPrefix': True},
                    {'name': 'Curvature', 'type': 'float', 'value': 0.0, 'suffix': '/m'},
                    {'name': 'Max curvature', 'type': 'float', 'value': 0.1, 'suffix': '/m'}]},
              {'name': 'Undulation rate', 'type': 'float', 'value': 0.1, 'suffix': 'per second'},
              {'name': 'Start', 'type': 'action'},
              {'name': 'Stop', 'type': 'action'}
              ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameter = Parameter.create(name='Simulation Parameters', type='group', children=self.params)
        self.setParameters(self.parameter, showTop=False)


class SimulationView(pg.GraphicsLayoutWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add beam pattern plot.
        self.azimuthal_plot = self.addPlot(title='Azimuthal Beam Pattern', colspan=2)
        self.azimuthal_plot.axes['left']['item'].setLabel('attenuation (dB)')
        self.azimuthal_plot.axes['bottom']['item'].setLabel('azimuthal angle (deg)')
        self.azimuthal_plot.vb.setLimits(xMin=-180, xMax=180)
        self.azimuthal_plot_curve = self.azimuthal_plot.plot([0], [0])

        # Add signal bearing plots
        self.nextRow()
        self.left_bearing_plot = self.addPlot(title='Left bearing estimate')
        self.right_bearing_plot = self.addPlot(title='Right bearing estimate')

        self.left_bearing_plot.axes['left']['item'].setLabel('time (s)')
        self.left_bearing_plot.axes['bottom']['item'].setLabel('bearing (deg True)')
        self.right_bearing_plot.axes['left']['item'].setLabel('time (s)')
        self.right_bearing_plot.axes['bottom']['item'].setLabel('bearing (deg True)')

        self.left_bearing_curve = self.left_bearing_plot.plot([0], [0])
        self.right_bearing_curve = self.right_bearing_plot.plot([0], [0])
