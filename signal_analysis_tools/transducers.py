"""
    -*- coding: utf-8 -*-
    Time    : 2021-06-15 8:46 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: transducers.py
    
    {Description}
    -------------
    
"""


class Transducer:
    def __init__(self, name='', **kwargs):
        self.area = kwargs.get('area', None)
        self.power = kwargs.get('power', None)
        self.power = kwargs.get('power', None)
        self.name = name
