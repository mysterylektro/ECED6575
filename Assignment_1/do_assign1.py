"""
    -*- coding: utf-8 -*-
    Time    : 2021-05-11 7:51 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: do_assign1.py

    {Description}
    -------------

    ECED 6575 (Underwater Acoustics Eng) - Assign 1:  Time Series Analysis
    2020/2021 Summer Term
    May 10, 2021

    calls function(s):  assign1(data_file)
"""

from assign1 import assign1

data_files = ['TRAC1_noise_time', 'TRAC3_sin100_time', 'TRAC11_burstnoise_time', 'TRAC11_burstnoise_time_clean']
for data_file in data_files:
    assign1(data_file + '.csv')
