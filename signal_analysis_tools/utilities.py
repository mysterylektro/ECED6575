"""
    -*- coding: utf-8 -*-
    Time    : 2021-05-21 10:08 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: utilities.py
    
    {Description}
    -------------
    
"""
import logging

import matplotlib as mpl


def set_minor_gridlines(ax):
    """
    Helper function to set minor gridlines on seaborn plots.
    Args:
        ax: input plot axis

    Returns: None

    """
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='minor', linewidth=0.5, linestyle=':')


def setup_logging(assignment_name='1a'):

    # Setup logging.
    output_log_filename = f'assignment{assignment_name}.log'
    logger = logging.getLogger(f'assignment{assignment_name}')
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    log_file_handler = logging.FileHandler(output_log_filename, 'w+')
    log_file_handler.setLevel(logging.INFO)
    logger.addHandler(log_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    return logger
