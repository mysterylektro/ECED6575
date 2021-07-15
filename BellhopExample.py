"""
    -*- coding: utf-8 -*-
    Time    : 2021-07-08 6:16 p.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: BellhopExample.py
    
    {Description}
    -------------
    
"""
import arlpy.uwapm as pm
import arlpy.plot as plt
import numpy as np

ssp = [
    [ 0, 1540],  # 1540 m/s at the surface
    [10, 1530],  # 1530 m/s at 10 m depth
    [20, 1532],  # 1532 m/s at 20 m depth
    [25, 1533],  # 1533 m/s at 25 m depth
    [30, 1535]   # 1535 m/s at the seabed
]
env = pm.create_env2d(soundspeed=ssp)
pm.plot_ssp(env, width=500)

pm.print_env(env)

env = pm.create_env2d()
rays = pm.compute_eigenrays(env)
pm.plot_rays(rays, env=env, width=900)
