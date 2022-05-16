"""Script which creates data to which GPR will be applied."""
import numpy as np


def define_zhouetal11(z, x):
    """Zhou, Q., Qian, P. Z., & Zhou, S. (2011). A simple approach to emulation for computer models with qualitative 
    and quantitative factors. Technometrics, 53(3)."""
    sign = 1
  
    if z == 1:
        factor = 6.8
    elif z == 2:
        factor = 7
        sign = -1
    elif z == 3:
        factor = 7.2
        
    return sign * np.cos(factor * np.pi * x / 2)


def create_1D_grid(start=0.0, end=1.0, step_size=0.1):
    return np.linspace(start, end, num = int((end - start) /step_size))


def create_data(func_param, func_name='zhouetal'):
    if func_name == 'zhouetal':
        func_values = define_zhouetal11(func_param[0], func_param[1])
    else:
        raise Exception('Function ' + str(func_name) + ' is not defined.')

    return func_values