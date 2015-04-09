#!/usr/bin/env python
# Copyright 2015
#
# Filename: fourier_coefficient_combination.py
# Author: Shuai
# Contact: lishuai918@gmail.com
# Created: Wed Mar 25 09:50:49 2015 (+0800)
# Package-Requires: ()
# Last-Updated:
#           By:
#     Update #: 96
#
#
# Commentary:
# This program is going to see the real effect of combining $e^{iwt}$ basis
# set: how they interfere and de-interfere with each other.
#
# The example chosen here is how sine waves make up squre wave.
#
# Code:

import numpy as np
import matplotlib.pyplot as plt

import theano.tensor as T
from theano import function

# Initialize canvas.
# #########################################################################
plt.close("all")
figure = plt.figure()
figure.clf()
axe_container = figure.add_subplot(111)
# #########################################################################

# Set up the signal we would like to visualize.
# #########################################################################
t = T.vector()
fundamental_frequency = 2*np.pi
basis_num = 8
freq_array = []
amplitude_array = []
basis_array = []
x = 0 * t
for basis_No in range(0, basis_num):
    freq = fundamental_frequency * (basis_No * 2 + 1)
    freq_array.append(freq)
    amplitude = 1.0 / (basis_No * 2 + 1)
    amplitude_array.append(amplitude)

    basis = amplitude * T.sin(freq * t)
    # Save individual basis.
    basis_array.append(function([t], basis))

    # Add the basis to the overall signal.
    x += basis

x_given_t = function([t], x)
# #########################################################################

# Actually visualize the signal and bases.
# #########################################################################
sample_points = np.arange(0, 2, 0.005)

signal = x_given_t(sample_points)
axe_container.plot(signal)

for basis in basis_array:
    axe_container.plot(basis(sample_points))

# Set a denser grid.
axe_container.grid(True)
GRID_DENSITY = 20
y_limit = axe_container.get_ylim()
span = (y_limit[1] - y_limit[0]) / GRID_DENSITY
axe_container.set_yticks(np.arange(y_limit[0], y_limit[1], span))
x_limit = axe_container.get_xlim()
span = (x_limit[1] - x_limit[0]) / GRID_DENSITY
axe_container.set_xticks(np.arange(x_limit[0], x_limit[1], span))

# Actually plot.
figure.show()
# #########################################################################

#
# fourier_coefficient_combination.py ends here
