#---------------------------------------------------
# Statistical Computing for Scientists and Engineers
# Homework 6
# Fall 2018
# University of Notre Dame
#---------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# For consistency, please use seed(21) in your reports in all computer languages.
np.random.seed(21)

# This function defines the true path of the drone for N time steps.
def data_gen(N):
    def R3(theta):
        return [[np.cos(theta), -np.sin(theta), 0.],
                [np.sin(theta),  np.cos(theta), 0.],
                [0.,       0.,       1.]]

    def R2(theta):
        return [[np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]]

    def R3i(theta):
        return np.linalg.inv(R3(theta))

    J2 = np.eye(2)

    def J1(alphl, betl, alphr, betr, l):
        return [[np.sin(alphl + betl),   -np.cos(alphl + betl),   -l * np.cos(betl)],
                [np.sin(alphr + betr),   -np.cos(alphr + betr),   -l * np.cos(betr)]]

    j1 = J1(np.pi/2, 0, -np.pi/2, np.pi, 1)

    j1i = np.linalg.pinv(j1)

    x = 0
    y = 0
    theta = 0

    vl = np.random.randn() * 0.05 + 0.05
    vr = np.random.randn() * 0.05 + 0.05

    data = np.zeros((N, 5))

    for i in range(N):
        if (i+1) % 10 == 0:
            vl = vl + np.random.rand() * 0.2 - 0.05
            vr = vr + np.random.rand() * 0.2 - 0.05

        vl = max(-0.1, min(0.3, vl))
        vr = max(-0.1, min(0.3, vr))

        r = np.matmul(np.matmul(np.matmul(R3i(theta), j1i), J2), [vl, vr])
        #print r
        x = x + r[0]
        y = y - r[1]
        theta = theta + r[2]

        data[i, 0] = x
        data[i, 1] = y
        data[i, 2] = theta
        data[i, 3] = vl
        data[i, 4] = vr

    return data


plt.show()