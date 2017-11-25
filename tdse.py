#!/usr/bin/python
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import plotutil


def gauss(x):
    return ((np.pi/2)**(-1/4))*np.exp(-(x**2) + .5j*x)


def Bdirichlet(q):
    q[0] = 0
    q[-1] = 0
    return q


"""
=================================================
POTENTIALS
-------------------------------------------------
____
free as in freedom
accepts: positional array size(J)
returns: 0 array size(J+2)

barrier
_______
accepts: positional array size(J)
returns: local potential barrier array size(J+2)
=================================================

"""


def free(x):
    return np.zeros((x.size), dtype=complex)


def barrier(x):
    v = (np.piecewise(x, [x < 6, x > 6], [0.0, .75])
         * np.piecewise(x, [x <= 6.5, x > 6.5], [.75, 0.0])
         )
    return v


"""
=================================================
cranknicholson
______________
accepts:
x    : position array
t    : time array
alpha: -1j * dt/ (dx ** 2)
fBNC : boundary condition function
fINC : initial condition function
fPOT : potential function

returns:
x
t
q    : solution array (J,N) matrix
=================================================
"""


def cranknicholson(x, t, dx, dt, alpha, fBNC, fINC, fPOT):
    J = x.size
    N = t.size
    q = np.zeros((J+2, N), dtype=complex)
    # fetch potentials
    V = np.zeros((J+2), dtype=complex)
    V[1:-1] = fPOT(x)
    # This sets lowercase q in mcdonald report
    k = alpha/2
    r = (1j*dt)/2
    # initial values
    b = np.zeros((J+2), dtype=complex)
    b[1:-1] = fINC(x)
    # apply boundary coniditon
    b = fBNC(b)
    # finite difference matrix + extra implicit terms
    A = np.zeros((J+2, J+2), dtype=complex)
    for i in range(1, J+1):
        for j in range(0, J+2):
            if i == j:
                A[i, j] = (2*k + 1) + 1j*r*V[i]
            elif i == j - 1:
                A[i, j] = -k
            elif i == j + 1:
                A[i, j] = -k

    # Boundary condition
    A[0, 0] = 1
    A[-1, -1] = 1
    # Inverte Left Side, remains same for all iterations
    Ainv = np.linalg.inv(A)
    for n in range(N):
        # add previous solution to solution array
        q[:, n] = b
        # intialize array to right side of linear system
        y = np.zeros((J+2), dtype=complex)
        for i in range(1, J+1):
            # generate right side
            y[i] = k*(b[i-1] - 2*b[i] + b[i+1]) - r*V[i]
        # add previous solution to the RHS vector
        y = b + y
        # calculate solution at next time step
        b = np.dot(Ainv, y)

    # return w/o boundary conditions
    return q[1:-1, :]


"""
=================================================
TDSE()
______
accepts:
J    : position resolution
N    : time resolution

returns:
x
t
q    : solution array (J,N) matrix
=================================================
"""


def TDSE(J, N, problem):

    fBNC = Bdirichlet
    fINC = gauss
    if problem == "free":
        fPOT = free
    if problem == "barrier":
        fPOT = barrier
    kappa = 1j
    # hbar/2m = 1
    # hbar = 1
    # m = 1/2
    minmaxx = np.array([-50, 50])
    minmaxt = np.array([0.0, 5.0])
    dx = (minmaxx[1]-minmaxx[0])/float(J-1)
    dt = (minmaxt[1]-minmaxt[0])/float(N-1)
    x = minmaxx[0]+np.arange(J)*dx
    t = minmaxt[0]+np.arange(N)*dt
    alpha = (kappa * (dt))/(dx ** 2)
    q = cranknicholson(x, t, dx, dt, alpha, fBNC, fINC, fPOT)

    return x, t, q


def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("J", type=int,
                        help="number of spatial support points\n"
                             "try 500"
                        )
    parser.add_argument("N", type=int,
                        help="number of time support points\n"
                             "try 500"
                        )

    parser.add_argument("vis", type=str,
                        help="visualization options\n"
                             "2d, 3d, vid"
                        )

    parser.add_argument("problem", type=str,
                        help="Choose Potential\n"
                             "free, barrier"
                        )

    args = parser.parse_args()
    J = args.J
    N = args.N
    vis = args.vis
    problem = args.problem

    x, t, q = TDSE(J, N, problem)
    # plot magnitude squared (probaiblity density function)
    q = np.abs(q) ** 2

    if vis == "3d":
        fig = plt.figure(num=1, figsize=(8, 8), dpi=100, facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        t2d, x2d = np.meshgrid(t, x)
        ax.plot_surface(x2d, t2d, q, cmap=cm.rainbow)
        plt.show()

    if vis == "2d":
        if problem == "barrier":
            plt.plot(x, barrier(x))
        plt.plot(x, q[:, 0])
        plt.plot(x, q[:, int(N/10)])
        plt.plot(x, q[:, int(N/5)])
        plt.plot(x, q[:, int(N/2)])
        plt.plot(x, q[:, int(N/1.5)])
        plt.show()

    if vis == "vid":
        plotutil.timevol(x, q, N, problem, barrier)


main()
