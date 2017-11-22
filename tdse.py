#!/usr/bin/python
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotutil

def gauss(x):
    return ((np.pi/2)**(-1/4))*np.exp(-(x**2) + 1j*x)

def Bdirichlet(q):
    q[0] = 0
    q[-1] = 0
    return q



def free(x):
    return np.zeros((x.size+2), dtype=complex)


def cranknicholson(x,t, alpha, fBNC, fINC, fPOT):
    J = x.size
    N = t.size
    V = fPOT(x)
    k = alpha/2
    q = np.zeros((J+2,N), dtype=complex)

    b = np.zeros((J+2), dtype=complex)
    b[1:-1] = fINC(x)
    b = fBNC(b)

    A = np.zeros((J+2, J+2), dtype=complex)
    for i in range(1, J+1):
        for j in range(0, J+2):
            if i == j:
                A[i, j] = (2*k + 1)
            elif i == j - 1:
                A[i, j] = -k
            elif i == j + 1:
                A[i, j] = -k
    A[0,0] = 1
    A[-1, -1] = 1

    Ainv = np.linalg.inv(A)
    for n in range(N):
        q[:, n] = b

        y = np.zeros((J+2),dtype=complex)

        for i in range(1, J+1):
            y[i] = k*(b[i-1] - 2*b[i] + b[i+1])
        y = b + y
        b = np.dot(Ainv, y)

    return q[1:-1,:]


def init():
    return None


def TDSE(J, N):

    fBNC = Bdirichlet
    fINC = gauss
    fPOT = free
    kappa = 1j
    # hbar/2m = 1
    # hbar = 1
    # m = 1/2
    #J = 500
    #N = 500
    minmaxx = np.array([-50, 50])
    minmaxt = np.array([0.0, 5.0])
    dx = (minmaxx[1]-minmaxx[0])/float(J-1)
    dt = (minmaxt[1]-minmaxt[0])/float(N-1)
    x = minmaxx[0]+np.arange(J)*dx
    t = minmaxt[0]+np.arange(N)*dt
    alpha = (kappa * (dt))/(dx ** 2)

    q = cranknicholson(x, t, alpha, fBNC, fINC, fPOT)

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

    args = parser.parse_args()
    J = args.J
    N = args.N
    x, t, q = TDSE(J, N)
    q = np.abs(q)

    #fig = plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    #ax = fig.add_subplot(111,projection='3d')
    #t2d,x2d = np.meshgrid(t,x)
    #ax.plot_surface(x2d,t2d,q)
    plt.plot(x,q[:,0])
    plt.plot(x,q[:,10])
    plt.plot(x,q[:,50])
    plt.plot(x,q[:,100])
    plt.show()

main()
