import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def timevol(x, q, N):

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, bitrate=1800)

    def _update_plot(i, fig, phi):
        ax.clear()
        ax.set_xlim([-50,50])
        ax.set_ylim([0,1])
        scat = plt.scatter(x, q[:,i])
        return scat

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_xlim([-50,50])
    ax.set_ylim([0,1])
    scat = plt.scatter(x,q[:,0])

    anim = animation.FuncAnimation(fig, _update_plot, fargs = (fig, scat),
                                   frames=N, interval =100)

    anim.save('scatter.mp4', writer=writer)
