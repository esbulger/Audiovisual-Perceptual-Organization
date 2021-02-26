"""
Eli Bulger

January 2020

LiMN Lab - Carnegie Mellon University

This contains functions to help show a sequence of visual stimuli with changing characteristics

"""

def plot_signal(time, rSignal, limit=None, limit_high=None,
                name='signal', showF=False, freq=None, fSignal=None):
    """

    plot the signal from generate_signal function


    :param time:
    :param rSignal:
    :param limit:
    :param showF:
    :param freq:
    :param fSignal:
    :return:
    """

    import numpy as np
    import matplotlib.pyplot as plt

    plt.plot(time, rSignal)
    plt.title(name)
    plt.ylabel('x(t)')
    plt.xlabel('time (s)')
    plt.show()

    if showF:
        plt.plot(freq, abs(fSignal))
        plt.xlim([-2 * limit, 2 * limit])
        plt.title('|X(w)| at %i Hz' % limit)
        plt.ylabel('|X(w)|')
        plt.xlabel('freq (Hz)')
        plt.show()

    return

