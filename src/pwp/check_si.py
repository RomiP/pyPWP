'''
This function checks and corrects static instabilities in the initial profile
for use in PWP model
'''

import seawater as sw
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

class InfLoop(Exception): pass


def read_in_data(filepath):
    dat = sio.loadmat(filepath)
    s = np.squeeze(dat['s'])
    t = np.squeeze(dat['t'])
    z = np.squeeze(dat['z'])
    d = sw.dens0(s,t)
    return s,t,d, z

def plot_profile(z, var):
    h = plt.plot(var, z)
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.show()
    return h

def smooth(x, i_start=0, k=1):
    # x_hat = np.zeros(len(x))
    x_hat = x[:]
    for i in range(i_start, len(x)):
        window = 0
        if i < k:
            window = x[:i + k]
        elif i > len(x) - k:
            window = x[i - k:]
        else:
            window = x[i - k:i + k]

        x_hat[i] = np.mean(window)

    return x_hat

def local_avg(x, i, k=1):
    if i < k:
        window = x[:i + k]
    elif i > len(x) - k:
        window = x[i - k:]
    else:
        window = x[i - k:i + k]

    return np.mean(window)

def smooth_si_2(s,t, i_start=1):
    d = sw.dens0(s,t)
    diff_d = np.diff(d)

    for i in range(i_start,len(diff_d)):
        k = 1
        d_lower = sw.dens0(s[i], t[i])
        d_upper = sw.dens0(s[i-1], t[i-1])
        update = False
        while d_lower < d_upper:
            print('ind =', i, 'k =', k)
            s_hat = local_avg(s, i, k)
            t_hat = local_avg(t, i, k)
            d_lower = sw.dens0(s_hat, t_hat)
            k += 1
            update = True
        if update:
            s[i], t[i] = s_hat, t_hat

    return s,t

def smooth_si(s,t,i_start):
    try:
        for i in range(len(s)-1, i_start,-1):
            d = sw.dens0(s,t)
            mix_to = i-1
            d_lower = d[i]
            d_upper = d[mix_to]
            mix = False
            while d_lower < d_upper:
                mix_to -= 1
                if mix_to < 0:
                    raise InfLoop
                d_upper = d[mix_to]
                mix = True

            if mix:
                s[mix_to:i + 1] = np.mean(s[mix_to:i + 1])
                t[mix_to:i + 1] = np.mean(t[mix_to:i + 1])
                d = sw.dens0(s, t)
    except InfLoop:
        print('Error smoothing SI')
        return

    return s, t

def overwrite_profile(filepath, s_new, t_new):
    dat = sio.loadmat(filepath)
    for i in dat.keys():
        if i == 't':
            dat[i] = t_new.reshape([len(t_new), 1])
        if i == 's':
            dat[i] = s_new.reshape([len(s_new), 1])
        print(i)

    sio.savemat(filepath, dat)


