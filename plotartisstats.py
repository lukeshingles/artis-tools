#!/usr/bin/env python3
import os
import sys
import math
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import glob

xminvalue, xmaxvalue = 3500, 7000
# xminvalue, xmaxvalue = 10000, 20000

numberofcolumns = 5
h = 6.62607004e-34  # m^2 kg / s
c = 299792458  # m / s


def main():
    logfiles = glob.glob('output_0-0.txt') + glob.glob('*/output_0-0.txt') + \
        glob.glob('*/*/output_0-0.txt')
    if not logfiles:
        print('no output log files found')
        sys.exit()

    fig, axes = plt.subplots(2, 1, sharey=True, figsize=(8, 5), tight_layout={
                             "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    for n, logfilename in enumerate(logfiles):
        runfolder = logfilename.split('/output_0-0.txt')[0]

        timesteptimes = []
        with open(runfolder + '/light_curve.out', 'r') as lcfile:
            for line in lcfile:
                timesteptimes.append(line.split()[0])

        timesteptimes = timesteptimes[:int(len(timesteptimes) / 2)]

        stats = []

        with open(logfilename) as flog:
            for line in flog:
                if line.startswith('timestep '):
                    currenttimestep = int(line.split(' ')[1].split(',')[0])
                    stats.append({})
                    if len(stats) != currenttimestep + 1:
                        print("WRONG TIMESTEP!")
                if line.startswith('k_stat_'):
                    (key, value) = line.split(' = ')
                    stats[-1][key] = int(value)

        linelabel = runfolder

        linestyle = ['-', '--'][int(n / 7)]
        # xvalues = range(len(stats))
        xvalues = timesteptimes
        yvalues = [timestepstats['k_stat_to_r_fb'] for timestepstats in stats]
        axes[0].plot(xvalues, yvalues, linestyle=linestyle, lw=1.5, label=linelabel)
        yvalues = [timestepstats['k_stat_to_ma_collexc'] for timestepstats in stats]
        axes[1].plot(xvalues, yvalues, linestyle=linestyle, lw=1.5, label=linelabel)

    for ax in axes:
        ax.set_xlim(xmin=250, xmax=300)
        # ax.set_ylim(ymin=-0.1,ymax=1.3)

    axes[0].legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
    axes[-1].set_xlabel(r'Time (days)')
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    axes[0].set_ylabel(r'k_stat_to_r_fb')
    axes[1].set_ylabel(r'k_stat_to_ma_collexc')

    fig.savefig('plotartisstats.pdf', format='pdf')
    plt.close()

    # plt.setp(plt.getp(ax, 'xticklabels'), fontsize=fsticklabel)
    # plt.setp(plt.getp(ax, 'yticklabels'), fontsize=fsticklabel)
    # for axis in ['top','bottom','left','right']:
    #    ax.spines[axis].set_linewidth(framewidth)

main()
