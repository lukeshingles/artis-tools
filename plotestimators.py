#!/usr/bin/env python3
import os
import sys
import math
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#import numpy as np
import glob

h = 6.62607004e-34 #m^2 kg / s
c = 299792458 #m / s

def main():
    timesteptimes = []
    with open('light_curve.out','r') as lcfile:
        for line in lcfile:
            timesteptimes.append(line.split()[0])

    list_grey_depth = []
    list_te = []
    list_w = []
    list_fe1 = []
    list_fe2 = []
    list_fe3 = []
    list_fe4 = []
    list_fe5 = []
    list_nne = []
    list_heating_bf = []
    list_cooling_bf = []
    list_cooling_ff = []
    list_cooling_coll = []
    with open('estimators_0000.out','r') as estfile:
      linenum = 0
      for line in estfile:
        linenum += 1
        row = line.split()
        if linenum < 7:
            print(linenum,row)
        if linenum % 7 == 1:
            timestep = int(row[1])
            list_grey_depth.append(row[13])
            list_te.append(row[7])
            list_w.append(math.log10(float(row[9])))
            list_nne.append(float(row[15]))
        if linenum % 7 == 2:
            list_fe1.append(math.log10(float(row[2])/list_nne[-1]))
            list_fe2.append(math.log10(float(row[3])/list_nne[-1]))
            list_fe3.append(math.log10(float(row[4])/list_nne[-1]))
            list_fe4.append(math.log10(float(row[5])/list_nne[-1]))
            list_fe5.append(math.log10(float(row[6])/list_nne[-1]))
        if linenum % 7 == 5:
            list_heating_bf.append(math.log10(max(1e-20,float(row[4]))))
        if linenum % 7 == 6:
            list_cooling_ff.append(math.log10(max(1e-15,float(row[2]))))
            list_cooling_bf.append(math.log10(max(1e-15,float(row[4]))))
            list_cooling_coll.append(math.log10(max(1e-15,float(row[6]))))

    timesteptimes = timesteptimes[:len(list_grey_depth)]
    fig, axes = plt.subplots(12, 1, sharex=True, figsize=(8,16), tight_layout={"pad":0.2,"w_pad":0.0,"h_pad":0.0})
    axes[0].plot(timesteptimes, list_grey_depth, lw=1.5, label="")
    axes[0].set_ylabel(r'Grey_depth')
    axes[1].plot(timesteptimes, list_te, lw=1.5, label="")
    axes[1].set_ylabel(r'T_e')
    axes[1].set_ylim(ymin=5800,ymax=6500)
    axes[2].plot(timesteptimes, list_w, lw=1.5, label="")
    axes[2].set_ylabel(r'log W')
    axes[2].set_ylim(ymin=-8,ymax=-5)
    axes[3].plot(timesteptimes, list_fe1, lw=1.5, label="")
    axes[3].set_ylabel(r'FeI')
    axes[4].plot(timesteptimes, list_fe2, lw=1.5, label="")
    axes[4].set_ylabel(r'Fe2')
    axes[5].plot(timesteptimes, list_fe3, lw=1.5, label="")
    axes[5].set_ylabel(r'Fe3')
    axes[6].plot(timesteptimes, list_fe4, lw=1.5, label="")
    axes[6].set_ylabel(r'Fe4')
    axes[7].plot(timesteptimes, list_fe5, lw=1.5, label="")
    axes[7].set_ylabel(r'Fe5')
    axes[8].plot(timesteptimes, list_cooling_ff, lw=1.5, label="")
    axes[8].set_ylabel(r'Cooling ff')
    axes[9].plot(timesteptimes, list_cooling_bf, lw=1.5, label="")
    axes[9].set_ylabel(r'Cooling bf')
    axes[10].plot(timesteptimes, list_cooling_coll, lw=1.5, label="")
    axes[10].set_ylabel(r'Cooling coll')
    axes[11].plot(timesteptimes, list_heating_bf, lw=1.5, label="")
    axes[11].set_ylabel(r'Heating bf')

    for ax in axes:
      pass
      #ax.set_xlim(xmin=270,xmax=300)
      #ax.set_ylim(ymin=-0.1,ymax=1.3)

    #ax.legend(loc='best',handlelength=2,frameon=False,numpoints=1,prop={'size':9})
    axes[-1].set_xlabel(r'Time (days)')
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))

    fig.savefig('plotestimators.pdf',format='pdf')
    plt.close()

main()