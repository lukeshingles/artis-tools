#!/usr/bin/env python3
import os
import sys
import math
import struct
import scipy.signal
import numpy as np
import pandas as pd
import glob

def main():
    lcfiles = glob.glob('light_curve.out') + glob.glob('*/light_curve.out') + glob.glob('*/*/light_curve.out')
    #could alternatively use
    #specfiles = glob.glob('**/spec.out',recursive=True)
    #but this might
    #be very slow if called from the wrong place
    
    if len(lcfiles) == 0:
        print('no light_curve.out files found')
        sys.exit()
    else:
        makeplot(lcfiles)

def makeplot(lcfiles):
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8,5), tight_layout={"pad":0.2,"w_pad":0.0,"h_pad":0.0})

    for n in range(len(lcfiles)):
        lighcurvedata = np.loadtxt(lcfiles[n])
        #specdata = pd.read_csv(specfiles[s], delim_whitespace=True)  #maybe switch to Pandas at some point

        linelabel = '{0}'.format(lcfiles[n].split('/light_curve.out')[0])

        arraytime = lighcurvedata[:,0]

        arrayflux = lighcurvedata[:,1]

        ax.plot(arraytime, arrayflux, lw=1.5-(0.1*n), label=linelabel)

    #ax.set_xlim(xmin=xminvalue,xmax=xmaxvalue)
    #ax.set_ylim(ymin=-0.1,ymax=1.3)

    ax.legend(loc='best',handlelength=2,frameon=False,numpoints=1,prop={'size':9})
    ax.set_xlabel(r'Time (days)')
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    ax.set_ylabel(r'F$_\lambda$')

    fig.savefig('plotartislightcurve.pdf',format='pdf')
    plt.close()

    #plt.setp(plt.getp(ax, 'xticklabels'), fontsize=fsticklabel)
    #plt.setp(plt.getp(ax, 'yticklabels'), fontsize=fsticklabel)
    #for axis in ['top','bottom','left','right']:
    #    ax.spines[axis].set_linewidth(framewidth)

main()