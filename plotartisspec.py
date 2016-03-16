#!/usr/bin/env python3
import os
import sys
import math
import scipy.signal
import numpy as np
import pandas as pd
import glob
import argparse

parser = argparse.ArgumentParser(description='Plot artis model spectra by finding spec.out files in the current directory or subdirectories.')
parser.add_argument('-specpath', action='store', default='**/spec.out',
                    help='Path to spec.out file (may include wildcards such as * and **)')
parser.add_argument('-listtimesteps', action='store_true', default=False,
                    help='Show the times at each timestep')
parser.add_argument('-timestepmin', type=int, default=70,
                    help='First or only included timestep')
parser.add_argument('-timestepmax', type=int, default=80,
                    help='Last included timestep')
parser.add_argument('-xmin', type=int, default=3500,
                    help='Plot range: minimum wavelength')
parser.add_argument('-xmax', type=int, default=7000,
                    help='Plot range: maximum wavelength')
parser.add_argument('-obsspec', action='append', dest='obsspecfiles',
                    help='Include observational spectrum with this file name')
args = parser.parse_args()

xminvalue, xmaxvalue = args.xmin, args.xmax
#xminvalue, xmaxvalue = 10000, 20000

numberofcolumns = 5
h = 6.62607004e-34 #m^2 kg / s
c = 299792458 #m / s

#specfiles = glob.glob('spec.out') + glob.glob('*/spec.out') + glob.glob('*/*/spec.out')
#could alternatively use
specfiles = glob.glob(args.specpath,recursive=True)
#but this might
#be very slow if called from the wrong place

def main():
    if len(specfiles) == 0:
        print('no spec.out files found')
        sys.exit()
    if args.listtimesteps:
        specdata = pd.read_csv(specfiles[0], delim_whitespace=True)
        print('Time steps and times in days:\n')

        times = specdata.columns
        indexendofcolumnone = math.ceil((len(times)-1) / numberofcolumns)
        for rownum in range(0,indexendofcolumnone):
            strline = ""
            for colnum in range(numberofcolumns):
                if colnum > 0:
                    strline += '\t'
                newindex = rownum + colnum*indexendofcolumnone
                if newindex < len(times):
                    strline += '{:4d}: {:.3f}'.format(newindex,float(times[newindex+1]))
            print(strline)
    else:
        makeplot()

def makeplot():
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8,5), tight_layout={"pad":0.2,"w_pad":0.0,"h_pad":0.0})

    dir = os.path.dirname(os.path.abspath(__file__))
    allobsspectra = {
                     'dop_dered_SN2013aa_20140208_fc_final.txt':('SN2013aa +360d (Maguire et al. in prep)','0.3'),
                     '2010lp_20110928_fors2.txt': ('SN2010lp +264d (Taubenberger et al. 2013)','0.1')
                    }
    obsspectra = [(fn,)+allobsspectra[fn] for fn in args.obsspecfiles]
    for (filename, serieslabel, linecolor) in obsspectra:
      obsfile = os.path.join(dir, 'spectra',filename)
      obsdata = np.loadtxt(obsfile)
      obsdata = obsdata[(obsdata[:,0] > xminvalue) & (obsdata[:,0] < xmaxvalue)]
      obsyvalues = obsdata[:,1] * (1.0 / max(obsdata[:,1]))
      obsyvalues = scipy.signal.savgol_filter(obsyvalues, 31, 3)
      ax.plot(obsdata[:,0], obsyvalues/max(obsyvalues), lw=1.5, label=serieslabel, zorder=-1, color=linecolor)

    #in the spec.out file, the column index is one more than the timestep (because column 0 is wavelength row headers, not flux at a timestep)
    timeindexlow = args.timestepmin+1
    if args.timestepmax:
        timeindexhigh = args.timestepmax+1
        print('Ploting timesteps {} to {}'.format(args.timestepmin,args.timestepmax))
    else:
        print('Ploting timestep {}'.format(args.timestepmin))
        timeindexhigh = timeindexlow

    for s in range(len(specfiles)):
        specdata = np.loadtxt(specfiles[s])
        #specdata = pd.read_csv(specfiles[s], delim_whitespace=True)  #maybe switch to Pandas at some point

        linelabel = '{0} at t={1}d'.format(specfiles[s].split('/spec.out')[0],specdata[0,timeindexlow])
        if timeindexhigh > timeindexlow:
            linelabel += ' to {0}d'.format(specdata[0,timeindexhigh])

        arraynu = specdata[1:,0]
        arraylambda = c / specdata[1:,0]

        arrayFnu = specdata[1:,timeindexlow]

        for timeindex in range(timeindexlow+1,timeindexhigh+1):
            arrayFnu += specdata[1:,timeindex]

        arrayFnu = arrayFnu / (timeindexhigh - timeindexlow + 1)

        #best to use the filter on this list (because it hopefully has regular sampling)
        arrayFnu = scipy.signal.savgol_filter(arrayFnu, 5, 2)

        arrayFlambda = arrayFnu * (arraynu ** 2) / c

        maxyvaluethisseries = max([arrayFlambda[i] if (xminvalue < 1e10 * arraylambda[i] < xmaxvalue) else -99.0 for i in range(len(arrayFlambda))])

        linestyle = ['-','--'][int(s / 7)]
        ax.plot(1e10 * arraylambda, arrayFlambda/maxyvaluethisseries, linestyle=linestyle, lw=1.5-(0.1*s), label=linelabel)

    ax.set_xlim(xmin=xminvalue,xmax=xmaxvalue)
    #        ax.set_xlim(xmin=12000,xmax=19000)
    #ax.set_ylim(ymin=-0.1*maxyvalueglbal,ymax=maxyvalueglobal*1.1)
    ax.set_ylim(ymin=-0.1,ymax=1.4)

    ax.legend(loc='best',handlelength=2,frameon=False,numpoints=1,prop={'size': 8})
    ax.set_xlabel(r'Wavelength ($\AA$)')
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    ax.set_ylabel(r'F$_\lambda$')

    #filenameout = 'plotartisspec_{:}_to_{:}.pdf'.format(*timesteparray)
    filenameout = 'plotartisspec.pdf'
    fig.savefig(filenameout,format='pdf')
    print('Saving {:}'.format(filenameout))
    plt.close()

    #plt.setp(plt.getp(ax, 'xticklabels'), fontsize=fsticklabel)
    #plt.setp(plt.getp(ax, 'yticklabels'), fontsize=fsticklabel)
    #for axis in ['top','bottom','left','right']:
    #    ax.spines[axis].set_linewidth(framewidth)

    #for (x,y,symbol) in zip(highlightedatomicnumbers,highlightedelementyposition,highlightedelements):
    #    ax.annotate(symbol, xy=(x, y - 0.0 * (x % 2)), xycoords='data', textcoords='offset points', xytext=(0,10), horizontalalignment='center', verticalalignment='center', weight='bold', fontsize=fs-1.5)
main()