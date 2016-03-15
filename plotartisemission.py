#!/usr/bin/env python3
import os
import sys
import math
import scipy.signal
import numpy as np
import pandas as pd
import glob

xminvalue, xmaxvalue = 3500, 7000

pydir = os.path.dirname(os.path.abspath(__file__))
elsymbols = ['n']+[line.split(',')[1] for line in open(os.path.join(pydir,'elements.csv'))]

roman_numerals = ('','I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII',
                  'XIII','XIV','XV','XVI','XVII','XVIII','XIX','XX')

colorlist = ['black',(0.0,0.5,0.7),(0.35,0.7,1.0),(0.9,0.2,0.0),(0.9,0.6,0.0),(0.0,0.6,0.5),(0.8,0.5,1.0),(0.95,0.9,0.25)]

elementlist = []

numberofcolumns = 5
h = 6.62607004e-34 #m^2 kg / s
c = 299792458 #m / s

specfiles = glob.glob('spec.out')# + glob.glob('*/spec.out') + glob.glob('*/*/spec.out')
#could alternatively use
#specfiles = glob.glob('**/spec.out',recursive=True)
#but this might
#be very slow if called from the wrong place

def main():
    if len(specfiles) == 0:
        print('no spec.out files found')
        sys.exit()
    if len(sys.argv) < 2:
        specdata = pd.read_csv(specfiles[0], delim_whitespace=True)
        print('Enter as a commandline argument the timestep for a given time in days (or a range, e.g., 50-100):\n')

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

    with open(specfiles[0].replace('spec.out','compositiondata.txt'),'r') as fcompdata:
        nelements = int(fcompdata.readline())
        fcompdata.readline() #T_preset
        fcompdata.readline() #homogeneous_abundances
        for element in range(nelements):
            line = fcompdata.readline()
            (Z,nions,lowermost_ionstage,uppermost_ionstage,nlevelsmax_readin,abundance,mass) = line.split()
            elementlist.append(int(Z))
    print('nelements {}'.format(nelements))
    maxion = 5

    selectedcolumn = 0 #very important! fix this later
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8,5), tight_layout={"pad":0.2,"w_pad":0.0,"h_pad":0.0})

    timesteparray = list(map(lambda x: int(x), sys.argv[1].split('-')))

    #in the spec.out file, the column index is one more than the timestep (because column 0 is wavelengths, not flux)
    if len(timesteparray) == 1:
        timeindexlow = timesteparray[0]
        timeindexhigh = timeindexlow
    else:
        timeindexlow = timesteparray[0]
        timeindexhigh = timesteparray[1]

    specdata = np.loadtxt(specfiles[0])
    emissiondata = np.loadtxt(specfiles[0].replace('spec.out','emission.out'))

    timearray = specdata[0,1:]
    arraynu = specdata[1:,0]
    arraylambda = c / specdata[1:,0]

    maxyvalueglobal = 0.0
    seriesnumber = 0
    for element in reversed(range(nelements)):
        nionsdisplayeddict = {8: 3, 26: 5}
        nions = nionsdisplayeddict[elementlist[element]]
        for ion in range(nions):
            ionserieslist = []
            if seriesnumber == 0:
                ionserieslist.append( (2*nelements*maxion,'ff') )
                seriesnumber += 1 #so the linestyle resets
            ionserieslist.append( (element*maxion+ion,'bb') )
            ionserieslist.append( (nelements*maxion + element*maxion+ion,'bf') )
            for (selectedcolumn,emissiontype) in ionserieslist:
                arrayFnu = emissiondata[timeindexlow::len(timearray),selectedcolumn]

                for timeindex in range(timeindexlow+1,timeindexhigh+1):
                    arrayFnu += emissiondata[timeindex::len(timearray),selectedcolumn]

                arrayFnu = arrayFnu / (timeindexhigh - timeindexlow + 1)

                #best to use the filter on this list (because it hopefully has regular sampling)
                #arrayFnu = scipy.signal.savgol_filter(arrayFnu, 5, 2)

                arrayFlambda = arrayFnu * (arraynu ** 2) / c

                maxyvaluethisseries = max([arrayFlambda[i] if (xminvalue < (1e10 * arraylambda[i]) < xmaxvalue) else -99.0 for i in range(len(arrayFlambda))])
                maxyvalueglobal = max(maxyvalueglobal,maxyvaluethisseries)

                linelabel = ''
                if emissiontype != 'ff':
                    linelabel += '{:} {:} '.format(elsymbols[elementlist[element]],roman_numerals[ion+1])
                linelabel += '{:} at t={:}d'.format(emissiontype,specdata[0,timeindexlow])
                if timeindexhigh > timeindexlow:
                    linelabel += ' to {0}d'.format(specdata[0,timeindexhigh])
                linewidth = [1.8,0.8][emissiontype=='bf']
                ax.plot(1e10 * arraylambda, arrayFlambda, color=colorlist[int(seriesnumber/2) % len(colorlist)], lw=linewidth, label=linelabel)
                seriesnumber += 1

    ax.set_xlim(xmin=xminvalue,xmax=xmaxvalue)
    #        ax.set_xlim(xmin=12000,xmax=19000)
    ax.set_ylim(ymin=-0.05*maxyvalueglobal,ymax=maxyvalueglobal*1.3)
    #ax.set_ylim(ymin=-0.1,ymax=1.4)

    ax.legend(loc='best',handlelength=2,frameon=False,numpoints=1,prop={'size':8})
    ax.set_xlabel(r'Wavelength ($\AA$)')
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    ax.set_ylabel(r'F$_\lambda$')

    #filenameout = 'plotartisspec_{:}_to_{:}.pdf'.format(*timesteparray)
    filenameout = 'plotartisemission.pdf'
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