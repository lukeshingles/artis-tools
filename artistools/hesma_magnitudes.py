#!/usr/bin/env python3
"""Artistools - spectra related functions."""
import argparse
import math
from collections import namedtuple
from functools import lru_cache
from pathlib import Path
import os

import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import yaml
from astropy import constants as const
from astropy import units as u
import re

import artistools as at
import artistools.radfield
import artistools.hesma_magnitudes

from scipy.interpolate import interp1d

#from artistools.lightcurve import bolometric_magnitude
from artistools.spectra import stackspectra


def get_hesma_magnitudes(modelpath, args, angle=None):
    """Method adapted from https://github.com/cinserra/S3/blob/master/src/s3/SMS.py"""

    # Take the optical spectral timeseries file from https://hesma.h-its.org/doku.php?id=data:overview for the
    # model you want and put it into the data/hesma folder




    # if Path(modelpath, 'def_2014_n1def_spectra.dat.txt').is_file():
    #     specfilenamehesma = os.path.join(modelpath, "def_2014_n1def_spectra.dat.txt")
    #     specdatahesma = pd.read_csv(specfilenamehesma, delim_whitespace=True)
    #
    #     timearray = [i for i in specdatahesma.columns.values[1:]]

    if args.plot_hesma_model:
        hesma_directory = os.path.join(at.PYDIR, 'data/hesma')
        specfilenamehesma = args.plot_hesma_model
        hesma_modelname = hesma_directory / specfilenamehesma
        specdatahesma = pd.read_csv(hesma_modelname, delim_whitespace=True)


        timearray = [i for i in specdatahesma.columns.values[1:]]





    filters_dict = {}
    if not args.filter:
        args.filter = ['B']

    # if args.filter[0] == 'bol':
    #     filters_dict['bol'] = [
    #         (time, bol_magnitude) for time, bol_magnitude in zip(timearray, bolometric_magnitude(modelpath, timearray, args, angle))
    #         if math.isfinite(bol_magnitude)]
    else:
        filters_list = args.filter
        #print("printing filters list line 216", filters_list)
        #print("printing filter_dict line 217", filters_list)

        # filters_list = ['B']

        for filter_name in filters_list:
            if filter_name not in filters_dict:
                filters_dict[filter_name] = []

        filterdir = os.path.join(at.PYDIR, 'data/filters/')


        #print("printing filters_list line 250", filters_list)
        for filter_name in filters_list:

            zeropointenergyflux, wavefilter, transmission, wavefilter_min, wavefilter_max \
                = get_filter_data(filterdir, filter_name)
            #print(zeropointenergyflux)
            #print(wavefilter)
            for timestep, time in enumerate(timearray):
                if args.xmin and float(time) < args.xmin:
                    continue
                if args.xmax and float(time) > args.xmax:
                    continue
                #print("wavelength from spectrum line 247", wavelength_from_spectrum)
                wavelength_from_spectrum, flux = get_spectrum_in_filter_range(modelpath, timestep, time, wavefilter_min, wavefilter_max, args, angle)

                if len(wavelength_from_spectrum) > len(wavefilter):
                    interpolate_fn = interp1d(wavefilter, transmission, bounds_error=False, fill_value=0.)
                    wavefilter = np.linspace(min(wavelength_from_spectrum), int(max(wavelength_from_spectrum)), len(wavelength_from_spectrum))
                    transmission = interpolate_fn(wavefilter)
                else:
                    #print(wavelength_from_spectrum, '-------------------------------------')
                    #print(flux, "+++++++++++++++++++++++++++++")
                    interpolate_fn = interp1d(wavelength_from_spectrum, flux, bounds_error=False, fill_value=0.)
                    wavelength_from_spectrum = np.linspace(wavefilter_min, wavefilter_max, len(wavefilter))
                    flux = interpolate_fn(wavelength_from_spectrum)


                phot_filtobs_sn = evaluate_magnitudes(flux, transmission, wavelength_from_spectrum, zeropointenergyflux)
                #print("printing phot_filtobs_sn at line 118", phot_filtobs_sn)
                if phot_filtobs_sn != 0.0:
                    phot_filtobs_sn = phot_filtobs_sn #- 25  # Absolute magnitude
                    filters_dict[filter_name].append((timearray[timestep], phot_filtobs_sn))

                #print("printing filters_dict at 281", filters_dict)
                hesma_filters_dict=filters_dict
                #print(hesma_filters_dict)
    #print(hesma_filters_dict, "hesma_filters_dict")            #print("printing hesma filter dict in hesma_magnitudes line 125", hesma_filters_dict)
    return hesma_filters_dict


def get_filter_data(filterdir, filter_name):
    """Filter data in 'data/filters' taken from https://github.com/cinserra/S3/tree/master/src/s3/metadata"""

    #print("in get_filter_data line 385")
    #print(filterdir)
    #print(filter_name)
    print(Path(filter_name + '.txt'))
    with open(filterdir / Path(filter_name + '.txt'), 'r') as filter_metadata:  # defintion of the file
        line_in_filter_metadata = filter_metadata.readlines()  # list of lines

    zeropointenergyflux = float(line_in_filter_metadata[0])
    # zero point in energy flux (erg/cm^2/s)

    wavefilter, transmission = [], []
    for row in line_in_filter_metadata[4:]:
        # lines where the wave and transmission are stored
        wavefilter.append(float(row.split()[0]))
        transmission.append(float(row.split()[1]))

    #print(transmission)
    wavefilter_min = min(wavefilter)
    wavefilter_max = int(max(wavefilter))  # integer is needed for a sharper cut-off

    return zeropointenergyflux, np.array(wavefilter), np.array(transmission), wavefilter_min, wavefilter_max



def get_spectrum_in_filter_range(modelpath, timestep, time, wavefilter_min, wavefilter_max, args, angle=None):
    if angle != None:
        if args.plotvspecpol:
            spectrum = at.spectra.get_vspecpol_spectrum(modelpath, time, angle, args)
        else:
            res_specdata = at.spectra.read_specpol_res(modelpath, angle=angle)
            res_specdata = at.spectra.select_viewing_angle(angle, args, res_specdata)
            spectrum = at.spectra.get_res_spectrum(modelpath, timestep, timestep, angle=angle,
                                                   res_specdata=res_specdata)
    else:
        #print("in this bit line 168") --------going into this bit
        #spectrum = at.spectra.get_spectrum(modelpath, timestep, timestep)
        spectrum = get_spectrum(modelpath, timestep, timestep, args=args)

    #print("printing spectrum at line 415", spectrum)
    wavelength_from_spectrum, flux = [], []

    for wavelength, flambda in zip(spectrum['lambda_angstroms'], spectrum['f_lambda']):

        if wavefilter_min <= wavelength <= wavefilter_max:  # to match the spectrum wavelengths to those of the filter
            wavelength_from_spectrum.append(wavelength)
            flux.append(flambda)


    return np.array(wavelength_from_spectrum), np.array(flux)



def get_res_spectrum(modelpath, timestepmin: int, timestepmax=-1, angle=None, res_specdata=None, fnufilterfunc=None, reftime=None, args=None):
    """Return a pandas DataFrame containing an ARTIS emergent spectrum."""
    if timestepmax < 0:
        timestepmax = timestepmin

    print(f"Reading spectrum at timestep {timestepmin}")

    if angle is None:
        angle = args.plotviewingangle[0]

    if res_specdata is None:
        print("Reading specpol_res.out")
        res_specdata = at.spectra.read_specpol_res(modelpath, angle)
        res_specdata = at.spectra.select_viewing_angle(angle, args, res_specdata)

    nu = res_specdata[angle].loc[:, 'nu'].values
    # if master_branch:
    timearray = [i for i in res_specdata[angle].columns.values[1:] if i[-2] != '.']
    # else:
    #     timearray = res_specdata[angle].columns.values[1:]




def get_spectrum(modelpath, timestepmin: int, timestepmax=-1, flambdafilterfunc=None, reftime=None, args=None):
    """Return a pandas DataFrame containing an ARTIS emergent spectrum."""
    if timestepmax < 0:
        timestepmax = timestepmin

    master_branch = False
    # if Path(modelpath, 'specpol.out').is_file():
    #     specfilename = Path(modelpath) / "specpol.out"
    #     master_branch = True
    # if Path(modelpath, 'def_2014_n1def_spectra.dat.txt').is_file():
    #     specfilename = Path(modelpath) / "def_2014_n1def_spectra.dat.txt"
    #     master_branch = True

    # Path(modelpath, 'def_2014_n1def_spectra.dat.txt').is_file():


    #specfilename = Path(modelpath) / "def_2014_n1def_spectra.dat.txt"
    # print("plot_hesma_model args line 219", at.plot_hesma_model(args))

    hesma_directory = os.path.join(at.PYDIR, 'data/hesma')
    specfilename = args.plot_hesma_model
    hesma_modelname = hesma_directory / specfilename
    specdata = pd.read_csv(hesma_modelname, delim_whitespace=True)



    # specfilename = args.plot_hesma_model
    # print("specfilename line 219", specfilename)
    # master_branch = True
    # specdata = pd.read_csv(specfilename, delim_whitespace=True)

    # elif Path(modelpath).is_dir():
    #     specfilename = at.firstexisting(['spec.out.xz', 'spec.out.gz', 'spec.out'], path=modelpath)
    # else:
    #     specfilename = modelpath
    #     specdata = pd.read_csv(specfilename, delim_whitespace=True)
    # print("specdata at line 59", specdata)
    #print("specdata at line 240", specdata)
    specdata = specdata.rename(columns={'0.00': 'lambda_angstroms'})
    #print("specdata at line 61", specdata)
    #print("printing specfilename line 60", specfilename)
    # if master_branch:
    # stokes_params = get_polarisation(args, specdata=specdata)
    # if args is not None:
    #     specdata = stokes_params[args.stokesparam]
    # else:
    #     specdata = stokes_params['I']

    # print("specdata at line 68 in spectra.py", specdata)
    lambda_angstroms = specdata.loc[:, 'lambda_angstroms'].values
    if master_branch:
        timearray = [i for i in specdata.columns.values[1:] if i[-2] != '.']
    else:
        timearray = specdata.columns.values[1:]

    def timefluxscale(timestep):
        if reftime is not None:
            return math.exp(float(timearray[timestep]) / 133.) / math.exp(reftime / 133.)
        else:
            return 1.

    f_lambda = stackspectra([
        (specdata[specdata.columns[timestep + 1]] * timefluxscale(timestep),
         at.get_timestep_time_delta(timestep, timearray))
        for timestep in range(timestepmin, timestepmax + 1)])

    # best to use the filter on this list because it
    # has regular sampling
    if flambdafilterfunc:
        print("Applying filter to ARTIS spectrum")
        f_lambda = flambdafilterfunc(f_lambda)

    dfspectrum = pd.DataFrame({'lambda_angstroms': lambda_angstroms, 'f_lambda': f_lambda})
    dfspectrum.sort_values(by='lambda_angstroms', ascending=False, inplace=True)

    #print("printing dfspectrum line 104", dfspectrum)

    dfspectrum.eval('nu = @c / lambda_angstroms', local_dict={'c': const.c.to('angstrom/s').value}, inplace=True)
    dfspectrum.eval('f_nu = f_lambda * lambda_angstroms / nu', inplace=True)

    # dfspectrum = dfspectrum.rename(columns={'nu': 'lambda_angstroms'})
    # dfspectrum = dfspectrum.rename(columns={'f_nu': 'f_lambda'})
    # dfspectrum = dfspectrum.rename(columns={'lambda_angstroms': 'nu'})
    # dfspectrum = dfspectrum.rename(columns={'f_lambda': 'f_nu'})

    #print(dfspectrum, "=====================================")
    return dfspectrum



def main(args=None, argsraw=None, **kwargs):

    if __name__ == '__main__':
        main()


def evaluate_magnitudes(flux, transmission, wavelength_from_spectrum, zeropointenergyflux):
    cf = flux * transmission
    flux_obs = abs(np.trapz(cf, wavelength_from_spectrum))  # using trapezoidal rule to integrate
    if flux_obs == 0.0:
        phot_filtobs_sn = 0.0
    else:
        phot_filtobs_sn = -2.5 * np.log10(flux_obs / zeropointenergyflux)

    return phot_filtobs_sn