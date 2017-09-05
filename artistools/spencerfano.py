#!/usr/bin/env python3
import argparse
import math
import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

import artistools as at
import artistools.estimators
import artistools.nonthermal


minionfraction = 1.e-4  # minimum number fraction of the total population to include in SF solution

defaultoutputfile = 'spencerfano_cell{cell:03d}_ts{timestep:02d}_{time_days:.0f}d.pdf'


def lossfunction_ergs(energy, nne):
    H = 6.6260755e-27
    ME = 9.1093897e-28
    EV = 1.6021772e-12
    PI = 3.1415926535987
    QE = 4.80325E-10

    eulergamma = 0.577215664901532

    omegap = math.sqrt(4 * math.pi * nne * pow(QE, 2) / ME)
    zetae = H * omegap / 2 / PI
    v = math.sqrt(2 * energy / ME)
    # print(f'omegap {omegap:.2e} s^-1')
    # print(f'zetae {zetae:.2e} erg = {zetae / EV:.2e} eV')
    # print(f'v {v:.2e}  cm/s')
    if energy > 14 * EV:
        return nne * 2 * PI * pow(QE, 4) / energy * math.log(2 * energy / zetae)
    else:
        return nne * 2 * PI * pow(QE, 4) / energy * math.log(ME * pow(v, 3) / (eulergamma * pow(QE, 2) * omegap))


def lossfunction(energy_ev, nne_cgs):
    nne = nne_cgs * 1e6   # convert from cm^-3 to m^-3
    energy = energy_ev * 1.60218e-19  # convert eV to J
    qe = 1.609e-19  # elementary charge in Coulombs
    me = 9.10938e-31  # kg
    h = 6.62606e-34  # J s
    eps0 = 8.854187817e-12  # F / m

    # h = 4.1356e-15   # Planck's constant in eV * s
    # me = 0.510998e6  # mass of electron in eV/c^2
    # c = 29979245800.  # c in cm/s

    eulergamma = 0.577215664901532

    # zetae = h / (2 * math.pi) * math.sqrt((4 * math.pi * nne * (qe ** 2)) / me)
    # omegap = 2 * math.pi * zetae / h

    omegap = math.sqrt(nne * pow(qe, 2) / me / eps0)
    zetae = h * omegap / 2 / math.pi

    v = math.sqrt(2 * energy / me)  # velocity in m/s
    # print(f'omegap {omegap:.2e} s^-1')
    # print(f'zetae {zetae:.2e} J = {zetae / 1.602e-19:.2e} eV')
    # print(f'v {v:.2e} m/s')
    if energy > 14:
        lossfunc = nne * 2 * math.pi * (qe ** 4 / (4 * math.pi * eps0) ** 2) / energy * math.log(2 * energy / zetae)
    else:
        lossfunc = (nne * 2 * math.pi * (qe ** 4 / (4 * math.pi * eps0) ** 2) / energy * math.log(me * (v ** 3) /
                    (eulergamma * (qe ** 2) * omegap / (4 * math.pi * eps0))))

    # lossfunc is now in J / m
    return lossfunc / 1.60218e-19 / 100  # eV / cm


def Psecondary(epsilon, e_p, I, J):
    e_s = epsilon - I
    return 1 / (J * math.atan((e_p - I) / (2 * J)) * (1 + ((e_s / J) ** 2)))


def get_J(Z, ionstage, ionpot_ev):
    # returns an energy in eV
    # values from Opal et al. 1971 as applied by Kozma & Fransson 1992
    if (ionstage == 1):
        if (Z == 2):  # He I
            return 15.8
        elif (Z == 10):  # Ne I
            return 24.2
        elif (Z == 18):  # Ar I
            return 10.0

    return 0.6 * ionpot_ev


def sfmatrix_add_ionization_shell(engrid, nnion, row, sfmatrix):
    ar_xs_array = at.nonthermal.get_arxs_array_shell(engrid, row)
    deltaen = engrid[1] - engrid[0]
    ionpot_ev = row.ionpot_ev
    J = get_J(row.Z, row.ionstage, ionpot_ev)
    npts = len(engrid)
    for i, en in enumerate(engrid):

        for j in range(i, npts):
            endash = engrid[j]

            prefactor = nnion * ar_xs_array[j] / (J * math.atan((endash - ionpot_ev) / (2 * J)))

            epsilonb = (endash + ionpot_ev) / 2
            epsilona = endash - en
            ij_contribution = (
                prefactor * J * (math.atan((epsilonb - ionpot_ev) / J) - math.atan((epsilona - ionpot_ev) / J))
                * deltaen)

            if endash >= 2 * en + ionpot_ev:
                epsilona = en + ionpot_ev
                ij_contribution -= (
                    prefactor * J * (math.atan((epsilonb - ionpot_ev) / J) - math.atan((epsilona - ionpot_ev) / J))
                    * deltaen)

            sfmatrix[i, j] += ij_contribution


def addargs(parser):
    parser.add_argument('-modelpath', default='.',
                        help='Path to ARTIS folder')

    parser.add_argument('-timedays', '-time', '-t',
                        help='Time in days to plot')

    parser.add_argument('-timestep', '-ts', type=int, default=70,
                        help='Timestep number to plot')

    parser.add_argument('-modelgridindex', '-cell', type=int, default=0,
                        help='Modelgridindex to plot')

    parser.add_argument('--print-lines', action='store_true', default=False,
                        help='Output details of matching line details to standard out')

    parser.add_argument('-o', action='store', dest='outputfile',
                        default=defaultoutputfile,
                        help='path/filename for PDF file')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot estimated spectra from bound-bound transitions.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    modelpath = args.modelpath
    modeldata, _ = at.get_modeldata(modelpath)
    estimators = at.estimators.read_estimators(modelpath, modeldata)

    fs = 13

    npts = 2048
    engrid = np.linspace(1, 1000, num=npts, endpoint=True)
    source = np.zeros(engrid.shape)

    sfmatrix = np.zeros((npts, npts))
    constvec = np.zeros(npts)

    # timestep = 30
    # modelgridindex = 48

    if args.timedays:
        timestep = at.get_closest_timestep(os.path.join(modelpath, "spec.out"), args.timedays)
    else:
        timestep = args.timestep
    modelgridindex = args.modelgridindex
    estim = estimators[(timestep, modelgridindex)]

    time_days = float(at.get_timestep_time(modelpath, timestep))

    ionpopdict = estim['populations']
    nne = estim['nne']
    nntot = estim['populations']['total']
    deposition_density_ev = estim['gamma_dep'] / 1.6021772e-12  # convert erg to eV

    print(f'timestep {timestep} cell {modelgridindex}')
    print(f'nntot:      {estim["populations"]["total"]:.1e} /cm3')
    print(f'nne:        {nne:.1e} /cm3')
    print(f'deposition: {deposition_density_ev:.2f} eV/s/cm3')

    deltaen = engrid[1] - engrid[0]

    source_spread_pts = math.ceil(npts * 0.03333)
    for s in range(npts):
        # spread the source over some energy width
        if (s < npts - source_spread_pts):
            source[s] = 0.
        elif (s < npts):
            source[s] = 1. / (deltaen * source_spread_pts)

    E_init_ev = np.dot(engrid, source) * deltaen
    print(f'E_init:     {E_init_ev:.2f} eV/s/cm3')

    for i in range(npts):
        for j in range(i, npts):
            constvec[i] += source[j] * deltaen

    for i in range(npts):
        en = engrid[i]
        sfmatrix[i, i] += lossfunction(en, nne)
        # EV = 1.6021772e-12  # in erg
        # print(f"electron loss rate nne={nne:.3e} and {i:d} {en:.2e} eV is {lossfunction(en, nne):.2e} or '
        #       f'{lossfunction_ergs(en * EV, nne) / EV:.2e}")

    dfcollion = at.nonthermal.read_colliondata()

    # ions = [
    #   (26, 1), (26, 2), (26, 3), (26, 4), (26, 5),
    #   (27, 2), (27, 3), (27, 4),
    #   (28, 2), (28, 3), (28, 4), (28, 5),
    # ]
    #
    # ions = [
    #   (26, 2), (26, 3)
    # ]

    ions = []
    for key in ionpopdict.keys():
        # keep only the single populations, not element or total population
        if isinstance(key, tuple) and len(key) == 2:
            ions.append(key)

    for Z, ionstage in ions:
        nnion = ionpopdict[(Z, ionstage)]
        if nnion / nntot >= minionfraction:
            print(f'including Z={Z:2} ion_stage {ionstage:3} ({at.get_ionstring(Z, ionstage)})')
            dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage')
            # print(dfcollion_thision)

            for index, row in dfcollion_thision.iterrows():
                sfmatrix_add_ionization_shell(engrid, nnion, row, sfmatrix)

    print(f'\nSolving Spencer-Fano with {npts} energy points...')
    lu_and_piv = linalg.lu_factor(sfmatrix, overwrite_a=False)
    yvec_reference = linalg.lu_solve(lu_and_piv, constvec, trans=0)
    yvec = yvec_reference * deposition_density_ev / E_init_ev

    # print("\n".join(["{:} {:11.5e}".format(i, y) for i, y in enumerate(yvec)]))

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 4), tight_layout={"pad": 0.3, "w_pad": 0.0, "h_pad": 0.0})

    ax.plot(engrid[1:], np.log10(yvec[1:]), marker="None", lw=1.5, color='black')

    #    plt.setp(plt.getp(ax, 'xticklabels'), fontsize=fsticklabel)
    #    plt.setp(plt.getp(ax, 'yticklabels'), fontsize=fsticklabel)
    #    for axis in ['top','bottom','left','right']:
    #        ax.spines[axis].set_linewidth(framewidth)
    #    ax.annotate(modellabel, xy=(0.97, 0.95), xycoords='axes fraction', horizontalalignment='right',
    #                verticalalignment='top', fontsize=fs)
    # ax.set_yscale('log')
    ax.set_xlim(xmin=engrid[0], xmax=engrid[-1] * 1.0)
    # ax.set_ylim(ymin=5, ymax=14)
    ax.set_xlabel(r'Electron energy [eV]', fontsize=fs)
    ax.set_ylabel(r'y(E)', fontsize=fs)

    outputfilename = args.outputfile.format(cell=modelgridindex, timestep=timestep, time_days=time_days)
    print(f"Saving '{outputfilename}'")
    fig.savefig(outputfilename, format='pdf')
    plt.close()

    frac_ionization = 0.
    for Z, ionstage in ions:
        nnion = ionpopdict[(Z, ionstage)]
        if nnion <= minionfraction:
            continue
        X_ion = nnion / nntot
        dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage')
        ionpot_valence = dfcollion_thision.ionpot_ev.min()

        print(f'\n====> Z={Z:2d} {at.get_ionstring(Z, ionstage)} (valence potential {ionpot_valence:.1f} eV)')

        print(f'   nnion: {nnion:.4f} /cm3')
        print(f'   nnion/nntot: {X_ion:.4f}')

        frac_ionization_ion = 0.
        integralgamma = 0.
        for index, row in dfcollion_thision.iterrows():
            ar_xs_array = at.nonthermal.get_arxs_array_shell(engrid, row)

            frac_ionization_shell = nnion * row.ionpot_ev * np.dot(yvec, ar_xs_array) * deltaen / deposition_density_ev
            print(f'      frac_ionization_shell(n {int(row.n):d} l {int(row.l):d}): '
                  f'{frac_ionization_shell:.4f} (ionpot {row.ionpot_ev:.2f} eV)')

            integralgamma += np.dot(yvec, ar_xs_array) * deltaen * row.ionpot_ev / ionpot_valence

            if frac_ionization_shell > 1:
                frac_ionization_shell = 0.
                print('Ignoring.')
                # for k in range(10):
                #     print(nnion * row.ionpot_ev * yvec_reference[k] * ar_xs_array[k] * deltaen / E_init_ev)

            frac_ionization_ion += frac_ionization_shell

        frac_ionization += frac_ionization_ion

        try:
            eff_ionpot = ionpot_valence * X_ion / frac_ionization_ion
        except ZeroDivisionError:
            eff_ionpot = float('inf')

        print(f'  frac_ionization:  {frac_ionization_ion:.4f}')
        print(f'       eff_ionpot:  {eff_ionpot:.2f} eV')
        print(f'            Gamma:  {deposition_density_ev / nntot / eff_ionpot:.2e}')
        print(f'Alternative Gamma:  {integralgamma:.2e}')

    print(f'\nfrac_ionization_tot: {frac_ionization:.2f}')


if __name__ == "__main__":
    main()
