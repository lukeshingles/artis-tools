import artistools as at
from pathlib import Path
import gzip

import artistools.estimators


def get_atomic_composition(modelpath):
    """Read ion list from output file"""
    atomic_composition = {}

    output = open(modelpath / "output_0-0.txt", 'r').read().splitlines()
    ioncount = 0
    for row in output:
        if row.split()[0] == '[input.c]':
            split_row = row.split()
            if split_row[1] == 'element':
                Z = int(split_row[4])
                ioncount = 0
            elif split_row[1] == 'ion':
                ioncount += 1
                atomic_composition[Z] = ioncount
    return atomic_composition


def parse_ion_row_master(row, outdict, atomic_composition):
    outdict['populations'] = {}

    elements = atomic_composition.keys()

    i = 6
    for atomic_number in elements:
        for ion_stage in range(1, atomic_composition[atomic_number] + 1):
            value_thision = float(row[i])
            outdict['populations'][(atomic_number, ion_stage)] = value_thision
            i += 1

            elpop = outdict['populations'].get(atomic_number, 0)
            outdict['populations'][atomic_number] = elpop + value_thision

            totalpop = outdict['populations'].get('total', 0)
            outdict['populations']['total'] = totalpop + value_thision


def read_master_estimators(modelpath, modeldata):
    estimfiles = at.estimators.get_estimator_files(modelpath)
    if not estimfiles:
        print("No estimator files found")
        return False
    print(f'Reading {len(estimfiles)} estimator files...')

    atomic_composition = get_atomic_composition(modelpath)
    estimators = {}
    for estfile in estimfiles:
        opener = gzip.open if estfile.endswith('.gz') else open

        with opener(estfile, 'rt') as estfile:
            timestep = 0
            modelgridindex = -1
            for line in estfile:
                row = line.split()
                if int(row[0]) < int(modelgridindex):
                    timestep += 1
                modelgridindex = int(row[0])

                estimators[(timestep, modelgridindex)] = {}
                estimators[(timestep, modelgridindex)]['velocity'] = modeldata['velocity'][modelgridindex]

                estimators[(timestep, modelgridindex)]['TR'] = float(row[1])
                estimators[(timestep, modelgridindex)]['Te'] = float(row[2])
                estimators[(timestep, modelgridindex)]['W'] = float(row[3])
                estimators[(timestep, modelgridindex)]['TJ'] = float(row[4])

                parse_ion_row_master(row, estimators[(timestep, modelgridindex)], atomic_composition)

    return estimators



def main():
    modelpath = Path("/Users/ccollins/OneDrive - Queen's University Belfast/test_master/test_master_CO_WD")

    modeldata, _ = at.get_modeldata(Path(modelpath, 'model.txt'))
    estim = read_master_estimators(modelpath, modeldata)

if __name__ == '__main__':
    main()
