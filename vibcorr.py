#!/usr/bin/python3
import sys
import os
import time
import configparser
import vibcorr_util as vb

start = time.time()
config = configparser.ConfigParser(interpolation=None)
config.sections()
config.read('vibcorr.ini')
orca = config['SYSTEM']['orcapath']
pal = config['SYSTEM']['pal']
method = config['CALCULATION']['method']
prop_block = config['CALCULATION']['prop_block']
stepsize = config.getfloat('OPTIONS', 'stepsize')
prop_func = vb.prop_funcs[config['OPTIONS']['prop_func']]
jobname = os.path.splitext(sys.argv[1])[0]

# optimize geometry and calculate frequencies
starting_geometry = ''
with open(sys.argv[1], 'r') as xyz:
    xyz.__next__()
    xyz.__next__()
    for i in xyz:
        starting_geometry += i

with open(jobname + '_eq.inp', 'w') as inp:
    inp.write(method.format('pmodel opt tightopt freq',
                            pal,
                            '',
                            starting_geometry,
                            prop_block))
vb.call_orca(jobname + '_eq', orca)


# calculate harmonic ZPVC
vb.harm_corr_from_hess(jobname, jobname + '_eq', prop_block, stepsize, prop_func, method, pal, orca)
print('Runtime ' + str((time.time() - start)/60) + ' min\n')
