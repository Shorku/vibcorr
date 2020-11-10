#!/usr/bin/python3
import sys
import os
import time
import vibcorr_util as vb

start = time.time()
myopt = vb.Options(os.path.splitext(sys.argv[1])[0])
sys.stdout = open(myopt.jobname + '_fullvibcorr.log', 'w')

# optimize geometry and calculate frequencies
starting_geometry = ''
with open(myopt.jobname + '.xyz', 'r') as xyz:
    xyz.__next__()
    xyz.__next__()
    for i in xyz:
        starting_geometry += i

with open(myopt.jobname + '_eq.inp', 'w') as inp:
    inp.write(myopt.method.format('pmodel opt verytightopt',
                                  myopt.pal,
                                  '',
                                  myopt.chmult,
                                  starting_geometry,
                                  myopt.prop_block))
vb.call_orca(myopt.jobname + '_eq', myopt.orca_path)

starting_geometry = ''
with open(myopt.jobname + '_eq.xyz', 'r') as xyz:
    xyz.__next__()
    xyz.__next__()
    for i in xyz:
        starting_geometry += i

with open(myopt.jobname + '_freq.inp', 'w') as inp:
    inp.write(myopt.method.format('moread anfreq rijcosx autoaux gridx6',
                                  myopt.pal,
                                  '''% moinp "{}.gbw"'''.format(myopt.jobname + '_eq'),
                                  myopt.chmult,
                                  starting_geometry,
                                  myopt.prop_block))
vb.call_orca(myopt.jobname + '_freq', myopt.orca_path)
os.rename(myopt.jobname + '_freq.hess', myopt.jobname + '_eq.hess')

# calculate harmonic ZPVC
if myopt.anharm:
    vb.anharm_corr_from_hess(myopt, myopt.jobname + '_eq')
else:
    vb.harm_corr_from_hess(myopt, myopt.jobname + '_eq')
print('Runtime ' + str((time.time() - start)/60) + ' min\n')
