import numpy as np
import struct
import subprocess


# read in everything from ORCA hess file
class VibMol:
    # hessian stores hessian in cartesians
    # nmodes stores cartesian displacements
    # atoms stores atom symbols
    # coordinates
    # frequencies stores vibrational frequencies in Hartree atomic units
    # M stores 1/sqrt(mi) in Hartree atomic units
    # filename stores the name of files with molecule calculation (without extension)
    # property stores (array of) the property of interest
    def __init__(self, filename=None, geom=None):
        if filename is not None:
            self.filename = filename
            self.property = None
            hess_file = open(filename + '.hess', 'r')

            for i in hess_file:
                if '$hessian' in i:
                    dim = int(hess_file.__next__().strip())  # type: int
                    self.hessian, self.nmodes = np.zeros((dim, dim)), np.zeros((dim, dim))
                    for k in range(dim//5 + [0, 1][bool(dim % 5)]):
                        columns = [int(j) for j in hess_file.__next__().split()]
                        for l in range(dim):
                            line = hess_file.__next__().split()
                            for n in range(len(columns)):
                                self.hessian[int(line[0])][columns[n]] = float(line[n + 1])
                if '$normal_modes' in i:
                    dim = int(hess_file.__next__().strip().split()[0])  # type: int
                    for k in range(dim // 5 + [0, 1][bool(dim % 5)]):
                        columns = [int(j) for j in hess_file.__next__().split()]
                        for l in range(dim):
                            line = hess_file.__next__().split()
                            for n in range(len(columns)):
                                self.nmodes[int(line[0])][columns[n]] = float(line[n + 1])
                if '$atoms' in i:
                    natoms = int(hess_file.__next__().strip())
                    self.M, self.coordinates, self.atoms = np.zeros((natoms * 3, 1)), np.zeros((natoms * 3, 1)), []
                    for j in range(natoms):
                        line = hess_file.__next__().split()
                        self.atoms.append(line[0])
                        for k in range(3):
                            self.coordinates[3*j + k][0] = float(line[k + 2])
                        for k in range(3):
                            self.M[3 * j + k][0] = 1 / (42.69529850619421 * float(line[1]) ** 0.5)
                if '$vibrational_frequencies' in i:
                    dim = int(hess_file.__next__().strip())  # type: int
                    self.frequencies = [0 for j in range(dim)]
                    for j in range(dim):
                        self.frequencies[j] = float(hess_file.__next__().split()[1]) * 5.29E-9 * 2 * 137 * 3.1415926

            hess_file.close()

        if filename is None and geom is not None:
            self.coordinates = geom


def readhfcb(filename):
    hfc = []
    filename = filename.strip('.out') + '.prop'
    with open(filename, 'rb') as out:
        nnucs = struct.unpack('<iii', out.read(12))[1]
        for i in range(nnucs):
            hfc.append(struct.unpack('<' + 'd'*14, out.read(8 * 14))[1])
    return np.array(hfc)


# return a new geometry for numerical differentiation
def step(molecule, mode_seq_num, stepsize):
    modstep = np.zeros(molecule.coordinates.shape)
    modstep[mode_seq_num] = stepsize
    geometry_step = np.dot(molecule.nmodes, modstep) * molecule.M
    incr = abs(stepsize)/np.max(np.abs(geometry_step))
    new_molecule = VibMol(geom=molecule.coordinates + geometry_step*incr)
    new_molecule.atoms = molecule.atoms
    return stepsize*incr, new_molecule


# write an input for step in numerical differentiation
def input_writer(molecule, jobname, index, method, nuclei, pal, mofilename=None):
    newflnm = jobname + '_' + index
    with open(newflnm + '.inp', 'w') as out:
        coords = ''
        for i, j in enumerate(molecule.atoms):
            coords += j + ' ' + ' '.join([str(k) for k in molecule.coordinates[3*i:3*i+3].reshape((3,))]) + '\n'
        if mofilename is None:
            out.write(method.format(coords, nuclei))
        else:
            out.write(method.format('bohrs moread',
                                    pal,
                                    '''% moinp "{}.gbw"'''.format(mofilename),
                                    coords,
                                    nuclei))
    return newflnm


def call_orca(filename, orca):
    with open(filename + '.out', 'w') as out:
        try:
            subprocess.run([orca, filename + '.inp'], timeout=900, stdout=out)
        except subprocess.TimeoutExpired:
            subprocess.run([orca, filename + '.inp'], timeout=900, stdout=out)
    return filename + '.out'


def nth_mode_derivative(molecule,
                        mode_seq_num,
                        stepsize,
                        method,
                        jobname,
                        prop_func,
                        nucs,
                        pal,
                        orca):
    names = []
    for i in ['f', 'b']:
        actual_stepsize, new_molecule = step(molecule, mode_seq_num, stepsize * [-1.0, 1.0][i == 'f'])
        names.append(call_orca(input_writer(new_molecule,
                                            jobname,
                                            str(mode_seq_num)+i,
                                            method,
                                            nucs,
                                            pal,
                                            molecule.filename), orca))
    difference = (prop_func(names[0]) + prop_func(names[1]) - 2 * molecule.property)
    print('Step size {}\nHFC forward {} MHz\nHFC backward {} MHz\nDifference {} MHz\nDerivative {} MHz/au'.format(
        round(actual_stepsize, 5),
        prop_func(names[0]),
        prop_func(names[1]),
        difference,
        difference / actual_stepsize ** 2))
    return difference / actual_stepsize**2


def harm_corr(molecule,
              stepsize,
              method,
              jobname,
              prop_func,
              nuclei,
              pal,
              orca,
              deriv_func):
    correction = np.zeros(molecule.property.shape)
    for i, j in enumerate(molecule.frequencies):
        if j == 0:
            continue
        else:
            print('//////////////////////////////////////////////////////////////////////////////////////////////')
            print('{}th mode frequency {}'.format(i, j))
            temp = deriv_func(molecule,
                              i,
                              stepsize,
                              method,
                              jobname,
                              prop_func,
                              nuclei,
                              pal,
                              orca) / 4 / j
            correction += temp
            print('{}th mode correction {} MHz'.format(i, temp))
    return correction


def harm_corr_from_hess(jobname,
                        starting_geometry_hessfile,
                        nuclei,
                        stepsize,
                        prop_func,
                        hfcmethod,
                        pal,
                        orca,
                        deriv_func=nth_mode_derivative):
    print('*********************************************************************************************')
    print(jobname)
    print('*********************************************************************************************')
    molecule = VibMol(filename=starting_geometry_hessfile)
    molecule.property = prop_func(starting_geometry_hessfile)
    harm_hfc = harm_corr(molecule,
                         stepsize,
                         hfcmethod,
                         jobname,
                         prop_func,
                         nuclei,
                         pal,
                         orca,
                         deriv_func)
    print(*harm_hfc/2.8)
    with open(jobname+'_vibcorr.log', 'w') as out:
        out.write('Equilibrium HFC {} G\n'.format(molecule.property/2.8))
        out.write('Step is {} au\n'.format(stepsize))
        out.write('Harmonic correction {} G\n'.format(harm_hfc/2.8))


prop_funcs = {'readhfcb': readhfcb}






