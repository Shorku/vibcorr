import numpy as np
import struct
import subprocess
import glob
import os
import random
import configparser


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
    # dim stores hessian dimensions
    # natoms stores number of atoms
    # displacement stores unity displacements along normal coordinates
    def __init__(self, filename=None, geom=None, options=None):
        if filename is not None:
            self.filename = filename
            self.property = None
            self.options = options
            self.gradient = None

            # Read in stuff from .hess file
            hess_file = open(filename + '.hess', 'r')
            for i in hess_file:
                if '$hessian' in i:
                    self.dim = int(hess_file.__next__().strip())  # type: int
                    self.hessian, self.nmodes = np.zeros((self.dim, self.dim)), np.zeros((self.dim, self.dim))
                    for k in range(self.dim//5 + [0, 1][bool(self.dim % 5)]):
                        columns = [int(j) for j in hess_file.__next__().split()]
                        for l in range(self.dim):
                            line = hess_file.__next__().split()
                            for n in range(len(columns)):
                                self.hessian[int(line[0])][columns[n]] = float(line[n + 1])
                if '$normal_modes' in i:
                    self.dim = int(hess_file.__next__().strip().split()[0])  # type: int
                    for k in range(self.dim // 5 + [0, 1][bool(self.dim % 5)]):
                        columns = [int(j) for j in hess_file.__next__().split()]
                        for l in range(self.dim):
                            line = hess_file.__next__().split()
                            for n in range(len(columns)):
                                self.nmodes[int(line[0])][columns[n]] = float(line[n + 1])
                if '$atoms' in i:
                    self.natoms = int(hess_file.__next__().strip())
                    self.M, self.coordinates, self.atoms = np.zeros((self.natoms * 3, 1)), np.zeros((self.natoms * 3, 1)), []
                    for j in range(self.natoms):
                        line = hess_file.__next__().split()
                        self.atoms.append(line[0])
                        for k in range(3):
                            self.coordinates[3*j + k][0] = float(line[k + 2])
                        for k in range(3):
                            self.M[3 * j + k][0] = 1 / (42.7064461515999 * float(line[1]) ** 0.5)
                if '$vibrational_frequencies' in i:
                    self.dim = int(hess_file.__next__().strip())  # type: int
                    self.frequencies = [0 for j in range(self.dim)]
                    for j in range(self.dim):
                        self.frequencies[j] = float(hess_file.__next__().split()[1]) * 5.29E-9 * 2 * 137 * 3.1415926 * 1822.888486
            hess_file.close()

            # TODO switch from mtr to hessian diag
            # Now form displacements using ORCA %mtr block
            self.displacement = np.zeros((self.dim, self.dim), dtype=float)
            self.displacement = self.displacement + self.coordinates
            rnd = str(random.randint(10000, 99999))
            coords = ''
            for i, j in enumerate(self.atoms):
                coords += j + ' ' + ' '.join(
                    [str(k) for k in self.coordinates[3 * i:3 * i + 3].reshape((3,))]) + '\n'

            for i, j in enumerate(self.frequencies):
                if j == 0:
                    continue
                else:
                    inp = mtr_inp.format(self.filename, i, 1.0, options.chmult, coords)
                    with open(rnd + 'mtr.inp', 'w') as inpfile:
                        inpfile.write(inp)
                    call_orca(rnd + 'mtr', options.orca_path)
                    with open(rnd + 'mtr_property.txt') as property_file:
                        for y in property_file:
                            if 'Geometry Index' in y:
                                if y.split()[6] == '1':
                                    property_file.__next__()
                                    for x in range(self.natoms):
                                        temp = property_file.__next__()
                                        for v in range(3):
                                            self.displacement[3 * x + v, i] = float(temp.split()[v + 1]) / 0.5291772083
                                    break
                    for z in glob.glob(os.getcwd() + '/' + rnd + 'mtr*'):
                        try:
                            os.remove(z)
                        except:
                            pass
            self.displacement = self.displacement - self.coordinates

        if filename is None and geom is not None:
            self.coordinates = geom


class Options:
    def __init__(self, jobname, opt_file='vibcorr.ini'):
        config = configparser.ConfigParser(interpolation=None)
        config.sections()
        config.read(opt_file)
        self.jobname = jobname
        self.orca_path = config['SYSTEM']['orcapath']
        self.pal = config['SYSTEM']['pal']
        self.method = config['CALCULATION']['method']
        self.prop_block = config['CALCULATION']['prop_block']
        self.stepsize = config.getfloat('OPTIONS', 'stepsize')
        self.prop_func = prop_funcs[config['OPTIONS']['prop_func']]
        self.chmult = config['CALCULATION']['charge'] + ' ' + config['CALCULATION']['multiplicity']
        self.anharm = config.getboolean('OPTIONS', 'anharmonic')


def readhfcb(filename):
    hfc = []
    filename = filename.strip('.out') + '.prop'
    with open(filename, 'rb') as out:
        nnucs = struct.unpack('<iii', out.read(12))[1]
        for i in range(nnucs):
            hfc.append(struct.unpack('<' + 'd'*14, out.read(8 * 14))[1])
    return np.array(hfc)


def read_grad(filename, molecule):
    grad = np.zeros((1, 3 * len(molecule.atoms)))
    with open(filename + '.engrad', 'r') as file:
        for i in file:
            if 'The current gradient in Eh/bohr' in i:
                file.__next__()
                for j in range(3 * len(molecule.atoms)):
                    grad[0, j] = float(file.__next__().strip())
    return grad


# return a new geometry for numerical differentiation
# in the previous version checked whether cartesian increment was not less than step size, now removed
def step(molecule, mode_seq_num, myopt):
    geometry_b = molecule.coordinates - molecule.displacement[:, mode_seq_num].reshape((molecule.dim, 1)) * myopt.stepsize
    geometry_f = molecule.coordinates + molecule.displacement[:, mode_seq_num].reshape((molecule.dim, 1)) * myopt.stepsize
    new_molecule_b, new_molecule_f = VibMol(geom=geometry_b), VibMol(geom=geometry_f)
    new_molecule_b.atoms, new_molecule_f.atoms = molecule.atoms, molecule.atoms
    return new_molecule_b, new_molecule_f


# write an input for step in numerical differentiation
def input_writer(molecule, myopt, index):
    newflnm = myopt.jobname + '_' + index
    with open(newflnm + '.inp', 'w') as out:
        coords = ''
        for i, j in enumerate(molecule.atoms):
            coords += j + ' ' + ' '.join([str(k) for k in molecule.coordinates[3*i:3*i+3].reshape((3,))]) + '\n'
        out.write(myopt.method.format('bohrs moread engrad' if myopt.anharm else 'bohrs moread',
                                      myopt.pal,
                                      '''% moinp "{}.gbw"'''.format(myopt.jobname + '_eq'),
                                      myopt.chmult,
                                      coords,
                                      myopt.prop_block))
    return newflnm


# TODO deal with timeout
def call_orca(filename, orca):
    with open(filename + '.out', 'w') as out:
        try:
            subprocess.run([orca, filename + '.inp'], timeout=8640000, stdout=out)
        except subprocess.TimeoutExpired:
            subprocess.run([orca, filename + '.inp'], timeout=8649000, stdout=out)
    return filename + '.out'


def nth_mode_derivative(molecule, mode_seq_num, myopt):
    names = []
    new_molecule_b, new_molecule_f = step(molecule, mode_seq_num, myopt)
    for i in ['b', 'f']:
        inpname = input_writer({'b': new_molecule_b, 'f': new_molecule_f}[i], myopt, str(mode_seq_num) + i)
        outname = call_orca(inpname, myopt.orca_path)
        names.append(outname)

    difference = (myopt.prop_func(names[0]) + myopt.prop_func(names[1]) - 2 * molecule.property)
    print('Step size {}\nHFC forward {} MHz\nHFC backward {} MHz\nDifference {} MHz'.format(
        round(myopt.stepsize, 5),
        myopt.prop_func(names[0]),
        myopt.prop_func(names[1]),
        difference))
    return difference / myopt.stepsize**2


def harm_corr(molecule, myopt, deriv_func=nth_mode_derivative):
    correction = np.zeros(molecule.property.shape)
    for i, j in enumerate(molecule.frequencies):
        if j == 0:
            continue
        else:
            print('//////////////////////////////////////////////////////////////////////////////////////////////')
            print('{}th mode frequency {}'.format(i, j))
            temp = deriv_func(molecule, i, myopt) / 4
            correction += temp
            print('Derivative {} MHz/au\n{}th mode correction {} MHz'.format(temp * j * 4, i, temp))
    return correction


def harm_corr_from_hess(myopt, starting_geometry_hessfile):
    print('*********************************************************************************************')
    print(myopt.jobname)
    print('*********************************************************************************************')
    molecule = VibMol(filename=starting_geometry_hessfile, options=myopt)
    molecule.property = myopt.prop_func(starting_geometry_hessfile)
    harm_hfc = harm_corr(molecule, myopt)
    print(*harm_hfc/2.8)
    with open(myopt.jobname+'_vibcorr.log', 'w') as out:
        out.write('Equilibrium HFC {} G\n'.format(molecule.property/2.8))
        out.write('Step is {} au\n'.format(myopt.stepsize))
        out.write('Harmonic correction {} G\n'.format(harm_hfc/2.8))


def anharm_corr_from_hess(myopt, starting_geometry_hessfile):
    print('*********************************************************************************************')
    print(myopt.jobname)
    print('Harmonic correction')
    print('*********************************************************************************************')
    molecule = VibMol(filename=starting_geometry_hessfile, options=myopt)
    molecule.property = myopt.prop_func(starting_geometry_hessfile)
    harm_hfc = harm_corr(molecule, myopt)
    print(*harm_hfc/2.8)
    with open(myopt.jobname+'_vibcorr.log', 'w') as out:
        out.write('Equilibrium HFC {} G\n'.format(molecule.property/2.8))
        out.write('Step is {} au\n'.format(myopt.stepsize))
        out.write('Harmonic correction {} G\n'.format(harm_hfc/2.8))

    molecule.gradient = np.dot(read_grad(starting_geometry_hessfile, molecule), molecule.displacement)
    gradient_f, gradient_b = np.zeros(molecule.displacement.shape), np.zeros(molecule.displacement.shape)
    Kjji = np.zeros(molecule.displacement.shape)
    anharm_hfc = np.zeros(molecule.property.shape)
    for i, j in enumerate(molecule.frequencies):
        if j == 0:
            continue
        else:
            gradient_f[i, :] = np.dot(read_grad(myopt.jobname + '_' + str(i) + 'f', molecule), molecule.displacement)
            gradient_b[i, :] = np.dot(read_grad(myopt.jobname + '_' + str(i) + 'b', molecule), molecule.displacement)
            Kjji[i, :] = (gradient_f[i, :] + gradient_b[i, :] - 2 * molecule.gradient) / (myopt.stepsize ** 2)

    print('*********************************************************************************************')
    print(myopt.jobname)
    print('Anharmonic correction')
    print('*********************************************************************************************')

    Kjji = (Kjji * np.array(molecule.frequencies).reshape((molecule.dim, 1))) * \
                   np.array([1/i if i > 0.0 else 0 for i in molecule.frequencies]).reshape((1, molecule.dim))

    for i, j in enumerate(molecule.frequencies):
        if j == 0:
            continue
        else:
            print('//////////////////////////////////////////////////////////////////////////////////////////////')
            print('{}th mode frequency {}'.format(i, j))
            derivative = (readhfcb(myopt.jobname + '_' + str(i) + 'f') - readhfcb(myopt.jobname + '_' + str(i) + 'b')) \
                          * (j**0.5) / myopt.stepsize / 2
            Q = - np.sum(Kjji[:, i]) / 2**0.5
            corr = derivative * Q
            anharm_hfc += corr
            print('Derivative {} MHz/au\n<Q> = {} au^-1\n correction {} MHz'.format(derivative, Q, corr))
    print(*anharm_hfc/2.8)
    with open(myopt.jobname + '_vibcorr.log', 'a') as out:
        out.write('Anharmonic correction {} G\n'.format(anharm_hfc/2.8))


prop_funcs = {'readhfcb': readhfcb}

bohr = 0.5291772083
mtr_inp = '''! BP86 def2-SV(P) noiter nori bohrs noautostart
%mtr
HessName "{}.hess"
modetype normal
MList {}
RSteps 1
LSteps 1
ddnc {}
end
* xyz {}
{}
*
'''
