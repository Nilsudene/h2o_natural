#!/usr/bin/env python3

from numpy import array, sin, cos, ndarray, pi, random

from pyscf import dft, scf
from pyscf import gto
from pyscf.geomopt.geometric_solver import optimize
from pyscf.gto.mole import tofile

from stalk.params.util import bond_angle, mean_distances
from stalk import ParameterStructure
from stalk.params import PesFunction


# Natural forward mapping using bond lengths and angles
def forward(pos: ndarray):
    pos = pos.reshape(-1, 3)  # make sure of the shape
    # for easier comprehension, list particular atoms
    O0 = pos[0]
    H0 = pos[1]
    H1 = pos[2]

    # for redundancy, calculate mean bond lengths
    r = mean_distances([
        (O0, H0),
        (O0, H1)
    ])
    a = bond_angle(H0, O0, H1, units='rad')
    params = [r, a]
    return params
# end def


# Backward mapping: produce array of atomic positions from parameters
def backward(params: ndarray):
    r = params[0]
    # Transform bond angle to triangular angle
    a = (pi - params[1]) / 2
    # place atoms on a hexagon in the xy-directions
    O0 = [0.0, 0.0, 0.0]
    H1 = [r * sin(a), r * cos(a), 0.0]
    H2 = [r * sin(a), -r * cos(a), 0.0]
    pos = array([O0, H1, H2])
    return pos
# end def


def kernel_pyscf(structure: ParameterStructure, xc = 'pbe'):
    atom = []
    for el, pos in zip(structure.elem, structure.pos):
        atom.append([el, tuple(pos)])
    # end for
    mol = gto.Mole()
    mol.atom = atom
    mol.verbose = 1
    mol.basis = 'ccpvdz'
    mol.unit = 'A'
    mol.ecp = 'ccecp'
    mol.charge = 0
    mol.spin = 0
    mol.symmetry = False
    mol.cart = True
    mol.build()
    if xc == 'rhf':
        mf = scf.RHF(mol)
    elif xc == 'uhf':
        mf = scf.UHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = xc
    return mf
# end def


def relax_pyscf(structure: ParameterStructure, outfile='relax.xyz', xc='pbe'):
    mf = kernel_pyscf(structure=structure, xc = xc)
    mf.kernel()
    mol_eq = optimize(mf, maxsteps=100)
    # Write to external file
    tofile(mol_eq, outfile, format='xyz')
# end def


def pes_pyscf(structure: ParameterStructure, xc='pbe', **kwargs):
    print(f'Computing: {structure.label} ({xc})')
    mf = kernel_pyscf(structure=structure, xc = xc)
    e_scf = mf.kernel()
    return e_scf, 0.0
# end def

def pes_pyscf_sigma(structure: ParameterStructure, sig = 0.0, xc='pbe', **kwargs):
    print(f'Computing: {structure.label} ({xc})')
    mf = kernel_pyscf(structure=structure, xc = xc)
    e_scf = mf.kernel()
    return random.normal(e_scf, sig), sig

sigmas = [0.01, 0.001]
b3lyp_sigma_dict = {}
for sig in sigmas:
    key = f"b3lyp_sigma_{sig}"
    b3lyp_sigma_dict[key] = PesFunction(pes_pyscf_sigma, {'sig': sig, 'xc': 'b3lyp'})
# Treat a collection PESs based on alternative XC functionals
xcs = ['pbe', 'b3lyp', 'lda', 'rhf']
# Colors to go with different functionals for plotting
colors = ['tab:blue', 'tab:orange', 'tab:green', 'rab:yellow']
pes_dict = {}
co_dict = {}
for xc, color in zip(xcs, colors):
    co_dict[xc] = color
    pes_dict[xc] = PesFunction(pes_pyscf, {'xc': xc})
# end for
