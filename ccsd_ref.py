from pyscf import gto, scf, cc
from pyscf.geomopt.geometric_solver import optimize
from stalk import ParameterStructure
from params import forward, backward
from h2o_vqe import kernel_vqe
import numpy as np
mol = gto.M(
    atom = 'O 0 0 0; H 0 -1 -0.8; H 0 0.1 -0.8',  # in Angstrom
    basis = 'ccpvdz',
    symmetry = True,
    verbose= 0,
    unit = "Angstrom", # doesnt seem to matter

)
mf = scf.RHF(mol)
mycc = cc.CCSD(mf , frozen=[0,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
mol_eq = optimize(mycc)
pos = mol_eq.atom_coords()

natural_pos = forward(pos)
natural_pos[0] *= 0.52917 # converting from Bohr to Angstrom
corrected_pos = backward(natural_pos)
np.savetxt("ccsd_ref_geom.txt", corrected_pos)
