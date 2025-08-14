from pyscf import gto, scf, cc
from pyscf.geomopt.geometric_solver import optimize
from stalk import ParameterStructure
from params import forward, backward
from h2o_vqe import kernel_vqe

mol = gto.M(
    atom = 'O 0 0 0; H 0 1 0; H 1 -0.8 0',  # in Angstrom
    basis = 'ccpvdz',
    symmetry = True,
    verbose= 0,
    unit = "ANG", # doesnt seem to matter

)
mf = scf.RHF(mol)
mf.kernel()
mycc = cc.CCSD(mf).run()
mol_eq = optimize(mycc, frozen=[0,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], verbose = 0)
pos = mol_eq.atom_coords()
natural_pos = forward(pos)
natural_pos[0] *= 0.529177249 # converting from Bohr to Angstrom
print(natural_pos)

structure = ParameterStructure(
    forward=forward,
    backward=backward,
    params=natural_pos,
    elem=['O'] + 2 * ['H'],
    units='A'
)
reference_energy, _ = kernel_vqe(structure, active_orbitals=[1,2,3,4,5], trials=1, optimizer="L-BFGS-B")
print(f"Reference energy: {reference_energy:.6f} Ha")
