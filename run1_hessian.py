#!/usr/bin/env python

from os import makedirs
from numpy import loadtxt, savetxt

from stalk import ParameterHessian

from params import pes_dict
from run0_relax import structure_relax


hessian_dir = 'hessian/'
makedirs(hessian_dir, exist_ok=True)

# Treat a collection parameter hessians based on alternative XC functionals
hessians = {}
for xc, pes in pes_dict.items():
    hessian = ParameterHessian(structure=structure_relax[xc])
    hessian_file = f'{hessian_dir}{xc}.dat'
    try:
        hessian_array = loadtxt(hessian_file)
        hessian.init_hessian_array(hessian_array)
        print(f'Loaded Hessian from: {hessian_file}')
    except FileNotFoundError:
        print('Computing Hessian with finite-difference method:')
        hessian.compute_fdiff(pes=pes, dp=0.01)
        savetxt(hessian_file, hessian.hessian)
    # end try
    hessians[xc] = hessian
    print(f'Hessian ({xc}):')
    print(hessians[xc])
# end for
