#!/usr/bin/env python3

from os import makedirs
from numpy import array, pi

from stalk import ParameterStructure
from stalk.io import XyzGeometry

from params import forward, backward, relax_pyscf, pes_dict


# Let us initiate a ParameterStructure object that implements the parametric mappings
params_init = array([0.97, 104.0 / 180 * pi])
elem = ['O'] + 2 * ['H']
structure = ParameterStructure(
    forward=forward,
    backward=backward,
    params=params_init,
    elem=elem,
    units='A'
)

relax_dir = 'relax/'
makedirs(relax_dir, exist_ok=True)

# Treat a collection relaxed geometries based on alternative XC functionals
structure_relax = {}
for xc, pes in pes_dict.items():
    outfile = f'{relax_dir}{xc}.xyz'
    try:
        geom = XyzGeometry({'suffix': outfile}).load('./')
    except FileNotFoundError:
        new_params = relax_pyscf(structure, outfile, xc=xc)
        geom = XyzGeometry({'suffix': outfile}).load('./')
    # end try
    new_params = structure.map_forward(geom.get_pos())
    structure_relax[xc] = structure.copy(params=new_params)
    print(f'Params ({xc}):')
    print(structure_relax[xc].params)
# end for
