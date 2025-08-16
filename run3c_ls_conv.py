#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import os

from stalk import LineSearchIteration

from params import pes_dict, co_dict
from run2_surrogate import surrogates


# Run line-searches between PES combinations
lsis = {}
M = 9
epsilon_p = [[0.005, 0.005]]

xc_srg = 'rhf'
pes_srg = pes_dict[xc_srg]
lsis[xc_srg] = {}
xc_ls = 'b3lyp'
pes_ls = pes_dict[xc_ls]
structure = surrogates[xc_srg].structure.copy()
if xc_srg == xc_ls:
    structure.shift_params([0.1, -0.1]) # only shifting if surrogate is the same as linesearch method
# end if
for eps in epsilon_p:
    eps_str = ''.join(f'_{int(e * 1000):03d}' for e in eps)
    path = f'ls_conv_m_noise/{M}-{xc_srg}-{xc_ls}-{eps_str}'

    lsis[xc_srg][eps_str] = {}
    surrogates[xc_srg].optimize(epsilon_p=eps)
    
    lsi = LineSearchIteration(
        surrogate=surrogates[xc_srg],
        structure=structure,
        path=path,
        pes=pes_ls,
        M = M,
    )
    for i in range(4):
        lsi.propagate(i, add_sigma=True)
    # end for
    # Evaluate the latest eqm structure
    for i in range(5):
        lsi.pls(i).evaluate_eqm(add_sigma=True)
    print(f'Line-search ({xc_ls} + noise) on {xc_srg} surrogate with {eps_str} epsilons:')
    print(lsi)
    print(surrogates[xc_ls].structure.params)
    print('^^Reference params^^')
    lsis[xc_srg][eps_str][f'{M}'] = lsi
# end for
# end for

