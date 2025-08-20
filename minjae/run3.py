#!/usr/bin/env python3

from h2o_vqe import vqe_pes_function
import numpy as np
from matplotlib import pyplot as plt
import os

from stalk import LineSearchIteration
from run2_surrogate import surrogates


# Run line-searches between PES combinations
lsis = {}
p_dep = [1e-3]
epsilon_p = [0.001, 0.001]
xc_srg = 'b3lyp'
xc_ls = 'VQE'
structure = surrogates[xc_srg].structure.copy()
# end if
for p in p_dep:
    lsis[p] = {}
    pes_ls = vqe_pes_function(p_dep=p)
    path = f'results_gatenoise/p_dep:{p}'

    surrogates[xc_srg].optimize(epsilon_p=epsilon_p)
    lsi = LineSearchIteration(
        surrogate=surrogates[xc_srg],
        structure=structure,
        path=path,
        pes=pes_ls,
    )
    for i in range(3):
        lsi.propagate(i, add_sigma=False)
    # end for
    # Evaluate the latest eqm structure
    lsi.pls().evaluate_eqm(add_sigma=False)
    print(f'Line-search ({xc_ls}) on B3LYP surrogate with {p} depolarizing noise on cx gates:')
    print(lsi)
    lsis[p] = lsi
# end for

