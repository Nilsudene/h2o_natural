#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import os

from stalk import LineSearchIteration

from params import pes_dict, co_dict, pes_pyscf_sigma
from run2_surrogate import surrogates
from stalk.params import PesFunction


# Run line-searches between PES combinations
lsis = {}
epsilon_p = [[0.01, 0.01]]


for xc_srg, pes_srg in pes_dict.items():
    lsis[xc_srg] = {}
    xc_ls = 'b3lyp'
    pes_ls = pes_dict[xc_ls]
    structure = surrogates[xc_srg].structure.copy()
    if xc_srg == xc_ls:
        structure.shift_params([np.random.uniform(-0.1,0.1), np.random.uniform(-0.1,0.1)]) # only shifting if surrogate is the same as linesearch method
    # end if
    for eps in epsilon_p:
        sigmas = [0.0,0.01, 0.001]
        eps_str = ''.join(f'_{int(e * 1000):03d}' for e in eps)
        surrogates[xc_srg].optimize(epsilon_p=eps)
        sigmas.append(min(surrogates[xc_srg].sigma_opt))
        for sig in sigmas:
            for k in range(25):
                # Create a unique path for each line search iteration
                sig_str = f"{sig:.5f}"
                path = f'ls_hist/{xc_srg}-{xc_ls}-{eps_str}-{sig_str}-{k}'

                lsi = LineSearchIteration(
                    surrogate=surrogates[xc_srg],
                    structure=structure,
                    path=path,
                    pes=PesFunction(pes_pyscf_sigma, {'sig': sig, 'xc': 'b3lyp'}),
                )
                # Attempt to propagate the line search
                # had to do this cause the angle mapping is only bijective in a range and values outside break stuff
                max_attempts = 10
                for attempt in range(max_attempts):
                    try:
                        for i in range(4):
                            lsi.propagate(i, add_sigma=False)
                        break
                    except ValueError as e:
                        print(f'attempt {attempt + 1} failed with error: {e}')
                        if attempt == max_attempts - 1:
                            print(f'Failed to propagate after {max_attempts} attempts, skipping this line search.')
                            break
                # end for
                # Evaluate the latest eqm structure
                lsi.pls().evaluate_eqm(add_sigma=False)
                print(f'Line-search ({xc_ls} + {sig_str} std) on {xc_srg} surrogate with {eps} epsilons:')
                print(lsi)
                print(surrogates[xc_ls].structure.params)
                print('^^Reference params^^')
                lsis[xc_srg][sig_str] = lsi
                filename = 'ls_hist/final_params.txt'
                with open(filename, 'a') as f:
                    f.write(f'{k} {lsi.pls(3).structure.params[0]} {lsi.pls(3).structure.params[1]} {lsi.pls(3).structure.params_err[0]} {lsi.pls(3).structure.params_err[1]}\n')
        # end for
    # end for

