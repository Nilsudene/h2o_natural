#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import os

from stalk import LineSearchIteration

from params import pes_dict, co_dict
from run2_surrogate import surrogates


lsis = {}
M_list = [4, 5, 7, 9]
epsilon_p = [[0.005, 0.005]]

xc_srg = 'rhf'
pes_srg = pes_dict[xc_srg]
lsis[xc_srg] = {}
xc_ls = 'b3lyp'
pes_ls = pes_dict[xc_ls]
structure = surrogates[xc_srg].structure.copy()

if xc_srg == xc_ls:
    structure.shift_params([0.1, -0.1])

for eps in epsilon_p:
    eps_str = ''.join(f'_{int(e * 1000):03d}' for e in eps)
    lsis[xc_srg][eps_str] = {}
    for M in M_list:

        path = f'ls_conv_m_noise/{M}-{xc_srg}-{xc_ls}-{eps_str}'

        # surrogates[xc_srg].optimize(epsilon_p=eps) not necessary when plotting

        lsi = LineSearchIteration(
            surrogate=surrogates[xc_srg],
            structure=structure,
            path=path,
            pes=pes_ls,
        )

        for i in range(4):
            lsi.propagate(i, add_sigma=True)

        # Evaluate the latest eqm structure
        lsi.pls().evaluate_eqm(add_sigma=True)
        lsi.pls(0).evaluate_eqm(add_sigma=True)

        print(f'Line-search ({xc_ls}) on {xc_srg} surrogate with {eps_str} epsilons and {M} evaluations:')
        for pls in lsi.pls_list:
            if pls.evaluated:
                if pls.structure.value == None:
                    pls.evaluate_eqm(add_sigma=True)
                
        print(lsi)
        print(surrogates[xc_ls].structure.params)
        print('^^Reference params^^')

        # Now safe to assign
        lsis[xc_srg][eps_str][f'{M}'] = lsi


# plotting
if __name__ == '__main__':

    
    param_colors = ['tab:red', 'tab:blue']  # r, θ

    n_params = len(surrogates[xc_srg].structure.params)

    # Two rows: params + energy, columns = len(M_list)
    fig, axs = plt.subplots(2, len(M_list), figsize=(4 * len(M_list), 5))
    fig.suptitle(f'Line search convergence on {xc_srg} surrogate', fontsize=14)

    ref_params = surrogates[xc_ls].structure.params

    for col_idx, M in enumerate(M_list):
        eps = epsilon_p[0]  
        eps_str = ''.join(f'_{int(e * 1000):03d}' for e in eps)
        lsi = lsis[xc_srg][eps_str][f'{M}']

        ax_params = axs[0, col_idx]
        ax_energy = axs[1, col_idx]

        # Collect data
        params = [lsi.pls(0).structure.params]
        params_err = [lsi.pls(0).structure.params_err]
        energies = []
        energies_err = []
        for pls in lsi.pls_list:
            energies.append(pls.structure.value)
            energies_err.append(pls.structure.error)
            if pls.evaluated:
                params.append(pls.structure_next.params)
                params_err.append(pls.structure_next.params_err)

        params = np.array(params) - ref_params
        params_err = np.array(params_err)
        energies = np.array([0.0 if e is None else e for e in energies], dtype=float)
        energies_err = np.array([0.0 if e is None else e for e in energies_err], dtype=float)

        # Plot parameters (both in same subplot)
        for p_idx, p_color in zip(range(n_params), param_colors):
            label = r'$r$' if p_idx == 0 else r'$\theta$'
            ax_params.errorbar(
                np.arange(len(params)),
                params[:, p_idx],
                yerr=params_err[:, p_idx],
                marker='o',
                linestyle='-',
                color=p_color,
                label=label if col_idx == 0 else None  # only label once
            )
        ax_params.axhline(0, color='black', linestyle='--', linewidth=0.5, label='Reference')
        ax_params.set_title(f'M = {M}')
        ax_params.set_ylabel(r'$\Delta$Parameter')
        ax_params.set_xlabel('Step')

        # Plot energy
        ax_energy.errorbar(
            np.arange(len(energies)),
            energies,
            yerr=energies_err,
            marker='o',
            linestyle='-',
            color='cornflowerblue',
            label=f'M={M}'
        )
        ax_energy.set_ylabel('Energy')
        ax_energy.set_xlabel('Step')

        # Legends
        if col_idx == 0:
            ax_params.legend()
        ax_energy.legend()

    plt.tight_layout()
    os.makedirs('figures_m_noise', exist_ok=True)
    plt.savefig(f'figures_m_noise/{xc_srg}-surrogate.png')
