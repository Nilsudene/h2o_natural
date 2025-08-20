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
epsilon_p = [[0.02, 0.02], [0.015, 0.015], [0.01, 0.01], [0.005, 0.005]]

for xc_srg, pes_srg in pes_dict.items():
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

            path = f'ls_conv_m{M}/{xc_srg}-{xc_ls}-{eps_str}'

            # surrogates[xc_srg].optimize(epsilon_p=eps) not necessary when plotting

            lsi = LineSearchIteration(
                surrogate=surrogates[xc_srg],
                structure=structure,
                path=path,
                pes=pes_ls,
            )

            for i in range(4):
                lsi.propagate(i, add_sigma=False)

            # Evaluate the latest eqm structure
            print(lsi.pls_list)
            lsi.pls().evaluate_eqm(add_sigma=False)
            lsi.pls(0).evaluate_eqm(add_sigma=False)

            print(f'Line-search ({xc_ls}) on {xc_srg} surrogate with {eps_str} epsilons and {M} evaluations:')
            for pls in lsi.pls_list:
                if pls.evaluated:
                    if pls.structure.value == None:
                        pls.evaluate_eqm(add_sigma=False)
                    
            print(lsi)
            print(surrogates[xc_ls].structure.params)
            print('^^Reference params^^')

            # Now safe to assign
            lsis[xc_srg][eps_str][f'{M}'] = lsi


# Plot
if __name__ == '__main__':

    M_colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']

    for xc_srg in pes_dict:

        n_params = len(surrogates[xc_srg].structure.params)

        # Subplots: n_params rows + 1 row for energy, columns = len(epsilon_p)
        fig, axs = plt.subplots(n_params + 1, len(epsilon_p), figsize=(4 * len(epsilon_p), 2.5 * (n_params + 1)))
        fig.suptitle(f'Line search convergence on {xc_srg} surrogate', fontsize=14)

        ref_params = surrogates[xc_ls].structure.params

        for col_idx, eps in enumerate(epsilon_p):
            ax_params_list = axs[:-1, col_idx]  # one subplot per parameter
            ax_energy = axs[-1, col_idx]        # bottom row: energy

            for m_idx, (M, color) in enumerate(zip(M_list, M_colors)):
                eps_str = ''.join(f'_{int(e * 1000):03d}' for e in eps)
                lsi = lsis[xc_srg][eps_str][f'{M}']

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

                params = abs(np.array(params) - ref_params)
                params_err = np.array(params_err)
                energies = np.array(energies)
                energies_err = np.array(energies_err)
                energies_err = np.array([0.0 if e is None else e for e in energies_err], dtype = float)
                energies = np.array([0.0 if e is None else e for e in energies], dtype = float)

                # Plot each parameter separately vs step with error bars
                for p_idx, ax in enumerate(ax_params_list):
                    label = f'M={M}'
                    ax.errorbar(
                        np.arange(len(params)),
                        params[:, p_idx],
                        yerr=params_err[:, p_idx],
                        marker='o',
                        linestyle='-',
                        color=color,
                        label=label
                    )
                    ax.set_ylabel(f'Param {p_idx}')
                    ax.set_xlabel('Step')
                    if p_idx == 0:
                        ax.set_title(rf"$\epsilon$ = {eps[0]:.3f}")

                # Plot energy vs step
                label_energy = f'M={M}'
                if energies_err is not None:
                    ax_energy.errorbar(
                        np.arange(len(energies)), energies, yerr=energies_err,
                        marker='o', linestyle='-', color=color, label=label_energy
                    )
                else:
                    ax_energy.errorbar(
                        np.arange(len(energies)), energies,
                        marker='o', linestyle='-', color=color, label=label_energy
                    )
                ax_energy.set_ylabel('Energy')
                ax_energy.set_xlabel('Step')

            
            for ax in list(ax_params_list) + [ax_energy]:
                ax.legend()

        plt.tight_layout()
        os.makedirs('figures_m',exist_ok=True )
        plt.savefig(f'figures_m/{xc_srg}-surrogate.png')