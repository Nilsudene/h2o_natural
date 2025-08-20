#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import os

from stalk import LineSearchIteration

from params import pes_dict, co_dict, forward
from run2_surrogate import surrogates
from h2o_vqe import vqe_pes, vqe_pes_function
ref_geom = np.loadtxt("ccsd_ref_geom.txt")
ref_params = forward(ref_geom)

def sci_notation(num, sig_figs=2):
    if num == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(num))))
    mantissa = num / (10 ** exponent)
    return f"{mantissa:.{sig_figs-1}f}·10^{exponent}"


pars = []
# Run line-searches between PES combinations
lsis = {}
# shots = [1e5, 1e6, 1e8]
trials = [3]
epsilon_p = [[0.01, 0.01]]
xc_srg = 'b3lyp'
xc_ls = 'VQE'
structure = surrogates[xc_srg].structure.copy()
# end if

surrogates[xc_srg].optimize(epsilon_p=epsilon_p[0])
shots = [1/min(surrogates[xc_srg].sigma_opt)**2]
print(shots)
for shots_count in shots:
    lsis[shots_count] = {}
    for trial in trials:
        pes_ls = vqe_pes_function(shots=shots_count, trials = trial)
        path = f'ls_vqe_shots/shots:{shots_count}reps:{trial}'

        lsi = LineSearchIteration(
            surrogate=surrogates[xc_srg],
            structure=structure,
            path=path,
            pes=pes_ls,
        )
        for i in range(4):
            lsi.propagate(i, add_sigma=False)
        # end for
        # Evaluate the latest eqm structure
        lsi.pls().evaluate_eqm(add_sigma=False)
        print(f'Line-search ({xc_ls}) on vqe with {shots_count} shots and {trial} trials per vqe:')
        print(lsi)
        print(ref_params)
        print('^^Reference params^^')
        lsis[shots_count][trial] = lsi
# end for
# end for


# Plot
if __name__ == '__main__':

    
    param_colors = ['tab:red', 'tab:blue']  # r, θ

    n_params = len(surrogates[xc_srg].structure.params)

    # Two rows: params + energy, columns = len(M_list)
    fig, axs = plt.subplots(2, len(shots), figsize=(4 * len(shots), 5))
    fig.suptitle(f'Line search convergence on {xc_srg} surrogate', fontsize=14)


    for col_idx, shot_count in enumerate(shots):
        eps = epsilon_p[0]  
        eps_str = ''.join(f'_{int(e * 1000):03d}' for e in eps)
        lsi = lsis[shot_count][1]

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
        ax_params.set_title(f'{sci_notation(shot_count)} shots')
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
            label='Energy' if col_idx == 0 else None  # only label once
        )
        ax_energy.set_ylabel('Energy')
        ax_energy.set_xlabel('Step')

        # Legends
        if col_idx == 0:
            ax_params.legend()
        ax_energy.legend()

    plt.tight_layout()
    os.makedirs('figures_vqe_shotnoise', exist_ok=True)
    plt.savefig(f'figures_vqe_shotnoise/vqe_congergences.png')
