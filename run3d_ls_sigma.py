#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import os

from stalk import LineSearchIteration

from params import pes_dict, co_dict, b3lyp_sigma_dict, sigmas
from run2_surrogate import surrogates


# Run line-searches between PES combinations
lsis = {}
epsilon_p = [[0.02, 0.02], [0.015, 0.015], [0.01, 0.01], [0.005, 0.005]]

for xc_srg, pes_srg in pes_dict.items():
    lsis[xc_srg] = {}
    xc_ls = 'b3lyp'
    pes_ls = pes_dict[xc_ls]
    structure = surrogates[xc_srg].structure.copy()
    if xc_srg == xc_ls:
        structure.shift_params([0.1, -0.1]) # only shifting if surrogate is the same as linesearch method
    # end if
    eps = epsilon_p[0]
    for sig in sigmas:

        sig_str = f"{sig:.3f}"
        path = f'ls_sigma/{xc_srg}-{xc_ls}-{sig_str}'

        surrogates[xc_srg].optimize(epsilon_p=eps)
        lsi = LineSearchIteration(
            surrogate=surrogates[xc_srg],
            structure=structure,
            path=path,
            pes=b3lyp_sigma_dict[f'b3lyp_sigma_{sig}'],
        )
        for i in range(4):
            lsi.propagate(i, add_sigma=True)
        # end for
        # Evaluate the latest eqm structure
        lsi.pls().evaluate_eqm(add_sigma=True)
        print(f'Line-search ({xc_ls} + {sig_str} std) on {xc_srg} surrogate with {eps} epsilons:')
        print(lsi)
        print(surrogates[xc_ls].structure.params)
        print('^^Reference params^^')
        lsis[xc_srg][sig_str] = lsi
    # end for
# end for


# Plot
if __name__ == '__main__':
    to_deg = 180 / np.pi
    epsilon_colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']

    for xc_srg in pes_dict:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'Line search convergence on {xc_srg} surrogate', fontsize=14)
        axs = axs.flatten()  # Turn 2x2 array into 1D for easy looping

        ref_params = surrogates[xc_ls].structure.params

        for idx, (sig, color) in enumerate(zip(sigmas, epsilon_colors)):
            ax = axs[idx]
            sig_str = f"{sig:.3f}"

            ax.set_title(rf'$\sigma$ = {sig:.3f}')
            ax.scatter(
                ref_params[0], ref_params[1] * to_deg,
                s=200,
                c="black",
                label="B3LYP optimal geometry"
            )
            

            if sig_str not in lsis[xc_srg]:
                ax.set_visible(False)  # hide empty subplot
                continue

            lsi = lsis[xc_srg][sig_str]

            # Plot error ellipse
            ax.add_patch(patches.Ellipse(
                (ref_params[0], ref_params[1] * to_deg),
                2 * eps[0],
                2 * eps[1] * to_deg,
                edgecolor=color,
                facecolor='none',
                linestyle='-',
                linewidth=1.5,
            ))

            # Line search trajectory
            params = [lsi.pls(0).structure.params]
            params_err = [lsi.pls(0).structure.params_err]
            for pls in lsi.pls_list:
                if pls.evaluated:
                    params.append(pls.structure_next.params)
                    params_err.append(pls.structure_next.params_err)
            params = np.array(params)
            params_err = np.array(params_err)

            ax.errorbar(
                params[:, 0],
                params[:, 1] * to_deg,
                xerr=params_err[:, 0],
                yerr=params_err[:, 1] * to_deg,
                marker='.',
                linestyle=':',
                color=color,
            )

            # Final point
            ax.plot(params[-1, 0], params[-1, 1] * to_deg,
                    marker='x',
                    color=color,
                    markersize=10)

            ax.set_xlabel('Bond length (Å)')
            ax.set_ylabel('Bond angle (deg)')
            ax.grid(True)

        # Hide any unused subplots
        for i in range(len(sigmas), 4):
            axs[i].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
        
        os.makedirs('figures_sigma', exist_ok=True)
        
        fig_name = f'{xc_srg}_convergence.png'
        plt.savefig(f'figures_sigma/{fig_name}', dpi=300)
        print(f'Saved figure: {fig_name}')
        plt.close(fig)
