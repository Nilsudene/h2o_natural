#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import os

from stalk import LineSearchIteration

from params import pes_dict, co_dict
from run2_surrogate import surrogates
from h2o_vqe import vqe_pes

pars = []
# Run line-searches between PES combinations
lsis = {}
epsilon_p = [[0.005, 0.005], [0.001, 0.001]]
for xc_srg, pes_srg in pes_dict.items():
    lsis[xc_srg] = {}
    xc_ls = 'VQE'
    pes_ls = vqe_pes
    structure = surrogates[xc_srg].structure.copy()
    # end if
    for eps in epsilon_p:

        eps_str = ''.join(f'_{int(e * 1000):03d}' for e in eps)
        path = f'ls_vqe/{xc_srg}-{xc_ls}-{eps_str}'

        surrogates[xc_srg].optimize(epsilon_p=eps)
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
        print(f'Line-search ({xc_ls}) on {xc_srg} surrogate with {eps_str} epsilons:')
        print(lsi)
        lsis[xc_srg][eps_str] = lsi
    # end for
# end for


# Plot
if __name__ == '__main__':
    to_deg = 180 / np.pi
    epsilon_colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']

    for xc_srg in pes_dict:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()  # Turn 2x2 array into 1D for easy looping

        

        for idx, (eps, color) in enumerate(zip(epsilon_p, epsilon_colors)):
            ax = axs[idx]
            eps_str = ''.join(f'_{int(e * 1000):03d}' for e in eps)


            if eps_str not in lsis[xc_srg]:
                ax.set_visible(False)  # hide empty subplot
                continue

            lsi = lsis[xc_srg][eps_str]


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
            pars.append([params[-1, 0], params[-1, 1]])
            ax.plot(params[-1, 0], params[-1, 1] * to_deg,
                    marker='x',
                    color=color,
                    markersize=10)

            ax.set_xlabel('Bond length (Å)')
            ax.set_ylabel('Bond angle (deg)')
            ax.grid(True)

        # Hide any unused subplots
        for i in range(len(epsilon_p), 4):
            axs[i].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
        
        os.makedirs('figures_vqe', exist_ok=True)
        
        fig_name = f'{xc_srg}_convergence.png'
        plt.savefig(f'figures_vqe/{fig_name}', dpi=300)
        print(f'Saved figure: {fig_name}')
        plt.close(fig)
pars = np.array(pars)
print("Convergence values:", np.mean(pars[:,0]), "pm", np.std(pars[:,0]), ",",np.mean(pars[:,1]), "pm", np.std(pars[:,1]) )
