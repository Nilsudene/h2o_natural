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


pars = []
# Run line-searches between PES combinations
lsis = {}
shots = [1e5, 1e6, 1e8, 1e12]
trials = [1]
epsilon_p = [0.001, 0.001]
xc_srg = 'b3lyp'
xc_ls = 'VQE'
structure = surrogates[xc_srg].structure.copy()
# end if
for shots_count in shots:
    lsis[shots_count] = {}
    for trial in trials:
        pes_ls = vqe_pes_function(shots=shots_count, trials = trial)
        path = f'ls_vqe_shots/shots:{shots_count}reps:{trial}'

        surrogates[xc_srg].optimize(epsilon_p=epsilon_p)
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
    to_deg = 180 / np.pi
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))  # 4x4 grid
    axs = axs.flatten()
    subplot_idx = 0

    for shots_count in shots:
        for trial in trials:
            lsi = lsis[shots_count][trial]
            ax = axs[subplot_idx]
            subplot_idx += 1

            # Extract and plot the trajectory
            params = [lsi.pls(0).structure.params]
            params_err = [lsi.pls(0).structure.params_err]
            for pls in lsi.pls_list:
                if pls.evaluated:
                    params.append(pls.structure_next.params)
                    params_err.append(pls.structure_next.params_err)

            params = np.array(params)
            params_err = np.array(params_err)

            ax.scatter(
                ref_params[0], ref_params[1] * to_deg,
                s=200,
                c="black",
                label="CCSD optimal geometry"
            )

            """ellipse = patches.Ellipse(
            (params[0], params[1] * to_deg),
            2 * epsilon_p[0],
            2 * epsilon_p[1] * to_deg,
            color=co,
            alpha=0.2
            )
            ax.add_patch(ellipse)"""

            ax.errorbar(
                params[:, 0],
                params[:, 1] * to_deg,
                xerr=params_err[:, 0],
                yerr=params_err[:, 1] * to_deg,
                marker='o',
                linestyle='--',
                color='tab:blue',
            )

            ax.plot(params[-1, 0], params[-1, 1] * to_deg,
                    marker='x',
                    color='red',
                    markersize=10)

            ax.set_title(f"Shots: {int(shots_count)}, Trials: {trial}")
            ax.set_xlabel('Bond length (Å)')
            ax.set_ylabel('Bond angle (deg)')
            ax.grid(True)
            ax.lengend()

            pars.append([params[-1, 0], params[-1, 1]])

    # Hide unused subplots
    for ax in axs[subplot_idx:]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs('figures_vqe_shotnoise', exist_ok=True)
    fig_name = 'vqe_convergence_grid.png'
    plt.savefig(f'figures_vqe_shotnoise/{fig_name}', dpi=300)
    print(f'Saved figure: {fig_name}')
    plt.close(fig)

# --- Print convergence summary ---
pars = np.array(pars)
print("Convergence values:",
      f"{np.mean(pars[:,0]):.4f} ± {np.std(pars[:,0]):.4f} Å,",
      f"{(np.mean(pars[:,1]) * to_deg):.2f} ± {(np.std(pars[:,1]) * to_deg):.2f} deg")