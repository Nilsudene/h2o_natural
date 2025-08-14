#!/usr/bin/env python3

from h2o_vqe import vqe_pes_function
import numpy as np
from matplotlib import pyplot as plt
import os

from stalk import LineSearchIteration
from run2_surrogate import surrogates


pars = []
# Run line-searches between PES combinations
lsis = {}
p_dep = [1e-6, 1e-5,1e-4,1e-3]
epsilon_p = [0.001, 0.001]
xc_srg = 'b3lyp'
xc_ls = 'VQE'
structure = surrogates[xc_srg].structure.copy()
# end if
for p in p_dep:
    lsis[p] = {}
    pes_ls = vqe_pes_function(p_dep=p)
    path = f'ls_vqe_gatenoise/p_dep:{p}'

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
    print(f'Line-search ({xc_ls}) on B3LYP surrogate with {p} depolarizing noise on cx gates:')
    print(lsi)
    lsis[p] = lsi
# end for

if __name__ == '__main__':
    to_deg = 180 / np.pi
    os.makedirs('figures_vqe', exist_ok=True)

    pars = []

    for p in p_dep:
        lsi = lsis[p]

        fig, ax = plt.subplots(figsize=(6, 6))

        # Extract and plot the trajectory
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
            marker='o',
            linestyle='--',
            color='tab:blue',
        )

        ax.plot(params[-1, 0], params[-1, 1] * to_deg,
                marker='x',
                color='red',
                markersize=10)

        ax.set_title(f"VQE convergence\np_dep = {p}")
        ax.set_xlabel('Bond length (Å)')
        ax.set_ylabel('Bond angle (deg)')
        ax.grid(True)

        pars.append([params[-1, 0], params[-1, 1]])

        fig_name = f'vqe_convergence_pdep_{p}.png'
        plt.tight_layout()
        plt.savefig(f'figures_vqe/{fig_name}', dpi=300)
        print(f'Saved figure: {fig_name}')
        plt.close(fig)

    # --- Print convergence summary ---
    pars = np.array(pars)
    print("Convergence values:",
          f"{np.mean(pars[:,0]):.4f} ± {np.std(pars[:,0]):.4f} Å,",
          f"{(np.mean(pars[:,1]) * to_deg):.2f} ± {(np.std(pars[:,1]) * to_deg):.2f} deg")
