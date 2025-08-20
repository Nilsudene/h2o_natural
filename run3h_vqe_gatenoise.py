#!/usr/bin/env python3

from h2o_vqe import vqe_pes_function
import numpy as np
from matplotlib import pyplot as plt
import os

from stalk import LineSearchIteration
from run2_surrogate import surrogates
from params import forward
ref_geom = np.loadtxt("ccsd_ref_geom.txt")
ref_params = forward(ref_geom)


pars = []
# Run line-searches between PES combinations
lsis = {}
p_dep = [1e-3,1e-4,1e-6]
epsilon_p = [0.005, 0.005]
shots = [1e6]
xc_srg = 'b3lyp'
xc_ls = 'VQE'
structure = surrogates[xc_srg].structure.copy()
# end if
for p in p_dep:
    lsis[p] = {}
    for shot_count in shots:
        pes_ls = vqe_pes_function(p_dep=p, optimizer='COBYLA', shots = shot_count, trials =3)
        path = f'ls_vqe_gatenoise/p_dep:{p}'

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
        print(f'Line-search ({xc_ls}) on B3LYP surrogate with {p} depolarizing noise on cx gates and :')
        print(lsi)
        print(ref_params)
        print('^^Reference params^^')
        lsis[p][shot_count] = lsi
# end for


# plot
if __name__ == '__main__':

    
    param_colors = ['tab:red', 'tab:blue']  # r, θ

    n_params = len(surrogates[xc_srg].structure.params)

    fig, axs = plt.subplots(2, len(p_dep), figsize=(4 * len(p_dep), 5))
    fig.suptitle(f'Line search convergence with depolarizing noise', fontsize=14)


    for col_idx, p in enumerate(p_dep):
        lsi = lsis[p][shot_count]

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

        # --- save data for this p_dep ---
        save_path = f'gatenoise_results.npz'
        run_data = {
            "params": params,
            "params_err": params_err,
            "energies": energies,
            "energies_err": energies_err,
            "ref_params": ref_params,
        }
        if os.path.exists(save_path):
            old = np.load(save_path, allow_pickle=True)
            all_data = dict(old["all"].item())
        else:
            all_data = {}
        all_data[p] = run_data
        np.savez(save_path, all=all_data)

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
        ax_params.set_title(rf'$p_{{dep}}$ = {p}')
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