import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm


from params import pes_dict, co_dict, b3lyp_sigma_dict
from run2_surrogate import surrogates

data = np.loadtxt('ls_hist/final_params.txt')

epsilon_p = [[0.01, 0.01], [0.02, 0.02]]
sigmas = [0.0, "max sigma"]

xc_ls = 'b3lyp'

i = 32
if __name__ == '__main__':
    to_deg = 180 / np.pi
    epsilon_colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']

    for xc_srg in pes_dict:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'100 line search convergences on {xc_srg} surrogate', fontsize=14)

        ref_params = surrogates[xc_ls].structure.params

        for id1, eps in enumerate(epsilon_p):
            eps_str = f"{eps[0]:.3f}"
            for id2, sig in enumerate(sigmas):
                if sig == 'max sigma':
                    sig_str = 'max sigma'
                else:
                    sig_str = f"{sig:.4f}"

                axs[id1,id2].set_title(rf'$\epsilon$: {eps_str}, $\sigma$: {sig_str}', fontsize=12)
                par0_res = data[i*100:(i+1)*100, 1]
                par1_res = data[i*100:(i+1)*100, 2]

                # histograms of the runs
                axs[id1,id2].hist(par0_res, bins=10, alpha=0.5, label='Parameter 0', color='tab:blue')
                axs[id1,id2].hist(par1_res, bins=10, alpha=0.5, label='Parameter 1', color='tab:orange')

                # b3lyp target results
                axs[id1,id2].axvline(ref_params[0], color='tab:blue', linestyle='--', label='Ref Param 0')
                axs[id1,id2].axvline(ref_params[1], color='tab:orange', linestyle='--', label='Ref Param 1')

                # gaussian fit to the histograms
                mu0, std0 = norm.fit(par0_res)
                mu1, std1 = norm.fit(par1_res)
                x0 = np.linspace(min(par0_res), max(par0_res), 100)
                x1 = np.linspace(min(par1_res), max(par1_res), 100)
                axs[id1,id2].plot(x0, norm.pdf(x0, mu0, std0), color='tab:blue', label=f'std = {std0:.3f}, bias = {mu0 - ref_params[0]:.3f}')
                axs[id1,id2].plot(x1, norm.pdf(x1, mu1, std1), color='tab:orange', label=f'std = {std1:.3f}, bias = {mu1 - ref_params[1]:.3f}')

                # plotting stuff
                axs[id1,id2].set_xlabel('Parameter Value')
                axs[id1,id2].set_ylabel('Frequency')
                axs[id1,id2].legend()
                axs[id1,id2].grid(True)
                i += 1
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
                
    