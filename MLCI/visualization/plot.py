import pandas as pd
import corner
from ..models import model as model_new
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import accuracy_score
from scipy.stats import gaussian_kde
sns.set_theme(style="ticks", palette="deep")
from astropy.cosmology import Planck18
import pandas as pd

sim_catalog = model_new.sim_catalog
output_para_name = model_new.output_para_name
col_rangs = {'R500_kpc':[1e-2, 2e3], 'Mgas500':[1e-2, 1e3], 'Lbol500':[1e-2, 2e3], 'T500':[1e-2, 3e1], 'z':[0, 1.2]}
labels = ['eFEDS', 'DR1', 'Simu']
units = ['kpc', '$10^{12} M_{sun}$', '$10^{42} erg s^{-1}$', 'keV', '']
colors = ['red', 'green', 'blue']



def plot_distribution(dataset, labels=labels, col_rangs=col_rangs, fname=''):
    vars = col_rangs.keys()
    xrangs = col_rangs.values()
    f_size = 14
    fig, axs = plt.subplots(1, 5, figsize=(20, 5), dpi=160)
    for var, ax, xrang in zip(vars, axs, xrangs):
        bw = (xrang[1] - xrang[0])/20
        for label, data, color in zip(labels, dataset, colors):
            label = label
            sns.histplot(data[var], ax=ax, kde=False, label=label, palette='dark', color=color, stat="density", alpha=0.4, binwidth=bw)
            ax.set_yscale('log')
        ax.set_xlabel(data[var].unit)
        x0, x1 = np.round(xrang[0],1), np.round(xrang[1],1)
        ax.legend(title=f'Density of {var} \nRange: {x0} -- {x1}',  framealpha=0.1, fontsize=f_size) #
        ax.set_ylabel('')
    axs[0].set_ylabel('Density', fontsize=f_size)
    plt.xticks(fontsize=f_size)
    plt.yticks(fontsize=f_size)
    if fname == '':
        fname = f'./figures/{fname}_distr.pdf'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_confusion_matrix(results, title='test', fname='', rerange='', normalize=True):
    prob_cols = [col for col in results.columns if col.startswith('prob_')]
    simulation_names = [col.replace('prob_', '') for col in prob_cols]

    probs = results[prob_cols]
    pred_index = np.argmax(probs, axis=-1) # find the largets prob index
    pred_label = np.array(simulation_names)[pred_index] # get the corresbonding label
    ture_label = results['label'].values # ture label
    if normalize:
        #confusion_matrix = pd.crosstab(ture_label, pred_label, normalize='index', rownames=['Actual'], colnames=['Predict'])
        confusion_matrix = pd.crosstab(pred_label, ture_label, normalize='index', rownames=['Predict'], colnames=['Actual'])
    else:
        confusion_matrix = pd.crosstab(pred_label, ture_label, rownames=['Predict'], colnames=['Actual'])
    if rerange != '':
        temp_sim = sim_catalog[sim_catalog['name'].isin(simulation_names)]
        temp_sim = temp_sim.sort_values(by=rerange, ascending=True)
        print(rerange, temp_sim[rerange])
        new_label = temp_sim['name'].values
        confusion_matrix = confusion_matrix.reindex(index=new_label, columns=new_label)
    else:
        confusion_matrix = confusion_matrix.reindex(index=simulation_names, columns=simulation_names)
    plt.figure(figsize=(8,8),dpi=160)
    sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    accuracy = accuracy_score(ture_label, pred_label)
    plt.title(f'{title} testing, accuracy = {accuracy:.3f}')
    if fname == '':
        fname = f'figures/CM_{title}.pdf'
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def plot_corner(results, ref_point, test_set, fname=f'./figures/test_corner.pdf', model_name='RF', show_cos_name=True, show_pdf=True, show_datapoint=True, smooth=0.5, truncation=False):
    
    prob_cols = [col for col in results.columns if col.startswith('prob_')]
    simulation_names = [col.replace('prob_', '') for col in prob_cols]
    #=== truncation
    if truncation:
        omegam_max = 0.35
        width = 0.20
        masks = (results['Omega'] >= omegam_max)
        results['weight'][masks] = results['weight'][masks]*np.exp(-((results['weight'][masks]-omegam_max)/width)**2)
        #masks = (results['z'] < 0.10)
        #results['weight'][masks] = 0
    
    samples, weights, prob = results[output_para_name], results['weight'], results[prob_cols]
    output_labels = output_para_name
    fontsize = {"fontsize": 12}
    contour_args = {'colors': 'blue', 'linestyles': 'dashed', 'linewidths': 2}
    label_names = ['$\Omega_m$', '$\sigma_8$', '$h_0$', '$\Omega_B$']
    para_range = [(0.16, 0.48), (0.45, 1.05), (0.60, 0.80), (0.03, 0.06)]
    #para_range = [(0.16, 0.45), (0.5, 1.0), (0.65, 0.75), (0.04, 0.05)]
    truth_kwargs = {'linewidth': 2, 'linestyle': '--'}

    
    if ref_point == 'None':
        truths=None
    elif ref_point == 'Planck2018':
        # see https://arxiv.org/abs/1807.06209 Table 2
        sigma8, h0 = 0.8102, 67.66/100
        OmegaB = round(0.02242/h0**2,  4)
        #planck2018 = pd.DataFrame([{'Omega':0.3111, 'Sigm8':sigma8, 'Hubble':h0, 'OmegaB':OmegaB}])
        planck2018 = pd.DataFrame([{'Omega':Planck18.Om0, 'Sigm8':sigma8, 'Hubble':Planck18.h, 'OmegaB':Planck18.Ob0}])
        ref_para = planck2018[output_labels]
        print(ref_para)
        truths=ref_para.values[0]
    else:
        ref_para = sim_catalog[sim_catalog['name']==ref_point][output_labels]
        print(ref_para)
        truths=ref_para.values[0]
    

    figure, axs = plt.subplots(len(output_labels), len(output_labels), figsize=(8, 8), dpi=160)
    corner.corner(      samples, 
                        weights= weights,
                        labels=label_names,
                        contour_args = contour_args,
                        label_kwargs=fontsize,
                        truths=truths,
                        levels=[0.68, 0.95], #contour 1sigma, 2sigma
                        quantiles=[0.16, 0.5, 0.84], # 1sigma
                        bins=30,
                        smooth=smooth,
                        title_fmt='.3f',
                        title_kwargs=fontsize,
                        range=para_range,
                        show_titles=True,
                        plot_datapoints=show_datapoint,
                        hist_kwargs={'histtype': 'stepfilled'},
                        truth_kwargs = truth_kwargs,
                        fig=figure,
                        )
    corner.overplot_points(figure, sim_catalog[output_labels].values, color='blue')
    if ref_point == 'None':
        show_text = f'Input: {test_set}\n Output: {", ".join(label_names)}\n Model name: {model_name}\n'
    elif ref_point == "Planck2018":
        show_text = f'Reference: {ref_point}\n Input:{test_set}'
    else:
        show_text = f'Reference: {ref_point}\n Input:{test_set}\n{ref_para.to_string(index=False)}\n{model_name}'
    figure.axes[2].text(0.5, 0.5, show_text,fontsize=14,ha='center',va='center')
    # add simulation lables
    label_size = 12
    ax_indx = [(1,0), (2,0), (2,1), (3,0), (3,1), (3,2)]
    var_name= [('Omega','Sigm8'),('Omega','Hubble'),('Sigm8','Hubble'),('Omega','OmegaB'),('Sigm8','OmegaB'),('Hubble','OmegaB')]
    
    if show_cos_name:
        for i, idx in enumerate(ax_indx):
            labels = sim_catalog['name'].values
            x = sim_catalog[var_name[i][0]].values
            y = sim_catalog[var_name[i][1]].values
            for j, label in enumerate(labels):
                axs[idx].text(x[j], y[j], label, fontsize=label_size, ha='right', va='top', color='blue',alpha=1.0)
    
    if show_pdf:
        # add the probabilities pannel
        yshift = 0.2
        for idx, ax in enumerate(axs.flat):
            current_position = ax.get_position()  # get the subfigure position
            width, height = current_position.x1-current_position.x0, current_position.y1-current_position.y0
            new_position = [current_position.x0, current_position.y0+yshift, width, height]  #x, y position, 'width' and 'height'
            ax.set_position(new_position)
        gs = GridSpec(10, 1, figure=figure)  # 1 row GridSpec object
        ax_bottom = figure.add_subplot(gs[-1, :])  # create a sub-figure
        mean_prob = []
        for label in simulation_names:
            mean_prob.append(np.sum(prob[f'prob_{label}']/len(prob[f'prob_{label}'])))
        mean_prob = pd.DataFrame(data=[mean_prob], index=['mean_P'], columns=simulation_names)
        sns.heatmap(mean_prob, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=ax_bottom)
        plt.title('mean PDF')

    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def get_expectation_1sigam(results, model_name, catalog_name):
    samples, weights = results[output_para_name], results['weight']
    quantiles = [0.16, 0.5, 0.84]
    final_res = {'Model Name':model_name, 'Catalog':catalog_name}
    for i in range(len(output_para_name)):
        values = corner.quantile(samples.values[:, i], quantiles, weights=weights)
        medians = values[1]  # 50th 百分位
        lower_errors = values[1] - values[0]  
        upper_errors = values[2] - values[1]
        final_res[output_para_name[i]+'_medians'] = medians
        final_res[output_para_name[i]+'_lower_errors'] = lower_errors
        final_res[output_para_name[i]+'_upper_errors'] = upper_errors
    print('constrain:', final_res)
    return final_res


def scalling_relation(xlabel, ylabel, simulated_data, observed_data, simulation_label, percent=16):
    plt.figure(figsize=(6, 4))
    data = simulated_data[simulated_data['label'] == simulation_label]
    x = data[xlabel].data
    y = data[ylabel].data
    sns.scatterplot(x=x, y=y, s=2, color='blue', label=f'Simulated = {simulation_label}')
    sns.kdeplot(x=x, y=y, levels=[0.16, 0.50, 0.84], color='black', linewidths=1)
    x = observed_data[xlabel].data
    y = observed_data[ylabel].data
    xy = np.vstack([x, y]) # 计算 2D 核密度估计值
    kde = gaussian_kde(xy)
    z = kde(xy)  # 每个点的密度值
    # seaborn 的 kdeplot 默认是密度累积分布等值线，可用百分位估计
    z_threshold = np.percentile(z, percent) 
    mask_inside = z >= z_threshold  # 位于高密度区域内的点
    x_selected = x[mask_inside]
    y_selected = y[mask_inside]
    sns.kdeplot(x=x, y=y, levels=[0.16, 0.50, 0.84], color='red', linewidths=1)
    sns.scatterplot(x=x_selected, y=y_selected, s=2, color='green', label='(eFEDS+DR1)')
    plt.xlabel(data[xlabel].unit)
    plt.ylabel(data[ylabel].unit)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(f'./figures/{xlabel}_{ylabel}scaling_relation.pdf')



def corner_plot(xlabels, simulated_data, observed_data, simulation_label):
    n = len(xlabels)
    fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i < j:
                ax.axis('off')  # 只绘制下三角
                continue
            elif i == j:
                # 对角线可以画 histogram 或 KDE
                x_sim = simulated_data[simulated_data['label'] == simulation_label][xlabels[i]]
                x_obs = observed_data[xlabels[i]]
                bins = np.histogram_bin_edges(np.concatenate([x_sim, x_obs]), bins='auto')
                sns.histplot(x_sim, ax=ax, kde=False, label=f'Simulated={simulation_label}', palette='dark', color='red', stat="density", alpha=0.4, bins=bins)
                sns.histplot(x_obs, ax=ax, kde=False, label='(eFEDS+DR1)', palette='dark', color='blue', stat="density", alpha=0.4, bins=bins)
                plt.legend()
                ax.set_yscale('log')
                if j == 0:
                    #ax.set_ylabel('density')
                    pass

            else:
                xlabel = xlabels[j]
                ylabel = xlabels[i]
                x_sim = simulated_data[simulated_data['label']==simulation_label][xlabel].data
                y_sim = simulated_data[simulated_data['label']==simulation_label][ylabel].data
                sns.scatterplot(x=x_sim, y=y_sim, s=2, color='red', ax=ax)
                sns.kdeplot(x=x_sim, y=y_sim, ax=ax, levels=[0.16, 0.50, 0.84], color='red', linewidths=1, label=f'Simulated={simulation_label}')
                x_obs = observed_data[xlabel].data
                y_obs = observed_data[ylabel].data
                sns.scatterplot(x=x_obs, y=y_obs, s=2, color='blue', ax=ax)
                sns.kdeplot(x=x_obs, y=y_obs, ax=ax, levels=[0.16, 0.50, 0.84], color='blue', linewidths=1,  label='(eFEDS+DR1)')
                if j == 0:
                    ax.set_ylabel(simulated_data[ylabel].unit)
                if i == (n-1):
                    ax.set_xlabel(simulated_data[xlabel].unit)
                plt.legend()      
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    #plt.tight_layout()
    plt.savefig('./figures/custom_corner_plot.pdf')
    plt.show()


def plot_detection_prob(observed_data, fname=''):
    L_vals = np.logspace(41, 46.0, 100)  # 光度范围
    z_vals = np.linspace(0.01, 1.4, 100)
    # 构建二维网格
    L_grid, z_grid = np.meshgrid(L_vals, z_vals)
    S_grid = model_new.selection_function(np.log10(L_grid), z_grid)
    plt.figure(figsize=(8, 5))
    contour = plt.contourf(z_grid, np.log10(L_grid), S_grid, levels=50, cmap='viridis')
    cbar = plt.colorbar(contour)
    cbar.set_label('probability')
    plt.scatter(observed_data['z'], observed_data['L'], color='red', s=5, marker='x', label='eFEDS+DR1')
    plt.ylabel('Log10(L / erg s-1)')
    plt.xlabel('$z$')
    plt.xlim(0, 1.4)
    plt.ylim(41, 46)
    plt.title('Selection Function $S(L, z)$')
    plt.legend()
    plt.tight_layout()
    if fname == '':
        fname = f'./figures/test_detection_prob.pdf'
    plt.savefig(fname)



def plot_cosmology_paras_vs_z(bins_cosmology, aveg_cosmology, fname='./figures/test_Cosmological_Parameters_vs_z.pdf'):
    para_names = ['Omega', 'Sigm8', 'Hubble', 'OmegaB']
    label_names = ['$\Omega_m$', '$\sigma_8$', '$h_0$', '$\Omega_B$']
    plt.figure(figsize=(8, 5))

    xoffsets = np.linspace(-0.02, 0.02, len(para_names)) 
    for para_name, label, x_off in zip(para_names, label_names, xoffsets):
        x = bins_cosmology['z'].data + x_off
        y = bins_cosmology[f'{para_name}_medians'].data
        yerr = np.array([bins_cosmology[f'{para_name}_lower_errors'], bins_cosmology[f'{para_name}_upper_errors']])
        norm_value = aveg_cosmology[f'{para_name}_medians']
        plt.errorbar(x, y/norm_value, yerr=yerr/norm_value, fmt='o', capsize=4, label=label)

    plt.legend()
    plt.axhline(y=1, color='red', linestyle='--', linewidth=1)
    plt.xlabel('$z$')
    plt.ylabel('Bin Meadian / Meadian')
    plt.title('Cosmological Parameters vs Redshift')
    plt.legend(title='Parameters')
    plt.tight_layout()
    plt.savefig(fname)