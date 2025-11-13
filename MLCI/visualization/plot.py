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
from astropy.table import Table 
from ..data_processing import preprocess

sim_catalog = model_new.sim_catalog
output_para_name = model_new.output_para_name
col_rangs = {'R500_kpc':[1e-2, 2e3], 'Mgas500':[1e-2, 1e3], 'Lbol500':[1e-2, 2e3], 'T500':[1e-2, 3e1], 'z':[0, 1.2]}
labels = ['eFEDS', 'DR1', 'Simu']
units = ['kpc', '$10^{12} M_{sun}$', '$10^{42} erg s^{-1}$', 'keV', '']
colors = ['red', 'green', 'blue']

# https://ui.adsabs.harvard.edu/abs/2020A%26A...641A...6P/abstract
# Omegam = 0.315 ± 0.007
# h0 = (67.4 ± 0.5) km s-1 Mpc-1
# σ8 = 0.811 ± 0.006
# Ωbh2 = 0.0224 ± 0.0001,
h0 = (67.4/100)
planck18_cosmology = {'Omega_medians':0.315, 'Omega_lower_errors':0.007, 'Omega_upper_errors':0.007,
                    'Sigm8_medians':0.811, 'Sigm8_lower_errors':0.006, 'Sigm8_upper_errors':0.006,
                    'Hubble_medians':67.4/100, 'Hubble_lower_errors':0.5/100, 'Hubble_upper_errors':0.5/100,
                    'OmegaB_medians':0.0224/h0**2, 'OmegaB_lower_errors':0.0001/h0**2, 'OmegaB_upper_errors':0.0001/h0**2}



def plot_distribution(dataset, labels=labels, col_rangs=col_rangs, fname=''):
    vars = col_rangs.keys()
    xrangs = col_rangs.values()
    f_size = 14
    fig, axs = plt.subplots(1, len(col_rangs.keys()), figsize=(4*len(col_rangs.keys()), 5), dpi=160)
    for var, ax, xrang in zip(vars, axs, xrangs):
        bw = (xrang[1] - xrang[0])/20
        for label, data, color in zip(labels, dataset, colors):
            label = label
            sns.histplot(data[var], ax=ax, kde=False, label=label, palette='dark', color=color, stat="density", alpha=0.4, binwidth=bw)
            ax.set_yscale('log')
        if hasattr(data[var], 'unit'):
            ax.set_xlabel(str(data[var].unit))
        else:
            ax.set_xlabel(var)
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
    csv_name = fname.replace('.pdf', '.csv')   # 同名 CSV 文件
    confusion_matrix.to_csv(csv_name, float_format='%.6f')
    diag_values = confusion_matrix.values.diagonal()
    print(fname, ' Diagonal values:', diag_values)
    return confusion_matrix.values



def plot_corner(results, ref_point, test_set, fname=f'./figures/test_corner.pdf', model_name='RF', show_cos_name=True, show_pdf=True, show_datapoint=True, smooth=0.5, truncation=False, selction=True):
    
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
    
    if selction:
        samples, weights, prob = results[output_para_name], results['weight'], results[prob_cols]
    else:
        samples, weights, prob = results[output_para_name], results['weight'], results[prob_cols]
        weights = weights*results['detect_prob']
        # weights = np.ones_like(weights)
    output_labels = output_para_name
    fontsize = {"fontsize": 12}
    contour_args = {'colors': 'blue', 'linestyles': 'dashed', 'linewidths': 2}
    label_names = ['$\Omega_m$', '$\sigma_8$', '$h_0$', '$\Omega_B$']
    para_range = [(0.16, 0.48), (0.45, 1.05), (0.60, 0.80), (0.03, 0.06)]
    para_range = [(0.20, 0.45), (0.5, 1.0), (0.65, 0.75), (0.04, 0.05)]
    truth_kwargs = {'linewidth': 2, 'linestyle': '--'}

    
    if ref_point == 'None':
        truths=None
    elif ref_point == 'Planck2018':
        # see https://arxiv.org/abs/1807.06209 abstract
        sigma8, h0 = 0.8102, 67.4/100
        OmegaB = round(0.0224/(h0**2),  4)
        planck2018 = pd.DataFrame([{'Omega':0.315, 'Sigm8':sigma8, 'Hubble':h0, 'OmegaB':OmegaB}])
        #planck2018 = pd.DataFrame([{'Omega':Planck18.Om0, 'Sigm8':sigma8, 'Hubble':Planck18.h, 'OmegaB':Planck18.Ob0}])
        ref_para = planck2018[output_labels]
        print(ref_para)
        truths=ref_para.values[0]
        truths=None
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
    corner.overplot_points(figure, sim_catalog[output_labels].values, color='blue', alpha=0.5, markersize=10,)
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
    
    if ref_point == "Planck2018":
        for i, idx in enumerate(ax_indx):
            x = planck18_cosmology[f'{var_name[i][0]}_medians'], 
            y = planck18_cosmology[f'{var_name[i][1]}_medians']
            xerr = [[planck18_cosmology[f'{var_name[i][0]}_lower_errors']], [planck18_cosmology[f'{var_name[i][0]}_upper_errors']]]
            yerr = [[planck18_cosmology[f'{var_name[i][1]}_lower_errors']], [planck18_cosmology[f'{var_name[i][1]}_upper_errors']]]
            axs[idx].errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", markersize=0, color="red", capsize=3)
        
        col_names = planck2018.keys()
        for i in range(len(col_names)):
            x = planck18_cosmology[f'{col_names[i]}_medians']
            xlower = x-planck18_cosmology[f'{col_names[i]}_lower_errors']
            xupper = x+planck18_cosmology[f'{col_names[i]}_upper_errors']
            axs[i,i].axvline(x=xlower, color='red', linestyle='--', linewidth=1)
            axs[i,i].axvline(x=xupper, color='red', linestyle='--', linewidth=1)
            y_min, y_max = axs[i,i].get_ylim()
            axs[i,i].fill_betweenx([y_min, y_max], xlower, xupper, color='red', alpha=0.2, label='Planck2018, 1$\sigma$')

    
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
        plt.legend()
        plt.title('mean PDF')

    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def get_expectation_1sigam(results, model_name, catalog_name, selection_fun = True):
    if selection_fun:
        samples, weights = results[output_para_name], results['weight']
    else:
        samples = results[output_para_name]
        weights = results['weight']*results['detect_prob']
    quantiles = [0.16, 0.50, 0.84]
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



def corner_plot(xlabels, simulated_data, observed_data, simulation_label=None, fname='./figures/custom_corner_plot.pdf'):
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
                if simulation_label is not None:
                    x_sim = simulated_data[simulated_data['label'] == simulation_label][xlabels[i]]
                else:
                    x_sim = simulated_data[xlabels[i]]
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
                if simulation_label is not None:
                    x_sim = simulated_data[simulated_data['label']==simulation_label][xlabel].data
                    y_sim = simulated_data[simulated_data['label']==simulation_label][ylabel].data
                else:
                    x_sim = simulated_data[xlabel].data
                    y_sim = simulated_data[ylabel].data
                sns.scatterplot(x=x_sim, y=y_sim, s=2, color='red', ax=ax)
                sns.kdeplot(x=x_sim, y=y_sim, ax=ax, levels=[0.16, 0.50, 0.84], color='red', linewidths=1, label=f'Simulated={simulation_label}')
                x_obs = observed_data[xlabel].data
                y_obs = observed_data[ylabel].data
                sns.scatterplot(x=x_obs, y=y_obs, s=2, color='blue', ax=ax)
                sns.kdeplot(x=x_obs, y=y_obs, ax=ax, levels=[0.16, 0.50, 0.84], color='blue', linewidths=1,  label='(eFEDS+DR1)')
                if j == 0:
                    #ax.set_ylabel(simulated_data[ylabel].unit)
                    ax.set_ylabel(ylabel)
                if i == (n-1):
                    #ax.set_xlabel(simulated_data[xlabel].unit)
                    ax.set_xlabel(xlabel)
                plt.legend()      
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    #plt.tight_layout()
    plt.savefig(fname)
    plt.show()


def plot_detection_prob(dataset, labels, colors, fname=''):
    L_vals = np.logspace(41, 46.0, 100)  # 光度范围
    z_vals = np.linspace(0.01, 1.4, 100)
    # 构建二维网格
    L_grid, z_grid = np.meshgrid(L_vals, z_vals)
    S_grid = model_new.selection_function(np.log10(L_grid), z_grid)
    plt.figure(figsize=(8, 5))
    contour = plt.contour(z_grid, np.log10(L_grid), S_grid, levels=[0.1, 0.3, 0.5, 0.7, 0.9], alpha=1.0, linestyles='-.', colors='black', linewidths=0.6,)
    plt.clabel(contour, fmt='%.2f')
    #contour = plt.contourf(z_grid, np.log10(L_grid), S_grid, levels=10, cmap='viridis')
    #cbar = plt.colorbar(contour)
    #cbar.set_label('probability')
    for data, label, color in zip(dataset, labels, colors):
        plt.scatter(data['z'], data['L'], color=color, s=2, marker='+', label=label)
    data = dataset[0]
    sns.kdeplot(x=data['z'],y=data['L'], levels=[0.16, 0.50, 0.84],  linewidths=0.6, linestyles='-', fill=False, colors='black')
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

def plot_detection_prob_and_cosmology_vs_Z():
    pass


def obtain_binned_cosmology(model_name, z_bins_edge=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], simulation=False, simulation_label=None):
    if simulation:
        results = pd.read_csv(f"./results/{model_name}/mix_simulations.csv")
        results = results[results['label'] == simulation_label]
    else:
        results = pd.read_csv(f"./results/{model_name}/observation.csv")
    z_0, z_1 = z_bins_edge[0], z_bins_edge[-1]
    results = results[(results['z'] >= z_0) & (results['z'] < z_1)]
    # plot the cosmological parameters vs redshift
    bins_cosmology = []
    z_bins_edge = np.array(z_bins_edge)
    aveg_cosmology = get_expectation_1sigam(results, model_name, model_name)
    for i in range(len(z_bins_edge)-1):
        subset = results[(results['z'] >= z_bins_edge[i]) & (results['z'] < z_bins_edge[i+1])]
        if len(subset) <= 0:
            continue
        print(f"redshift: {z_bins_edge[i]} - {z_bins_edge[i+1]}, num={len(subset)}")
        cosmology_para = get_expectation_1sigam(subset, model_name, model_name)
        cosmology_para['z'] = (z_bins_edge[i] + z_bins_edge[i+1])/2
        bins_cosmology.append(cosmology_para)
    aveg_cosmology['z'] = z_bins_edge
    return Table(bins_cosmology), aveg_cosmology


def plot_cosmology_paras_vs_z(model_name, z_bins_edge, fname='./figures/test_Cosmological_Parameters_vs_z.pdf'):
    bins_cosmology, aveg_cosmology = obtain_binned_cosmology(model_name, z_bins_edge)
    para_names = ['Omega', 'Sigm8', 'Hubble', 'OmegaB']
    label_names = ['$\Omega_m$', '$\sigma_8$', '$h_0$', '$\Omega_B$']
    colors = ['blue', 'orange', 'green', 'red']
    plt.figure(figsize=(8, 5))
    plt.tick_params(axis='both', which='major', labelsize=16)  
    plt.tick_params(axis='both', which='minor', labelsize=16)
    xoffsets = np.linspace(-0.02, 0.02, len(para_names))
    '''
    for para_name, label, x_off in zip(para_names, label_names, xoffsets):
        x = bins_cosmology['z'].data + x_off
        y = bins_cosmology[f'{para_name}_medians'].data
        yerr = np.array([bins_cosmology[f'{para_name}_lower_errors'], bins_cosmology[f'{para_name}_upper_errors']])
        norm_value = aveg_cosmology[f'{para_name}_medians']
        plt.errorbar(x, y/norm_value, yerr=yerr/norm_value, fmt='o', capsize=4, label=label)
    '''
    for para_name, label, x_off, color in zip(para_names, label_names, xoffsets, colors):
        x = bins_cosmology['z'].data + x_off
        y_uperr = bins_cosmology[f'{para_name}_upper_errors']/aveg_cosmology[f'{para_name}_upper_errors']
        y_lowerr = bins_cosmology[f'{para_name}_lower_errors']/aveg_cosmology[f'{para_name}_lower_errors']
        diff = bins_cosmology[f'{para_name}_medians'] - aveg_cosmology[f'{para_name}_medians']
        norm_err = np.where(diff > 0, aveg_cosmology[f'{para_name}_upper_errors'], aveg_cosmology[f'{para_name}_lower_errors'])
        y = diff / norm_err
        plt.errorbar(x, y, yerr=[y_lowerr, y_uperr], fmt='o', capsize=4, label=label, color=color)
    plt.axhline(y=0.0, color='black', linestyle=':', linewidth=1)
    plt.axhline(y=-1.0, color='black', linestyle=':', linewidth=1)
    plt.axhline(y=1.0, color='black', linestyle=':', linewidth=1)
    plt.xlabel('$z$', fontsize=18)
    plt.ylabel('bias (error) / 1$\sigma$', fontsize=18)
    plt.title('Cosmological Parameters vs Redshift',fontsize=18)
    plt.legend(title='Parameters',fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.savefig(fname)



def plot_cosmology_paras_vs_z_simulation(model_name, simulation_label, z_bins_edge, fname='./figures/test_Cosmological_Parameters_vs_z_simulation.pdf'):
    bins_cosmology, aveg_cosmology = obtain_binned_cosmology(model_name, z_bins_edge)
    para_names = ['Omega', 'Sigm8', 'Hubble', 'OmegaB']
    label_names = ['$\Omega_m$', '$\sigma_8$', '$h_0$', '$\Omega_B$']
    colors = ['blue', 'orange', 'green', 'red']

    model_name1 = 'RFtest41'
    results1 = pd.read_csv(f"./results/{model_name1}/observation.csv")
    model_name2 = 'RFtest40'
    results2 = pd.read_csv(f"./results/{model_name2}/observation.csv")

    redshift_bin = z_bins_edge
    z_0, z_1 = redshift_bin[0], redshift_bin[-1]
    results_ref = pd.read_csv(f"./results/RFtest1/observation.csv")
    results_ref = results_ref[(results_ref['z'] >= z_0) & (results_ref['z'] < z_1)]
    ref_cosmology = get_expectation_1sigam(results_ref, model_name='RFtest1', catalog_name='RFtest1')
    # plot the cosmological parameters vs redshift
    bins_cosmology1, bins_cosmology2 = [], []
    for i in range(len(redshift_bin)-1):
        subset1 = results1[(results1['z'] >= redshift_bin[i]) & (results1['z'] < redshift_bin[i+1])]
        subset2 = results2[(results2['z'] >= redshift_bin[i]) & (results2['z'] < redshift_bin[i+1])]
        if len(subset1) <= 0:
            continue
        cosmology_para1 = get_expectation_1sigam(subset1, model_name1, model_name1, selection_fun=False)
        cosmology_para1['z'] = (redshift_bin[i] + redshift_bin[i+1])/2
        bins_cosmology1.append(cosmology_para1)
        cosmology_para2 = get_expectation_1sigam(subset2, model_name2, model_name2, selection_fun=False)   
        cosmology_para2['z'] = (redshift_bin[i] + redshift_bin[i+1])/2
        bins_cosmology2.append(cosmology_para2) 
    ref_cosmology['z'] = [redshift_bin[0],redshift_bin[-1]]
    bins_cosmology1 = Table(bins_cosmology1)
    bins_cosmology2 = Table(bins_cosmology2)

    
    fig, axes = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    ax1 = axes[0]
    xoffsets = np.linspace(-0.02, 0.02, len(para_names)) 
    for para_name, label, x_off, color in zip(para_names, label_names, xoffsets, colors):
        x = bins_cosmology1['z'].data + x_off
        y_uperr = bins_cosmology1[f'{para_name}_upper_errors']/aveg_cosmology[f'{para_name}_upper_errors']
        y_lowerr = bins_cosmology1[f'{para_name}_lower_errors']/aveg_cosmology[f'{para_name}_lower_errors']
        diff = bins_cosmology1[f'{para_name}_medians'] - aveg_cosmology[f'{para_name}_medians']
        norm_err = np.where(diff > 0, aveg_cosmology[f'{para_name}_upper_errors'], aveg_cosmology[f'{para_name}_lower_errors'])
        y = diff/norm_err
        x_fit = np.linspace(min(bins_cosmology1['z'].data)-0.02, max(bins_cosmology1['z'].data)+0.02, 100)
        ax1.errorbar(x, y, yerr=[y_lowerr, y_uperr], fmt='o', capsize=4, label='Lx<44 '+label, color=color)
    
    for para_name, label, x_off, color in zip(para_names, label_names, xoffsets, colors):
        x = bins_cosmology2['z'].data + x_off + 0.007
        y_uperr = bins_cosmology2[f'{para_name}_upper_errors']/aveg_cosmology[f'{para_name}_upper_errors']
        y_lowerr = bins_cosmology2[f'{para_name}_lower_errors']/aveg_cosmology[f'{para_name}_lower_errors']
        diff = bins_cosmology2[f'{para_name}_medians'] - aveg_cosmology[f'{para_name}_medians']
        norm_err = np.where(diff > 0, aveg_cosmology[f'{para_name}_upper_errors'], aveg_cosmology[f'{para_name}_lower_errors'])
        y = diff / norm_err
        ax1.errorbar(x, y, yerr=[y_lowerr, y_uperr], fmt='x', capsize=4, label='Lx>44 '+label, color=color)

    ax1.axhline(y=0.0, color='black', linestyle=':', linewidth=1)
    ax1.axhline(y=-1.0, color='black', linestyle=':', linewidth=1)
    ax1.axhline(y=1.0, color='black', linestyle=':', linewidth=1)
    ax1.set_ylim(-3, 3)
    ax1.set_xlim(0.1, 1.0)
    ax1.set_ylabel('bias (error) / 1$\sigma$', fontsize=16)
    #ax1.set_title('Cosmological Parameters vs Redshift',fontsize=16)
    ax1.legend(ncol=2, fontsize=12, title_fontsize=12,markerscale=1)

    ax2 = axes[1]
    xoffsets = np.linspace(-0.02, 0.02, len(para_names))
    for para_name, label, x_off, color in zip(para_names, label_names, xoffsets, colors):
        x = bins_cosmology['z'].data + x_off
        y_uperr = bins_cosmology[f'{para_name}_upper_errors']/aveg_cosmology[f'{para_name}_upper_errors']
        y_lowerr = bins_cosmology[f'{para_name}_lower_errors']/aveg_cosmology[f'{para_name}_lower_errors']
        diff = bins_cosmology[f'{para_name}_medians'] - aveg_cosmology[f'{para_name}_medians']
        norm_err = np.where(diff > 0, aveg_cosmology[f'{para_name}_upper_errors'], aveg_cosmology[f'{para_name}_lower_errors'])
        y = diff / norm_err
        ax2.errorbar(x, y, yerr=[y_lowerr, y_uperr], fmt='o', capsize=4, label=label, color=color)

    ref_cosmology = pd.read_csv(f"./simulation_paras.csv")
    ref_cosmology = ref_cosmology[ref_cosmology['name']==simulation_label]
    xoffsets = np.linspace(-0.02, 0.02, len(para_names))
    for para_name, label, x_off, color in zip(para_names, label_names, xoffsets, colors):
        x = bins_cosmology['z'].data + x_off + 0.007
        ref_value = ref_cosmology[f'{para_name}'].values[0]
        y_uperr = bins_cosmology[f'{para_name}_upper_errors']/ref_value
        y_lowerr = bins_cosmology[f'{para_name}_lower_errors']/ref_value
        diff = bins_cosmology[f'{para_name}_medians'] - ref_value
        y = diff / ref_value
        ax2.errorbar(x, y, yerr=[y_lowerr, y_uperr], fmt='x', capsize=4, label=simulation_label+' '+label, color=color)

    ax2.axhline(y=0.0, color='black', linestyle=':', linewidth=1)
    ax2.axhline(y=-1.0, color='black', linestyle=':', linewidth=1)
    ax2.axhline(y=1.0, color='black', linestyle=':', linewidth=1)
    ax2.set_ylim(-3, 3)
    ax2.set_xlim(0.1, 1.0)
    ax2.set_xlabel('$z$', fontsize=16)
    ax2.set_ylabel('bias (error) / 1$\sigma$ (ref)', fontsize=16)
    ax2.legend(ncol=2, fontsize=12, title_fontsize=12, markerscale=1)
    plt.tick_params(axis='both', which='major', labelsize=16)  
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.tight_layout()
    plt.savefig(fname)




def plot_detection_prob_and_cosmology_vs_z(dataset, labels, colors, alphas, model_name, z_bins_edge, fname=''):

    z_0, z_1 = 0.0, 1.0
    fig, axes = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    plt.tick_params(axis='both', which='major', labelsize=16)  
    plt.tick_params(axis='both', which='minor', labelsize=16)
    ax1 = axes[0]
    L_vals = np.logspace(41, 46.0, 100) 
    z_vals = np.linspace(0.01, 1.4, 100)
    L_grid, z_grid = np.meshgrid(L_vals, z_vals)
    S_grid = model_new.selection_function(np.log10(L_grid), z_grid)
    contour = ax1.contour(z_grid, np.log10(L_grid), S_grid, levels=[0.1, 0.3, 0.5, 0.7, 0.9], alpha=1.0, linestyles='-.', colors='black', linewidths=0.6,)
    ax1.clabel(contour, fmt='%.2f')
    for data, label, color, alpha in zip(dataset, labels, colors, alphas):
        ax1.scatter(data['z'], data['L'], color=color, s=2, alpha=alpha, marker='+', label=label)
    data = dataset[0]
    sns.kdeplot(x=data['z'],y=data['L'], ax=ax1, levels=[0.16, 0.50, 0.84],  linewidths=0.6, linestyles='-', fill=False, colors='black')
    ax1.set_ylabel('Log10(L / erg s-1)', fontsize=18)
    ax1.set_xlim(z_0, z_1)
    ax1.set_ylim(41, 46)
    ax1.legend(title='$S(L, z)$',fontsize=14, title_fontsize=16, markerscale=5)

    ax2 = axes[1]
    bins_cosmology, aveg_cosmology = obtain_binned_cosmology(model_name, z_bins_edge)
    para_names = ['Omega', 'Sigm8', 'Hubble', 'OmegaB']
    label_names = ['$\Omega_m$', '$\sigma_8$', '$h_0$', '$\Omega_B$']
    colors = ['blue', 'orange', 'green', 'red']
    xoffsets = np.linspace(-0.02, 0.02, len(para_names))
    for para_name, label, x_off, color in zip(para_names, label_names, xoffsets, colors):
        x = bins_cosmology['z'].data + x_off
        y_uperr = bins_cosmology[f'{para_name}_upper_errors']/aveg_cosmology[f'{para_name}_upper_errors']
        y_lowerr = bins_cosmology[f'{para_name}_lower_errors']/aveg_cosmology[f'{para_name}_lower_errors']
        diff = bins_cosmology[f'{para_name}_medians'] - aveg_cosmology[f'{para_name}_medians']
        norm_err = np.where(diff > 0, aveg_cosmology[f'{para_name}_upper_errors'], aveg_cosmology[f'{para_name}_lower_errors'])
        y = diff / norm_err
        ax2.errorbar(x, y, yerr=[y_lowerr, y_uperr], fmt='o', capsize=4, label=label, color=color)
    ax2.axhline(y=0.0, color='black', linestyle=':', linewidth=1)
    ax2.axhline(y=-1.0, color='black', linestyle=':', linewidth=1)
    ax2.axhline(y=1.0, color='black', linestyle=':', linewidth=1)
    ax2.set_xlabel('$z$', fontsize=18)
    ax2.set_ylabel('bias (error) / 1$\sigma$', fontsize=18)
    ax2.legend(title='Cosmological',fontsize=14, title_fontsize=16, markerscale=1)
    plt.tight_layout()
    if fname == '':
        fname = f'./figures/test_detection_prob_cosmology_vs_z.pdf'
    plt.savefig(fname)



def plot_simulation_detection_prob_and_cosmology_vs_z(simulation_label, model_name, z_bins_edge, fname=''):
    observed_data = preprocess.observed_data()
    simulation_results = pd.read_csv(f"./results/{model_name}/mix_simulations.csv")
    simulation_results = simulation_results[simulation_results['label']==simulation_label]
    dataset = [observed_data, simulation_results]
    labels = ['eFEDS+DR1', f'{simulation_label}']
    colors = ['orange', 'blue']
    alphas = [1.0, 0.3]
    z_0, z_1 = 0.0, 1.0
    fig, axes = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    plt.tick_params(axis='both', which='major', labelsize=16)  
    plt.tick_params(axis='both', which='minor', labelsize=16)
    ax1 = axes[0]
    L_vals = np.logspace(41, 46.0, 100) 
    z_vals = np.linspace(0.01, 1.4, 100)
    L_grid, z_grid = np.meshgrid(L_vals, z_vals)
    S_grid = model_new.selection_function(np.log10(L_grid), z_grid)
    contour = ax1.contour(z_grid, np.log10(L_grid), S_grid, levels=[0.1, 0.3, 0.5, 0.7, 0.9], alpha=1.0, linestyles='-.', colors='black', linewidths=0.6,)
    ax1.clabel(contour, fmt='%.2f')
    
    for data, label, color, alpha in zip(dataset, labels, colors, alphas):
        ax1.scatter(data['z'], data['L'], color=color, s=2, alpha=alpha, marker='+', label=label)
    data = dataset[0]
    sns.kdeplot(x=data['z'],y=data['L'], ax=ax1, levels=[0.16, 0.50, 0.84],  linewidths=0.6, linestyles='-', fill=False, colors='black')
    ax1.set_ylabel('Log10(L / erg s-1)', fontsize=18)
    ax1.set_xlim(z_0, z_1)
    ax1.set_ylim(41, 46)
    ax1.legend(title='$S(L, z)$',fontsize=14, title_fontsize=16, markerscale=5)

    ax2 = axes[1]
    bins_cosmology, aveg_cosmology = obtain_binned_cosmology(model_name, z_bins_edge, simulation=True, simulation_label=simulation_label)
    ref_cosmology = pd.read_csv(f"./simulation_paras.csv")
    ref_cosmology = ref_cosmology[ref_cosmology['name']==simulation_label]
    para_names = ['Omega', 'Sigm8', 'Hubble', 'OmegaB']
    label_names = ['$\Omega_m$', '$\sigma_8$', '$h_0$', '$\Omega_B$']
    colors = ['blue', 'orange', 'green', 'red']
    xoffsets = np.linspace(-0.02, 0.02, len(para_names))
    for para_name, label, x_off, color in zip(para_names, label_names, xoffsets, colors):
        x = bins_cosmology['z'].data + x_off
        ref_value = ref_cosmology[f'{para_name}'].values[0]
        y_uperr = bins_cosmology[f'{para_name}_upper_errors']/ref_value
        y_lowerr = bins_cosmology[f'{para_name}_lower_errors']/ref_value
        diff = bins_cosmology[f'{para_name}_medians'] - ref_value
        y = diff / ref_value
        ax2.errorbar(x, y, yerr=[y_lowerr, y_uperr], fmt='o', capsize=4, label=label, color=color)
    ax2.axhline(y=0.0, color='black', linestyle=':', linewidth=1)
    ax2.axhline(y=-1.0, color='black', linestyle=':', linewidth=1)
    ax2.axhline(y=1.0, color='black', linestyle=':', linewidth=1)
    ax2.set_xlabel('$z$', fontsize=18)
    ax2.set_ylabel('bias (error) / ref_value', fontsize=18)
    ax2.legend(title=f'Simulation {simulation_label} Cosmological',fontsize=14, title_fontsize=16, markerscale=1)
    plt.tight_layout()
    if fname == '':
        fname = f'./figures/test_simulation_detection_prob_cosmology_vs_z.pdf'
    plt.savefig(fname)
    


def plot_cosmology_paras_vs_test(model_names, show_name=[], keff=[], fname='./figures/paras_vs_test.pdf', z_0=0.1, z_1=0.8, L0=None, significant='results/sinificants.fits', chi2_range=None, chi2_opt='normal_chi2', ref_chi_name='ref'):
    cosmology_paras = []
    signif, dof = [], []
    significant = Table.read(significant)
    for model_name, k in zip(model_names, keff):
        if model_name == 'synthesis':
            results = pd.read_csv(f"./results/RFtest1/observation.csv")
        else:
            results = pd.read_csv(f"./results/{model_name}/observation.csv")
        if L0 is not None:
            subset = results[(results['z'] >= z_0) & (results['z'] < z_1) & (results['L']>=L0)]
        else:
            subset = results[(results['z'] >= z_0) & (results['z'] < z_1)]
        paras = get_expectation_1sigam(subset, model_name, 'observation')
        cosmology_paras.append(paras)
        signif.append(significant[significant['model_name']==model_name]['chi2'].data)
        dof.append(significant[significant['model_name']==model_name]['dof'].data - k)
    ref_cosmology = planck18_cosmology
    cosmology_paras = Table(cosmology_paras)
    subset_significant = np.concatenate(signif)
    subset_dof = np.concatenate(dof)
    subset_significant = subset_significant/subset_dof
    ref_chi2 = significant[significant['model_name']==ref_chi_name]['chi2'].data/significant[significant['model_name']==ref_chi_name]['dof'].data
    ref_chi2 = ref_chi2[0]
    fic_chi2 = subset_significant[0]
    if chi2_opt == 'new_chi2':
        subset_significant = (ref_chi2-subset_significant)/(ref_chi2-fic_chi2)
    elif chi2_opt == 'normal_chi2':
        subset_significant = subset_significant/fic_chi2
    else:
        raise ValueError('chi2_opt should be new_chi2 or normal_chi2')
    para_names = ['Omega', 'Sigm8', 'Hubble', 'OmegaB']
    label_names = ['$\Omega_m$', '$\sigma_8$', '$h_0$', '$\Omega_B$']

    fig, axes = plt.subplots(nrows=1, ncols=5, sharey=True, figsize=(12, 8), gridspec_kw={'wspace': 0})
    for para_name, label, ax in zip(para_names, label_names, axes[0:4]):
        test_num = np.arange(len(cosmology_paras))
        value = cosmology_paras[f'{para_name}_medians']
        value_err = np.array([cosmology_paras[f'{para_name}_lower_errors'], cosmology_paras[f'{para_name}_upper_errors']])
        ref_value = ref_cosmology[f'{para_name}_medians']
        ref_low_error = ref_cosmology[f'{para_name}_lower_errors']
        ref_up_error = ref_cosmology[f'{para_name}_upper_errors']
        ax.errorbar(value, test_num, xerr=value_err, fmt='o', capsize=4)
        h0, err = ref_value, 0.003
        ax.fill_betweenx([-0.1, len(cosmology_paras)-0.1], ref_value - ref_low_error, ref_value + ref_up_error, color='orange', alpha=0.4, label='Planck2018, 1$\sigma$')
        ax.set_xlabel(label)
    
    axes[-1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='best fits')
    axes[-1].axvline(x=1, color='r', linestyle='--', linewidth=2,)
    axes[-1].barh(test_num, subset_significant, color="skyblue")
    #axes[-1].set_xlabel(f'normalized $\chi^2$')
    axes[-1].set_xlabel(f'Significance')
    if chi2_range is not None:
        axes[-1].set_xlim(chi2_range[0], chi2_range[1]) 

    
    if len(show_name) == 0:
        show_name = model_names
    axes[0].legend(loc='best')
    plt.yticks(ticks=np.arange(len(cosmology_paras)), labels=show_name)
    plt.tight_layout()
    plt.savefig(fname)
    return cosmology_paras



def plot_distribution_selection_fun(dataset, model_name, simulation_label, labels, colors, col_rangs, fname=''):
    vars = col_rangs.keys()
    xrangs = col_rangs.values()
    f_size = 14
    fig, axs = plt.subplots(1, len(col_rangs.keys())+1, figsize=(4*len(col_rangs.keys()), 4), dpi=160)
    units = ["Log10(R/kpc)", "Log10(M/M_sun)", "Log10(L/erg s-1)", "Log10(T/keV)", "z"]
    for var, ax, xrang, index in zip(vars, axs[1:], xrangs, range(len(vars))):
        bw = (xrang[1] - xrang[0])/20
        for label, data, color in zip(labels, dataset, colors):
            if index == 4:
                sns.histplot(data[var], ax=ax, kde=False, label=label, palette='dark', color=color, stat="density", alpha=0.4, binwidth=bw)
                ax.set_ylabel('Density', fontsize=f_size)
                ax.yaxis.set_label_position("right")
                ax.legend(framealpha=0.1, fontsize=f_size, loc='upper right')
            else:
                sns.histplot(data[var], ax=ax, kde=False, label=None, palette='dark', color=color, stat="density", alpha=0.4, binwidth=bw)
                ax.set_ylabel(None)
        ax.set_yscale('log')
        x0, x1 = np.round(xrang[0],1), np.round(xrang[1],1)
        #ax.set_title(f'{var} range: {x0} -- {x1}')
        
    for ax, unit in zip(axs[1:], units):
        ax.set_xlabel(unit)
    
    observed_data = preprocess.observed_data()
    simulation_results = pd.read_csv(f"./results/{model_name}/mix_simulations.csv")
    simulation_results = simulation_results[simulation_results['label']==simulation_label]
    dataset = [observed_data, simulation_results]
    dataset = [observed_data]
    labels = ['eFEDS+eRASS1', f'{simulation_label}']
    colors = ['orange', 'blue']
    alphas = [1.0, 0.3]
    z_0, z_1 = 0.0, 1.0
    ax1 = axs[0]
    L_vals = np.logspace(41, 46.0, 100) 
    z_vals = np.linspace(0.01, 1.4, 100)
    L_grid, z_grid = np.meshgrid(L_vals, z_vals)
    S_grid = model_new.selection_function(np.log10(L_grid), z_grid)
    contour = ax1.contour(z_grid, np.log10(L_grid), S_grid, levels=[0.1, 0.3, 0.5, 0.7, 0.9], alpha=1.0, linestyles='-.', colors='black', linewidths=0.6,)
    ax1.clabel(contour, fmt='%.2f')
    for data, label, color, alpha in zip(dataset, labels, colors, alphas):
        ax1.scatter(data['z'], data['L'], color=color, s=2, alpha=alpha, marker='+', label=label)
    data = dataset[0]
    sns.kdeplot(x=data['z'],y=data['L'], ax=ax1, levels=[0.16, 0.50, 0.84],  linewidths=0.6, linestyles='-', fill=False, colors='black')
    ax1.set_ylabel('Log10(L / erg s-1)', fontsize=18)
    ax1.set_xlim(z_0, z_1)
    ax1.set_ylim(41, 46)
    ax1.legend(title='$S(L, z)$',fontsize=14, title_fontsize=16, markerscale=5)
    plt.tick_params(axis='both', which='major', labelsize=16)  
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.xticks(fontsize=f_size)
    plt.yticks(fontsize=f_size)
    if fname == '':
        fname = f'./figures/{fname}_distr.pdf'
    plt.subplots_adjust(wspace=0.00, hspace=0.00)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)